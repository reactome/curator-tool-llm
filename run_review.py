"""Second opinion on a finished run, from a different model.

Extraction and merge are both Claude judging its own work. This shows the same material
to OpenAI — the chunks that went in, the reactions that came out — and asks for a plain
subjective read on whether the extraction and the merge were done well.

It is a reviewer, not a scorer: the output is prose a curator can skim. The one number it
does return is a gate — score below --threshold and it runs a second, per-chunk pass
looking for reactions the pipeline walked past. That pass is the expensive half, so a run
that already looks sound does not pay for it.

Usage:
  python run_review.py                          # defaults to PINK1
  python run_review.py znfx1                    # -> results/znfx1_merged.json
  python run_review.py results/znfx1_merged.json
  python run_review.py znfx1 --audit always     # sweep for missed reactions regardless
  python run_review.py znfx1 --threshold 8      # sweep unless it scores 8+/10

Writes results/<stem>_review.md and prints it.

CHUNKS come from results/<stem>_sections.log when check_sections.py has been run for this
paper — that log holds the exact chunks, and each header carries a character count this
script checks the parse against. Without it, they are re-derived through the pipeline's
own fetcher and splitter object, which costs a re-fetch.
"""
import os, sys, re, json, time, argparse
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import PubMedFetcher as fetcher

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODEL = 'gpt-5.6-luna'

parser = argparse.ArgumentParser(description='Get a second opinion on a merged run.')
parser.add_argument('target', nargs='?', default='pink1',
                    help='gene prefix (e.g. znfx1) or path to a *_merged.json file')
parser.add_argument('--model', default=MODEL, help=f'OpenAI model (default: {MODEL})')
parser.add_argument('--audit', choices=['auto', 'always', 'never'], default='auto',
                    help='the missed-reaction sweep: auto runs it only below --threshold')
parser.add_argument('--threshold', type=float, default=6.0,
                    help='score out of 10 at or above which the sweep is skipped (default: 6)')
parser.add_argument('--sections-log', help='chunk log to read (default: results/<stem>_sections.log)')
parser.add_argument('--workers', type=int, default=8, help='concurrent sweep calls')
parser.add_argument('--out', help='output path (default: results/<stem>_review.md)')
args = parser.parse_args()

cand = args.target if os.path.isabs(args.target) else os.path.join(PROJECT_ROOT, args.target)
if args.target.endswith('.json') or os.path.isfile(cand):
    MERGED = cand
else:
    MERGED = os.path.join(RESULTS_DIR, f'{args.target.lower()}_merged.json')
if not os.path.isfile(MERGED):
    sys.exit(f"[error] merged file not found: {MERGED}")
OUT = args.out or os.path.splitext(MERGED)[0].replace('_merged', '') + '_review.md'

merged = json.load(open(MERGED))
if not merged:
    sys.exit(f"[error] {MERGED} holds no reactions")
source_id = merged[0].get('source', '')
# source_id round-trips for a PDF ("ZNFX1.pdf") but a PMID comes back decorated
spec = source_id.split(':', 1)[1] if source_id.upper().startswith('PMID:') else source_id
print(f"[setup] merged: {MERGED}  ({len(merged)} reactions from {source_id})")


# ── chunks: read the log if it is there, otherwise re-derive ──────────────────
def chunks_from_log(path):
    """Parse PART 3 of a check_sections.py log back into the chunk list.

    Each header carries the chunk's character count, so the parse is checked rather than
    trusted: any mismatch means the log format moved and the caller should re-derive
    instead of reviewing against text that is subtly not what the extractor saw."""
    text = open(path).read()
    part3 = text.find('PART 3')
    if part3 == -1:
        return None, 'no PART 3 section'
    blocks = re.split(r'^--- chunk (\d+) \(([\d,]+) chars[^)]*\) ---\n',
                      text[part3:], flags=re.M)
    if len(blocks) < 4:
        return None, 'no chunk blocks found'
    out = []
    # blocks = [preamble, idx, chars, body, idx, chars, body, ...]
    for i in range(1, len(blocks) - 2, 3):
        idx, want, body = int(blocks[i]), int(blocks[i + 1].replace(',', '')), blocks[i + 2]
        # the last block runs into the trailing token-spread summary — cut that off first,
        # because doing it after the newline trim below leaves the summary's own newline behind
        cut = body.find('\n' + '=' * 70)
        if cut != -1:
            body = body[:cut]
        # the writer does print(chunk) then print(), so the body carries newlines the chunk
        # itself never had: two, or one on the final block where the cut consumed the other
        if body.endswith('\n\n'):
            body = body[:-2]
        elif body.endswith('\n'):
            body = body[:-1]
        if len(body) != want:
            return None, f'chunk {idx}: log says {want} chars, parsed {len(body)}'
        if idx != len(out):
            return None, f'chunk numbering jumped at {idx}'
        out.append(body)
    return out, f'{len(out)} chunks'


SECTIONS_LOG = args.sections_log or os.path.join(
    RESULTS_DIR, f'{fetcher.output_stem(spec)}_sections.log')
chunks, how = None, ''
if os.path.isfile(SECTIONS_LOG):
    chunks, note = chunks_from_log(SECTIONS_LOG)
    if chunks:
        how = f'{os.path.basename(SECTIONS_LOG)} ({note})'
        print(f"[chunks] read from {SECTIONS_LOG} — {note}")
    else:
        print(f"[chunks] {os.path.basename(SECTIONS_LOG)} did not parse ({note}) — re-deriving")
else:
    print(f"[chunks] no {os.path.basename(SECTIONS_LOG)} — re-deriving "
          f"(run check_sections.py {spec} once to skip this)")

if not chunks:
    # the splitter OBJECT, not a copy of its settings, so this cannot drift out of step
    # with run_extraction.py. Its CLI is behind an __main__ guard, so importing is cheap.
    import run_extraction as pipeline
    _sid, results_text, route = fetcher.load_source(spec, client=pipeline.client,
                                                    model=pipeline.MODEL_NAME)
    chunks = pipeline.splitter.split_text(results_text)
    how = f're-derived from source (via {route})'
    print(f"[chunks] {len(results_text.split()):,} words (via {route}) -> {len(chunks)} chunks")

# ── OpenAI ────────────────────────────────────────────────────────────────────
if not os.getenv('OPENAI_API_KEY'):
    sys.exit("[error] OPENAI_API_KEY is not set. Add it to ~/curator-tool-llm/.env as\n"
             "        OPENAI_API_KEY=sk-...\n"
             "        (an `export` in another terminal does not reach this process)")
from openai import OpenAI
oai = OpenAI(timeout=300.0, max_retries=2)

CONVENTIONS = """Two Reactome conventions worth holding them to:
- the CATALYST directly performs the chemistry. An entity that facilitates, promotes, \
enables or is merely required for a reaction is a REGULATOR, not a catalyst. Transport, \
binding, translocation and conformational change usually have no catalyst at all.
- regulation is additive across experiments — two records testing different regulators of \
the same event are still the same event, and merging them is correct."""


def brief(e, i):
    """One reaction, trimmed to what a reviewer needs to judge it."""
    a = e['annotation_result']
    ca = a.get('catalystActivity') or {}
    regs = a.get('regulatedBy') or []
    s = a.get('summation')
    summ = s.get('text') if isinstance(s, dict) else (
        ' '.join(x.get('text', '') for x in s if isinstance(x, dict)) if isinstance(s, list) else '')
    names = a.get('merged_names') or []
    lines = [f"### {i}. {a.get('name', '')}",
             f"- type: {a.get('reactionType', '')}",
             f"- input: {', '.join(a.get('input') or []) or 'none'}",
             f"- output: {', '.join(a.get('output') or []) or 'none'}",
             f"- catalyst: {ca.get('catalyst') or 'none'} ({ca.get('molecularFunction') or 'none'})",
             "- regulation: " + (', '.join(
                 f"{r.get('regulationType')} by {r.get('regulator')}"
                 + (f" [note: {r.get('note')}]" if r.get('note') else '')
                 for r in regs if isinstance(r, dict)) or 'none'),
             f"- compartment: {a.get('compartment') or 'none'}",
             f"- condition: {a.get('condition') or 'none'}",
             f"- summation: {summ or 'none'}"]
    if len(names) > 1:
        lines.append(f"- MERGED from {len(names)} separately extracted variants: "
                     + '; '.join(names))
    for q in a.get('evidence') or []:
        lines.append(f'- evidence: "{q}"')
    return '\n'.join(lines)


REVIEW_SYSTEM = f"""You are an experienced Reactome biocurator giving a colleague a second \
opinion. Another AI read this paper's Results section in chunks and pulled reaction \
records out of it, then merged the ones it thought were duplicates. You are seeing the \
same source text it saw and the records it produced.

You are saying whether you would trust this output and where you would look twice. Be \
direct about what is wrong and equally direct about what is fine.

Judge against the source text only. Something the paper does not say is not supported, \
however plausible it is from what you know about the biology. Equally, do not fault a \
record for something the paper does not say either — if the source is vague about a \
compartment or a stoichiometry, a record that is vague in the same way is correct, not \
incomplete.

Be objective, not exacting. Every criticism must point at a specific record by number and \
quote the text that contradicts it or the text it left out. If you cannot point at the \
text, it is a preference, not a defect — either drop it or label it as a preference. Do \
not invent a house style and grade against it: naming, wording, and level of detail are \
only problems when they change what the record means.

Calibrate to what this output is for. It is a STARTING POINT a curator will edit, not a \
finished Reactome record. Judge it the way you would judge a competent junior curator's \
first pass:
- a real defect changes the biology: a participant the paper never names, a catalyst that \
is really a regulator, a direction reversed, two distinct events fused, one event split in \
two, a claim with no support in the source.
- not a defect: imprecise-but-correct naming, a summation you would have phrased \
differently, a missing detail the paper does not supply, a judgment call you would have \
made the other way but that the text can support.

Score on this scale, and use the middle of it — most competent output lands at 6-8:
- 9-10: you would hand it to a curator as-is; nothing you found changes the biology.
- 7-8: sound. Real biology, correctly merged, with small fixes a curator makes in minutes.
- 5-6: usable but needs work — one or two records are wrong or a real event was missed.
- 3-4: several records misrepresent the paper, or the merge is substantially wrong.
- 0-2: faster to start over.
Reserve scores below 5 for output with defects of the first kind above. A tidy, accurate \
extraction does not lose points for being less thorough than you would have been.

{CONVENTIONS}"""

REACTIONS_MD = '\n\n'.join(brief(e, i) for i, e in enumerate(merged, 1))
CHUNKS_MD = '\n'.join(f'--- chunk {i} ---\n{c}\n' for i, c in enumerate(chunks))

review_user = f"""## SOURCE TEXT — the Results section, in the chunks the extractor saw

{CHUNKS_MD}

## FINAL OUTPUT — {len(merged)} reactions after extraction and merging

{REACTIONS_MD}

## Your review

Start with a single line, exactly:

SCORE: <0-10>

That is your overall confidence in this output as a starting point for curation, on the \
scale you were given. Then, in markdown, for someone who will read it in two minutes.

Show your work: every point you make gets one concrete example, named by record number \
or chunk number and quoted. A general statement with no example attached is not useful to \
a curator and should be cut. Give positive examples as well as negative ones — where the \
output got something right, say which record and why it is right. A section with nothing \
wrong in it should say so and show the best example, not manufacture a complaint.

**Overall** — would you trust this? Two or three sentences.

**Extraction** — did it catch the real biology in this paper? Call out anything it \
invented or overstated, and anything real it walked past, naming the chunk and quoting \
the sentence. Then give one example of a record it got right, and say what makes it right \
— the source sentence it rests on.

**Merging** — did it fuse records that are genuinely the same event, and leave apart the \
ones that differ? Flag anything wrongly merged or wrongly kept separate, naming the \
records. Also name one merge (or one deliberate non-merge) it called correctly.

**Look twice at these** — the specific records you would check by hand first, and why, \
each with the quote that raised the question. Only records where the biology may actually \
be wrong — not records you would word differently. Skip this heading entirely if there \
aren't any; do not pad it."""

print(f"[review] asking {args.model} ({len(review_user):,} chars of context)...", flush=True)
t0 = time.time()
resp = oai.chat.completions.create(
    model=args.model,
    messages=[{'role': 'system', 'content': REVIEW_SYSTEM},
              {'role': 'user', 'content': review_user}])
review = (resp.choices[0].message.content or '').strip()
u = getattr(resp, 'usage', None)
tok_in = (u.prompt_tokens if u else 0)
tok_out = (u.completion_tokens if u else 0)
print(f"[review] done in {time.time()-t0:.0f}s")

m = re.search(r'SCORE:\s*([\d.]+)', review)
score = float(m.group(1)) if m else None
print(f"[review] score: {score if score is not None else 'not reported'}/10")

# ── gated sweep: only when the review says the output is shaky ────────────────
# A low score means the reviewer already doubts the output; that is when it is worth
# paying one call per chunk to ask the harder question the review cannot answer from a
# summary — what is in the paper that never made it out at all.
if args.audit == 'always':
    do_sweep, why = True, '--audit always'
elif args.audit == 'never':
    do_sweep, why = False, '--audit never'
elif score is None:
    do_sweep, why = False, 'no score was reported — pass --audit always to force it'
else:
    do_sweep = score < args.threshold
    why = (f'score {score} < threshold {args.threshold}' if do_sweep
           else f'score {score} >= threshold {args.threshold}')

SWEEP_SYSTEM = f"""You are auditing an automated Reactome biocuration pipeline for RECALL. \
Another model read this chunk and the rest of the paper, and produced the final reaction \
list you are given. Your only job is to find reactions THIS CHUNK states that are missing \
from that list.

A reaction qualifies only if the chunk itself describes a molecular event with \
identifiable participants — a modification, binding, dissociation, transport, cleavage, \
degradation or conformational change. These do NOT qualify:
- a phenotype, a measurement, a rate, a level, or a correlation
- a method, construct, mutant or assay described for its own sake
- background or a hypothesis the authors state without showing it
- a rewording of a reaction already in the list, at any level of detail or granularity

Return an empty list when nothing is missing. That is the expected answer for most chunks, \
and a padded list is worse than an empty one — every false positive costs a curator time \
to dismiss.

{CONVENTIONS}"""

SWEEP_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['missed'],
    'properties': {'missed': {
        'type': 'array',
        'items': {'type': 'object', 'additionalProperties': False,
                  'required': ['name', 'quote', 'why'],
                  'properties': {
                      'name': {'type': 'string',
                               'description': 'the reaction as <entity> <action> <entity>'},
                      'quote': {'type': 'string',
                                'description': 'the verbatim sentence from this chunk'},
                      'why': {'type': 'string',
                              'description': 'under 20 words: why it is a reaction and why '
                                             'no entry in the list already covers it'}}}}}}

FINAL_NAMES = '\n'.join(f"  - {e['annotation_result'].get('name', '')}" for e in merged)
sweep_md = ''
if do_sweep:
    print(f"\n[sweep] {why} — checking all {len(chunks)} chunks for missed reactions, "
          f"{args.workers} concurrent...", flush=True)

    def sweep(i):
        for attempt in (1, 2):
            try:
                r = oai.chat.completions.create(
                    model=args.model,
                    response_format={'type': 'json_schema',
                                     'json_schema': {'name': 'missed_reactions',
                                                     'strict': True, 'schema': SWEEP_SCHEMA}},
                    messages=[{'role': 'system', 'content': SWEEP_SYSTEM},
                              {'role': 'user', 'content':
                                  f"CHUNK {i}:\n{chunks[i]}\n\n"
                                  f"THE PIPELINE'S FINAL REACTION LIST "
                                  f"(the whole paper, {len(merged)} reactions):\n{FINAL_NAMES}\n\n"
                                  f"What does chunk {i} state that this list is missing?"}])
                got = json.loads(r.choices[0].message.content or '{}')
                return i, got.get('missed') or [], getattr(r, 'usage', None), None
            except Exception as ex:
                if attempt == 2:
                    return i, [], None, f'{type(ex).__name__}: {str(ex)[:120]}'

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(sweep, range(len(chunks))))
    for _, _, usage, _ in results:
        if usage:
            tok_in += usage.prompt_tokens
            tok_out += usage.completion_tokens
    errs = [(i, e) for i, _, _, e in results if e]
    hits = [(i, ms) for i, ms, _, e in results if ms and not e]
    n = sum(len(ms) for _, ms in hits)
    print(f"[sweep] {len(chunks)} chunks in {time.time()-t1:.0f}s — {n} candidate miss(es)"
          + (f", {len(errs)} chunk(s) failed" if errs else ''))

    lines = [f"\n---\n\n## Possible missed reactions\n",
             f"_Swept because {why}. One pass per chunk against the final list of "
             f"{len(merged)} reactions. These are candidates, not findings — the reviewer "
             f"was told to return nothing when nothing is missing._\n"]
    if n:
        for i, ms in hits:
            for x in ms:
                lines += [f"**chunk {i} — {x.get('name', '')}**  ",
                          f"{x.get('why', '')}  ",
                          f"> {x.get('quote', '')}\n"]
    else:
        lines.append("None. Every reaction the chunks state is already in the output.\n")
    if errs:
        lines.append(f"\n_{len(errs)} chunk(s) failed and were not swept: "
                     + ', '.join(f'{i} ({e})' for i, e in errs) + "_\n")
    sweep_md = '\n'.join(lines)
else:
    print(f"\n[sweep] skipped — {why}")
    sweep_md = f"\n---\n\n_Missed-reaction sweep skipped: {why}._\n"

header = (f"# Second opinion — {source_id}\n\n"
          f"reviewer: {args.model} | reviewed: {os.path.basename(MERGED)} "
          f"({len(merged)} reactions) | chunks: {how}\n\n---\n\n")
with open(OUT, 'w') as f:
    f.write(header + review + '\n' + sweep_md)

print('\n' + review)
if sweep_md:
    print(sweep_md)
print(f"[usage] tokens in {tok_in:,} / out {tok_out:,}")
print(f"[done] -> {OUT}")
