"""Exhaustive semantic merger for extracted reactions from any PDF.

Four stages, so that no single judgement can silently destroy a reaction:

  1) pairwise candidate generation — dedup to unique signatures, then LLM-judge every
     same-source pair. Produces same/different edges, saved alongside the output.
  2) conflict-aware clustering — a cluster may contain NO pair judged different, i.e.
     every cluster is a clique in the "same" graph. Transitive closure over the same
     edges would merge A with C on the strength of A~B and B~C even when A-vs-C was
     judged different; on ZNFX1 that collapsed three distinct mechanistic steps into
     one 11-variant reaction.
  3) cluster-level validation — the pairwise judge only ever sees two records, so it cannot
     see that a cluster spans successive steps of one mechanism, or that a regulatory
     statement got folded into the reaction it regulates. Each cluster at or above
     --validate-min-size is shown to the judge WHOLE and asked the Reactome-specific
     question: is this one reaction? The stage may only SPLIT a cluster, never join.
  4) subsection consolidation — fold a surviving cluster's repeated wording into one
     non-repeating record per reaction.

Usage:
    python run_merge.py                                     # defaults to PINK1
    python run_merge.py znfx1                               # gene prefix -> results/znfx1_extraction.json
    python run_merge.py results/znfx1_extraction.json   # explicit input path
    python run_merge.py znfx1 --out results/znfx1_custom_merged.json
    python run_merge.py znfx1 --no-validate                 # skip stage 3
"""
import os, sys, json, re, time, argparse, itertools
from concurrent.futures import ThreadPoolExecutor
PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)
import anthropic

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-5'

# Token accounting over every judge/consolidate call, so a run can be costed. The
# judge fans out over ThreadPoolExecutor, so the counters are lock-guarded.
USAGE = {'calls': 0, 'in': 0, 'out': 0, 'cache_read': 0, 'cache_write': 0}
_usage_lock = __import__('threading').Lock()


def track_usage(msg):
    u = getattr(msg, 'usage', None)
    if not u:
        return
    with _usage_lock:
        USAGE['calls'] += 1
        USAGE['in'] += getattr(u, 'input_tokens', 0) or 0
        USAGE['out'] += getattr(u, 'output_tokens', 0) or 0
        USAGE['cache_read'] += getattr(u, 'cache_read_input_tokens', 0) or 0
        USAGE['cache_write'] += getattr(u, 'cache_creation_input_tokens', 0) or 0


def usage_line():
    cache = ''
    if USAGE['cache_read'] or USAGE['cache_write']:
        cache = f", cache r{USAGE['cache_read']:,}/w{USAGE['cache_write']:,}"
    return (f"{USAGE['calls']} call(s), tokens in {USAGE['in']:,} / out "
            f"{USAGE['out']:,}{cache}")


def resolve_paths(target, out=None):
    """Accept either an extraction-file path or a gene prefix and return (IN, OUT)."""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    cand = target if os.path.isabs(target) else os.path.join(PROJECT_ROOT, target)
    if target.endswith('.json') or os.path.isfile(cand):
        in_path = cand
    else:
        # treat as a gene prefix, e.g. "znfx1" -> results/znfx1_extraction.json
        in_path = os.path.join(results_dir, f'{target.lower()}_extraction.json')
    if out:
        out_path = out if os.path.isabs(out) else os.path.join(PROJECT_ROOT, out)
    elif '_extraction' in in_path:
        out_path = in_path.replace('_extraction', '_merged')
    else:
        base, ext = os.path.splitext(in_path)
        out_path = f'{base}_merged{ext}'
    return in_path, out_path


parser = argparse.ArgumentParser(description='Semantically merge extracted reactions.')
parser.add_argument('target', nargs='?', default='pink1',
                    help='gene prefix (e.g. znfx1) or path to an *_extraction.json file (default: pink1)')
parser.add_argument('--out', help='output path (default: input with _extraction -> _merged)')
parser.add_argument('--validate-min-size', type=int, default=3,
                    help='run stage-3 cluster validation on clusters of at least this many '
                         'variants; a 2-variant cluster rests on one direct verdict and has no '
                         'transitivity to check (default: 3)')
parser.add_argument('--no-validate', action='store_true',
                    help='skip stage-3 cluster-level validation')
parser.add_argument('--revote-below', type=float, default=0.9,
                    help='re-judge any pair whose verdict came back below this confidence two '
                         'more times and take the majority of the three; without it one '
                         'unstable verdict vetoes a whole group (default: 0.9, 0 disables)')
args = parser.parse_args()

IN, OUT = resolve_paths(args.target, args.out)
if not os.path.isfile(IN):
    sys.exit(f"[error] extraction file not found: {IN}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(f"[setup] input : {IN}", flush=True)
print(f"[setup] output: {OUT}", flush=True)

extraction_results = json.load(open(IN))

def _norm(x): return (x or '').strip().lower()
def _sig(a):
    ins  = tuple(sorted(_norm(i) for i in (a.get('input')  or [])))
    outs = tuple(sorted(_norm(o) for o in (a.get('output') or [])))
    return (_norm(a.get('name')), ins, outs)

# 1) collapse exact duplicates to unique signatures
_seen, unique = set(), []
for e in extraction_results:
    a = e.get('annotation_result') or {}
    s = _sig(a)
    if s and s not in _seen:
        _seen.add(s); unique.append(e)

print(f"[setup] {len(extraction_results)} entries -> {len(unique)} unique signatures", flush=True)
print("[setup] unique reactions BEFORE semantic merge:", flush=True)
for i, e in enumerate(unique, 1):
    print(f"   {i:2d}. {e['annotation_result'].get('name','')}", flush=True)

def _reaction_brief(a):
    ca = a.get('catalystActivity') or {}
    regs = a.get('regulatedBy') or []
    # the note carries any condition-dependence flag already recorded, so the judge can use it
    # instead of re-deriving why one regulator appears in two directions
    reg_str = ', '.join(
        f"{r.get('regulationType')} by {r.get('regulator')}"
        + (f" [note: {r.get('note')}]" if r.get('note') else '')
        for r in regs) if regs else 'none'
    return (f"name: {a.get('name','')}\n"
            f"type: {a.get('reactionType','')}\n"
            f"input: {', '.join(a.get('input',[]) or []) or 'none'}\n"
            f"output: {', '.join(a.get('output',[]) or []) or 'none'}\n"
            f"catalyst: {ca.get('catalyst') or 'none'} ({ca.get('molecularFunction') or 'none'})\n"
            f"regulation: {reg_str}\n"
            f"condition: {a.get('condition') or 'none'}\n"
            f"compartment: {a.get('compartment') or 'none'}\n"
            f"relationships: {' | '.join(a.get('relationships',[]) or []) or 'none'}")

# pairs whose verdict we never got — they default to NOT duplicate, so the
# final count may under-merge. Reported at the end rather than lost in the log.
failures = []

def _conf(x):
    """Judge confidence clamped to 0-1, or None when the model returned none/garbage."""
    try:
        return round(min(1.0, max(0.0, float(x))), 3)
    except (TypeError, ValueError):
        return None


def reactions_are_duplicate(a1, a2):
    """Return (is_duplicate, confidence). confidence is None when the judge gave no usable
    number — including every failed verdict, which must not look like a confident 'no'."""
    prompt = f"""You are a Reactome biocurator deduplicating extracted reactions.
Two records are DUPLICATES if they describe the SAME underlying biochemical event — the same
molecular transformation of the same participants — EVEN IF they differ in:
  - wording or synonyms
  - level of detail / granularity (e.g. one adds a qualifier like "leading to degradation")
  - reactionType label (e.g. one tagged "blackBoxEvent", the other "transition")
They are DIFFERENT reactions — do NOT merge them — if ANY of the following differs:
  - the participants (input) or the products (output)
  - the catalyst, A catalyst is the entity whose activity
    performs the input-to-output conversion; an entity that acts upstream, or that modulates
    whether or how fast the event happens, is REGULATION, not a catalyst — so a record naming
    X as catalyst and one naming X only as a regulator do not agree on the catalyst either.
  - a different regulationType on the SAME regulator (one says positiveRegulation by X, the
    other negativeRegulation by X) — that is a genuine contradiction, not extra detail.
    When the difference is due to conditional circumstances, such as a different experimental
    condition, add a "note" on that regulation flagging it for the curator.
  - the biological condition under which the reaction occurs — EXCEPT when the two records are
    the SAME core transformation (same participants, same catalyst, same molecular action) and
    the condition is the ONLY thing separating them. That is one reaction tested under two
    conditions, not two reactions: merge it, keep BOTH regulatory directions, and flag each
    with a "note" naming the condition it holds under.
  - the compartment, when the two are genuinely different locations (cytosol vs
    mitochondrion) — but NOT when one merely contains the other
  - the molecular action (e.g. phosphorylation vs ubiquitination)
They are also DIFFERENT if they merely share one participant.

Likewise these are the SAME, not different:
  - a domain or subunit vs its parent protein as catalyst (e.g. "ZNFX1 RZ domain" = "ZNFX1")
  - one record listing an accessory cofactor (E1/E2 enzyme, ATP, ubiquitin) the other omits
  - a compartment vs a sub-compartment of it ("mitochondrial outer membrane" = "mitochondrion")
  - DIFFERENT SETS OF REGULATORS. Regulation hangs OFF a reaction; it is not part of the
    reaction's identity. Two experiments on the same event naturally test different
    regulators — one knocks out X, another uses a phospho-null mutant of Y — so their
    regulator lists are additive and get unioned when the records merge. A regulator in one
    record and absent from the other, or two records naming entirely different regulators,
    is NOT a difference. (A conflicting regulationType on the same regulator still is; see
    above.)
  - GO molecular-function wording for the same catalytic act ("ubiquitin-protein ligase
    activity" = "ubiquitin-protein transferase activity")
Judge the CORE transformation: which entity acts on which, and what change it undergoes.

When in doubt, do NOT merge. Two records left separate can still be merged by a curator
later; a merge that should not have happened is harder to undo.

Reaction A:
{_reaction_brief(a1)}

Reaction B:
{_reaction_brief(a2)}

Answer with the "same" key FIRST so the verdict survives truncation.
Keep "reason" under 15 words. "confidence" is how sure you are OF THIS VERDICT, 0-1: high when
the two records plainly describe one event, low when you had to weigh a judgement call.
Return ONLY JSON: {{"same": <true|false>, "confidence": <0-1>, "reason": "<short phrase>"}}"""
    for attempt in (1, 2):
        try:
            # no temperature: claude-sonnet-5 rejects it ("`temperature` is deprecated for this model")
            # max_tokens covers THINKING plus the answer, and thinking is bounded by it rather
            # than by a budget of its own, so too low a ceiling gets spent entirely on thinking
            # and the reply comes back with no text block at all. 1000 was enough for all 990
            # ZNFX1 pairs, but a hard pair would silently return "not duplicate", so leave room.
            msg = client.messages.create(model=MODEL_NAME, max_tokens=4000,
                                         messages=[{'role': 'user', 'content': prompt}])
            # Skip thinking/other blocks; take the first text block. With extended
            # thinking on, content[0] is a ThinkingBlock (no .text) and would crash.
            track_usage(msg)
            txt = next((b.text for b in msg.content if getattr(b, 'type', None) == 'text'), '')
            txt = txt.strip().replace('```json','').replace('```','').strip()
            if not txt:
                raise ValueError(f'no text block in response (stop_reason={msg.stop_reason})')
            try:
                got = json.loads(txt)
                return bool(got.get('same', False)), _conf(got.get('confidence'))
            except json.JSONDecodeError:
                # truncated or malformed JSON — the verdict is still readable
                m = re.search(r'"same"\s*:\s*(true|false)', txt, re.I)
                if m:
                    c = re.search(r'"confidence"\s*:\s*([0-9.]+)', txt)
                    return m.group(1).lower() == 'true', _conf(c and c.group(1))
                raise
        except Exception as ex:
            if attempt == 1:
                continue
            failures.append((type(ex).__name__, str(ex)[:150]))
            # a systemic failure (bad param, auth, rate limit) hits every pair — print
            # the first few, then stay quiet and report the totals at the end
            if len(failures) <= 3:
                print(f"    dup-judge FAILED after retry ({type(ex).__name__}: {ex}) -> NOT duplicate", flush=True)
            elif len(failures) == 4:
                print("    (further dup-judge failures suppressed; totals reported at the end)", flush=True)
            return False, None

def _key(x):
    """Dedup key: case-, punctuation- and whitespace-insensitive, so cosmetic
    variants collapse ("PINK1 - requirement -> X" == "PINK1-requirement->X")."""
    return re.sub(r'[^a-z0-9]+', '', _norm(x))


def _as_list(s):
    return [] if not s else (s if isinstance(s, list) else [s])


def merge_two(e1, e2):
    a1, a2 = e1['annotation_result'], e2['annotation_result']
    def union(l1, l2):
        seen, out = set(), []
        for x in (l1 or []) + (l2 or []):
            if _key(x) not in seen:
                seen.add(_key(x)); out.append(x)
        return out
    def union_regs(l1, l2):
        """One entry per distinct (regulationType, regulator): the same requirement
        restated by every variant in the cluster is ONE regulation, not N. A duplicate
        entry is dropped but its curator "note" is not — the flag survives on the entry
        that is kept, so a condition-dependence flag cannot be lost to dedup."""
        seen, out = {}, []
        for r in (l1 or []) + (l2 or []):
            if not isinstance(r, dict):
                continue
            k = (_key(r.get('regulationType')), _key(r.get('regulator')))
            if not any(k):
                continue
            if k not in seen:
                seen[k] = len(out); out.append(dict(r))   # copy: never mutate the input record
                continue
            kept, note = out[seen[k]], (r.get('note') or '').strip()
            if note and _key(note) != _key(kept.get('note')):
                kept['note'] = f"{kept['note']} {note}".strip() if kept.get('note') else note
        return out
    def pick_compartment(c1, c2):
        """Keep the MORE SPECIFIC compartment. The judge now treats a compartment and a
        sub-compartment of it as the same location, so a merge must not throw the finer
        one away: "mitochondrial outer membrane" beats "mitochondrion"."""
        if not c1 or not c2:
            return c1 or c2
        k1, k2 = _key(c1), _key(c2)
        if k1 == k2:
            return c1
        if k1 in k2:
            return c2
        if k2 in k1:
            return c1
        return c1 if len(c1) >= len(c2) else c2
    ca1, ca2 = a1.get('catalystActivity',{}) or {}, a2.get('catalystActivity',{}) or {}
    merged_ca = ca1 if sum(1 for v in ca1.values() if v) >= sum(1 for v in ca2.values() if v) else ca2
    # When only one record recorded a catalyst, keeping it silently asserts it for both. Name the
    # variant it came from instead. Copy: the source record is reused by the rest of the fold.
    other = ca2 if merged_ca is ca1 else ca1
    if _key(merged_ca.get('catalyst')) and not _key(other.get('catalyst')):
        src = (a1 if merged_ca is ca1 else a2).get('name', '')
        merged_ca = dict(merged_ca)
        merged_ca.setdefault('note', f'catalyst recorded only by the merged variant "{src}"')
    # keep every distinct summation text for now; consolidate_sections() folds them
    # into the one summation section the reaction is supposed to have
    seen_t, summ = set(), []
    for s in _as_list(a1.get('summation')) + _as_list(a2.get('summation')):
        if isinstance(s, dict) and _key(s.get('text')) and _key(s.get('text')) not in seen_t:
            seen_t.add(_key(s.get('text'))); summ.append(s)
    names1 = a1.get('merged_names') or [a1.get('name', '')]
    names2 = a2.get('merged_names') or [a2.get('name', '')]
    merged = {
        'name': a1.get('name') or a2.get('name',''),
        'reactionType': a1.get('reactionType') or a2.get('reactionType',''),
        'input': union(a1.get('input',[]), a2.get('input',[])),
        'output': union(a1.get('output',[]), a2.get('output',[])),
        'catalystActivity': merged_ca,
        'regulatedBy': union_regs(a1.get('regulatedBy'), a2.get('regulatedBy')),
        'compartment': pick_compartment(a1.get('compartment'), a2.get('compartment')),
        'condition': a1.get('condition') or a2.get('condition'),
        'summation': summ,
        'relationships': union(a1.get('relationships',[]), a2.get('relationships',[])),
        # combine the source-text excerpts from both copies so the merged reaction cites every place it was found
        'evidence': union(a1.get('evidence',[]), a2.get('evidence',[])),
        'confidence': round(((a1.get('confidence') or 0)+(a2.get('confidence') or 0))/2, 3),
        'merged_names': sorted(set(names1 + names2)),   # full cluster membership
        # which neighbouring chunks the variants had to consult — "none" only survives
        # if no variant looked outside its own chunk
        'context_used': union(_as_list(a1.get('context_used')), _as_list(a2.get('context_used'))),
    }
    if len(merged['context_used']) > 1:
        merged['context_used'] = [c for c in merged['context_used'] if c != 'none']
    return {'source': e1['source'], 'annotation_result': merged}

# _key() only collapses cosmetic duplicates. Wording-level duplicates survive it
# ("Parkin (mitochondria)" vs "Parkin (mitochondrial)"), as do restatements of the same
# fact across a cluster's summations, relationships and evidence. One LLM pass per merged
# cluster reduces every subsection to a single non-repeating section. Field shapes are
# unchanged — summation stays a list, just with one consolidated entry.
consolidate_failures = []

def normalize_summation(a):
    """Coerce summation to ONE section shaped like extraction emits it — a dict — so no
    consumer has to branch on dict-vs-list (labeled_docx_comments.py reads .get('text')
    straight off it). merge_two() accumulates the cluster's distinct texts in a list and
    consolidate_sections() normally reduces them to one; this is the single choke point
    before writing, so it also covers a consolidation that failed or never ran, joining
    whatever texts remain instead of leaving a list on disk. Nothing is dropped."""
    s = a.get('summation')
    if not s or isinstance(s, dict):
        return a                          # already the target shape
    texts, refs = [], []
    for x in _as_list(s):
        if isinstance(x, dict):
            t = (x.get('text') or '').strip()
            if t and t not in texts:
                texts.append(t)
            for r in x.get('literatureReference') or []:
                if r not in refs:
                    refs.append(r)
        elif isinstance(x, str) and x.strip() and x.strip() not in texts:
            texts.append(x.strip())
    a['summation'] = {'text': ' '.join(texts), 'literatureReference': refs}
    return a


def consolidate_sections(a):
    """Collapse wording-level duplicates within one merged reaction's subsections.
    Returns a new annotation dict; returns `a` unchanged on any failure."""
    # condition is read-only context (never returned): it tells the model the coarse category
    # this cluster was observed under, which it needs to write a condition-dependence note
    payload = {k: a.get(k) for k in
               ('name', 'input', 'output', 'catalystActivity', 'regulatedBy',
                'relationships', 'summation', 'evidence', 'condition')}
    prompt = f"""You are a Reactome biocurator cleaning up ONE reaction record that was
assembled by merging several extractions of the same reaction. Because each extraction
described the reaction in its own words, the subsections now repeat themselves.

Reduce each subsection to a single non-repeating section:
- input / output: one canonical name per distinct entity. "Parkin (mitochondria)" and
  "Parkin (mitochondrial)" are the SAME entity — keep one. Different entities, or the
  same protein in a genuinely different state/compartment, stay separate.
- regulatedBy: one entry per distinct (regulationType, regulator). Collapse synonymous
  regulators ("UbS65A" = "UbS65A (phospho-null ubiquitin)"), keeping the more informative
  name. A different regulationType on the same regulator is a DIFFERENT entry — keep both.
  When the same regulator appears in two directions and the evidence excerpts show the split
  is due to conditional circumstances, such as a different experimental condition, set "note"
  on each entry flagging that condition for the curator. Take the condition from the evidence
  or the condition field — never invent one. Leave "note" null otherwise, and preserve any
  note already present.
- relationships: one line per distinct relation. Drop a line that is a less specific
  restatement of another ("PINK1 - positiveRegulation -> Parkin translocation" is subsumed
  by "PINK1 - positiveRegulation -> Parkin translocation to mitochondria"). Contradictory
  or genuinely different relations stay.
- summation: ONE text. Combine every distinct fact from all the texts into a single
  coherent curator summary; state each fact once. Do not add facts that are not there.
- evidence: these are VERBATIM quotes from the paper — never reword or shorten them.
  Only drop a quote that is identical to another or fully contained within another.
  Every quote covering a distinct passage must be kept.
- catalystActivity: return it as-is.

Rules: NEVER invent an entity, regulator, relation, fact or quote that is not in the input.
NEVER drop a distinct piece of biology — only fold together things that say the same
thing. If unsure whether two items are the same, KEEP BOTH. Every output list must be
the same length or shorter than its input.

Reaction record:
{json.dumps(payload, indent=2)}

Return ONLY JSON with exactly these keys:
{{"input": [...], "output": [...], "regulatedBy": [{{"regulationType": "...", "regulator": "...", "note": "<null, or a curator flag naming the condition this direction holds under>"}}],
  "relationships": [...], "evidence": [...], "summation": {{"text": "<single combined summary>"}}}}"""
    for attempt in (1, 2):
        try:
            # 8000 was not enough: on ZNFX1's 7-variant cluster thinking consumed the whole
            # ceiling and the reply carried no answer, leaving that reaction unconsolidated
            msg = client.messages.create(model=MODEL_NAME, max_tokens=16000,
                                         messages=[{'role': 'user', 'content': prompt}])
            track_usage(msg)
            txt = next((b.text for b in msg.content if getattr(b, 'type', None) == 'text'), '')
            txt = txt.strip().replace('```json', '').replace('```', '').strip()
            if not txt:
                raise ValueError(f'no text block in response (stop_reason={msg.stop_reason})')
            got = json.loads(txt)
            out = dict(a)
            # a list that came back LONGER than it went in means the model invented
            # items — reject that subsection and keep the code-merged version
            for f in ('input', 'output', 'regulatedBy', 'relationships', 'evidence'):
                new = got.get(f)
                if isinstance(new, list) and len(new) <= len(a.get(f) or []):
                    out[f] = new
            text = (got.get('summation') or {}).get('text') if isinstance(got.get('summation'), dict) else None
            if text and text.strip():
                refs = []
                for s in _as_list(a.get('summation')):
                    for r in (s.get('literatureReference') or []) if isinstance(s, dict) else []:
                        if r not in refs:
                            refs.append(r)
                # shape unchanged (list), now holding the one consolidated summation
                out['summation'] = [{'text': text.strip(), 'literatureReference': refs}]
            return out
        except Exception as ex:
            if attempt == 1:
                continue
            consolidate_failures.append((type(ex).__name__, str(ex)[:150]))
            print(f"    consolidate FAILED for '{a.get('name','')}' "
                  f"({type(ex).__name__}: {ex}) -> left as-is", flush=True)
            return a


# Stage 3. The pairwise judge is structurally blind to anything that only shows up across
# three or more records: successive steps of one mechanism each look like "the same event,
# more detail" to their neighbour, and a regulatory statement looks like the reaction it
# regulates. Showing the whole cluster at once is what makes those visible. The question is
# deliberately about Reactome reaction IDENTITY, not semantic similarity — the records in a
# cluster are always topically similar, that is why they got clustered.
validate_failures = []


def validate_cluster(idxs):
    """Ask whether `idxs` is ONE Reactome reaction; return a list of index groups.

    Returns [idxs] unchanged when the cluster holds together, or two or more groups when it
    spans several reactions. Returns [idxs] on any failure or unusable response: this stage
    is a refinement, so a bad call must leave the cluster as the previous stage built it.
    Every input index appears in exactly one returned group — never dropped, never repeated.
    """
    listing = '\n\n'.join(f"[{k}]\n{_reaction_brief(unique[i]['annotation_result'])}"
                          for k, i in enumerate(idxs, 1))
    prompt = f"""You are a Reactome biocurator. The {len(idxs)} records below were grouped
together as descriptions of ONE reaction, because each was judged a duplicate of another in
the group. Judging them two at a time cannot reveal a group that has drifted across more than
one reaction, so check the whole group at once.

Decide: do ALL of these records describe the SAME biological transformation, such that they
should be represented as a SINGLE Reactome reaction? If not, divide them into groups where
each group is exactly one reaction.

A Reactome reaction is ONE transformation: defined inputs converted to defined outputs, with
at most one catalyst activity. Judge reaction IDENTITY, not topical similarity — these records
all concern the same protein and the same paper, so similarity tells you nothing here.

DIVIDE the group when it mixes:
  - successive steps of one mechanism. Each step is its own reaction: charging an E2, transfer
    of ubiquitin from E2 to the E3 catalytic cysteine (a thioester intermediate), and transfer
    from that intermediate to the substrate are THREE reactions, not one described three ways.
  - a different product of the same chemistry — mono-ubiquitination vs polyubiquitination, or a
    distinct chain linkage — but ONLY when both records name the product and they conflict.
    "ubiquitinated X" with no chain specified is less detail, not a different product.
  - a regulatory statement whose own inputs and outputs are a DIFFERENT event. Judge by input
    and output, not by the name: "X activates R" recorded with R's participants is R.
  - a binding or complex-formation event together with the catalytic event that follows it.
  - a different substrate, or a different catalyst.

KEEP records together when they are the same transformation:
  - described in different words, or at a different level of detail
  - with an accessory cofactor named in one and omitted in another (E1, E2, ATP, ubiquitin)
  - with a domain or subunit named as catalyst in one and the parent protein in the other
  - measured by a different assay, or under a different experimental condition
  - with different sets of regulators tested. Merging unions them into the ONE reaction's
    regulatedBy, each noting the condition it holds under, so four records of one
    transformation testing four regulators are ONE reaction with four regulations.

Every pair here was already judged the same reaction directly, so do NOT divide because you are
unsure — name which reason above applies, or return one group.

Records:

{listing}

Every record number 1-{len(idxs)} must appear in exactly ONE group. Do not invent, drop or
renumber records. Set "valid_single_cluster" to true only when there is exactly one group.
Keep "reason" under 30 words.
Return ONLY JSON: {{"valid_single_cluster": <true|false>, "groups": [[1, 2], [3]], "reason": "<short phrase>"}}"""
    for attempt in (1, 2):
        try:
            # Partitioning N records is combinatorial, so this is the most thinking-hungry call
            # in the pipeline: 11 ZNFX1 records took 3854 thinking tokens against a 230-character
            # answer. A ceiling of 2000 was spent entirely on thinking and returned no answer,
            # which silently meant "do not split" on the very clusters most in need of splitting.
            msg = client.messages.create(model=MODEL_NAME, max_tokens=16000,
                                         messages=[{'role': 'user', 'content': prompt}])
            track_usage(msg)
            txt = next((b.text for b in msg.content if getattr(b, 'type', None) == 'text'), '')
            txt = txt.strip().replace('```json', '').replace('```', '').strip()
            if not txt:
                raise ValueError(f'no text block in response (stop_reason={msg.stop_reason})')
            got = json.loads(txt)
            raw = got.get('groups')
            if got.get('valid_single_cluster') and not raw:
                return [list(idxs)]
            if not isinstance(raw, list):
                raise ValueError('no usable "groups" list')
            # Rebuild the partition ourselves rather than trusting it: a number out of range
            # or repeated would otherwise duplicate or lose a reaction.
            taken, groups = set(), []
            for g in raw:
                grp = []
                for k in (g if isinstance(g, list) else [g]):
                    try:
                        k = int(k)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= k <= len(idxs) and k not in taken:
                        taken.add(k); grp.append(idxs[k - 1])
                if grp:
                    groups.append(sorted(grp))
            # A record the model forgot becomes its own group. Splitting is the recoverable
            # direction of error; dropping it from the output is not.
            missing = [idxs[k - 1] for k in range(1, len(idxs) + 1) if k not in taken]
            if missing:
                print(f"    [validate] response omitted {len(missing)} record(s); "
                      f"kept as singleton(s)", flush=True)
                groups.extend([[i] for i in missing])
            if not groups:
                raise ValueError('response partitioned nothing')
            return groups
        except Exception as ex:
            if attempt == 1:
                continue
            validate_failures.append((type(ex).__name__, str(ex)[:150]))
            print(f"    [validate] FAILED for cluster of {len(idxs)} "
                  f"({type(ex).__name__}: {ex}) -> left as one cluster", flush=True)
            return [list(idxs)]


# 1b) pairwise candidate generation — judge every same-source pair CONCURRENTLY
MAX_WORKERS = 12          # parallel judge calls (independent, so safe to run concurrently)
n = len(unique)

pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
         if unique[i]['source'] == unique[j]['source']]
print(f"[merge] exhaustive judge over {n} reactions ({len(pairs)} pairs), {MAX_WORKERS} concurrent...", flush=True)
t0 = time.time(); done = [0]; total = [len(pairs)]   # total grows when shaky pairs are re-voted

def judge(pair):
    i, j = pair
    same, conf = reactions_are_duplicate(unique[i]['annotation_result'],
                                         unique[j]['annotation_result'])
    done[0] += 1
    if done[0] % 100 == 0:
        print(f"  judged ~{done[0]}/{total[0]} pairs | {time.time()-t0:.0f}s "
              f"| {usage_line()}", flush=True)
    return (i, j, same, conf)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    verdicts = list(pool.map(judge, pairs))

# Verdicts keyed by (lower, higher) index, because clustering now QUERIES arbitrary pairs
# repeatedly — "may these two groups join, i.e. is every pair across them same?" — instead of
# consuming the list once. A pair that is absent was never judged (a different source, or a
# call that failed) and every rule below reads absent as NOT same.
# Nothing is printed per same-edge any more: under the conflict-free rule a same-edge does not
# imply those two reactions end up together, since a conflict elsewhere in the group can veto
# it. What actually merged is printed per cluster, once clustering has settled.
SAME = {(i, j): bool(same) for i, j, same, conf in verdicts}
CONF = {(i, j): conf for i, j, same, conf in verdicts}
n_same = sum(1 for v in SAME.values() if v)
print(f"[merge] judged {len(pairs)} pairs in {time.time()-t0:.0f}s "
      f"| {n_same} same, {len(SAME) - n_same} different", flush=True)

# 1c) re-vote the shaky verdicts. One verdict per pair decides a merge, and the conflict-free
# rule needs EVERY pair across two groups to say "same" — so a single unstable verdict vetoes a
# merge that every other pair supports. The three "Parkin ubiquitinates Mfn1" records judged
# same on all three pairs when re-judged, yet came out as a 2-cluster plus a singleton during
# the run: claude-sonnet-5 takes no temperature parameter, so a borderline pair really does
# flip between identical calls. Confident verdicts do not flip, so only the low-confidence ones
# are re-judged, and the cost scales with how many pairs were actually borderline rather than
# tripling the whole run. A pair whose call FAILED has conf None and is left alone — it is
# already reported below, and a systemic failure would otherwise re-judge everything for nothing.
shaky = [p for p in SAME if CONF.get(p) is not None and CONF[p] < args.revote_below]
if shaky:
    print(f"[revote] {len(shaky)} verdict(s) under confidence {args.revote_below}; "
          f"2 more votes each, majority wins, {MAX_WORKERS} concurrent...", flush=True)
    t1 = time.time(); total[0] += 2 * len(shaky)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        extra = list(pool.map(judge, shaky * 2))
    votes = {p: [SAME[p]] for p in shaky}
    for i, j, same, conf in extra:
        votes[(i, j)].append(bool(same))
    flipped = 0
    for p, vs in votes.items():
        win = sum(vs) * 2 > len(vs)
        if win != SAME[p]:
            flipped += 1
            print(f"    {SAME[p]} -> {win} (votes {vs}): "
                  f"{unique[p[0]]['annotation_result'].get('name','')[:44]!r} vs "
                  f"{unique[p[1]]['annotation_result'].get('name','')[:44]!r}", flush=True)
        SAME[p] = win     # CONF keeps the first vote's number: a pair that flipped INTO "same"
                          # stays marked low-confidence, which is the honest label for it
    n_same = sum(1 for v in SAME.values() if v)
    print(f"[revote] done in {time.time()-t1:.0f}s | {flipped} verdict(s) flipped "
          f"| {n_same} same, {len(SAME) - n_same} different", flush=True)

if failures:
    from collections import Counter
    print(f"[warn] {len(failures)}/{len(pairs)} pair(s) never got a verdict and were treated as "
          f"NOT duplicate — the merge is under-merged:", flush=True)
    for (name, detail), count in Counter(failures).most_common():
        print(f"         {count}x {name}: {detail}", flush=True)
    if len(failures) > len(pairs) * 0.1:
        print("[warn] that is a systemic failure, not noise — fix it and re-run; "
              "the output below is not trustworthy", flush=True)


# 2) conflict-aware clustering
def conflict_free_clusters():
    """Cluster the verdicts so that every cluster is a clique in the "same" graph: no pair
    inside a cluster was judged different.

    Transitive closure over the same edges would put A and C together on the strength of A~B
    and B~C even when A-vs-C was judged different — on ZNFX1 that chained three distinct
    mechanistic steps into one 11-variant reaction.

    Agglomerative: repeatedly join the two groups that are FULLY connected to each other,
    taking the join whose weakest connecting verdict is strongest, so the result does not
    depend on the order pairs happen to be listed in. Joining stops when no two groups are
    fully connected, which leaves conflicting variants in separate clusters.
    """
    groups = [[i] for i in range(n)]
    while True:
        best = None
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                cross = [(min(a, b), max(a, b)) for a in groups[gi] for b in groups[gj]]
                if not all(SAME.get(p) for p in cross):
                    continue
                # a "same" the judge put no number on counts as 0.0 here: it can still join,
                # but only after every join backed by an actual confidence
                weakest = min(CONF.get(p) or 0.0 for p in cross)
                if best is None or weakest > best[0]:
                    best = (weakest, gi, gj)
        if best is None:
            return sorted(groups, key=lambda g: (-len(g), g))
        _, gi, gj = best
        groups[gi] = sorted(groups[gi] + groups[gj]); groups.pop(gj)


groups = conflict_free_clusters()
print(f"[cluster] {n} -> {len(groups)} conflict-free cluster(s)", flush=True)

# 3) cluster-level validation — may only split a cluster, never join
if args.no_validate:
    print("[validate] skipped (--no-validate)", flush=True)
else:
    to_check = [g for g in groups if len(g) >= args.validate_min_size]
    if not to_check:
        print(f"[validate] no cluster reaches {args.validate_min_size} variants — "
              f"nothing to check", flush=True)
    else:
        print(f"[validate] checking {len(to_check)} cluster(s) of >= {args.validate_min_size} "
              f"variants for Reactome reaction identity, {MAX_WORKERS} concurrent...", flush=True)
        t2 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            splits = list(pool.map(validate_cluster, to_check))
        kept = [g for g in groups if len(g) < args.validate_min_size]
        for original, parts in zip(to_check, splits):
            if len(parts) > 1:
                print(f"   SPLIT: {len(original)} variants -> {len(parts)} reaction(s)", flush=True)
            kept.extend(parts)
        groups = sorted(kept, key=lambda g: (-len(g), g))
        print(f"[validate] {sum(1 for p in splits if len(p) > 1)}/{len(to_check)} cluster(s) "
              f"split in {time.time()-t2:.0f}s -> {len(groups)} cluster(s)", flush=True)
        if validate_failures:
            from collections import Counter
            print(f"[warn] {len(validate_failures)}/{len(to_check)} cluster(s) were not "
                  f"validated and were left exactly as stage 2 built them:", flush=True)
            for (name, detail), count in Counter(validate_failures).most_common():
                print(f"         {count}x {name}: {detail}", flush=True)

multi = [g for g in groups if len(g) > 1]
if multi:
    print(f"[cluster] {len(multi)} multi-variant cluster(s):", flush=True)
    for g in multi:
        cs = [CONF[p] for p in itertools.combinations(g, 2)
              if SAME.get(p) and CONF.get(p) is not None]
        print(f"   {len(g)} variants, weakest verdict {min(cs) if cs else '?'}:", flush=True)
        for k in g:
            print(f"      - {unique[k]['annotation_result'].get('name','')}", flush=True)

# merge_confidence answers "was this reaction correctly merged?", which is a different
# question from the per-reaction "confidence" extraction self-assigns (that one is left
# alone — cosine_similarity_score.py reads it as llm_confidence). A cluster is only as
# sound as the shakiest verdict holding it together, so take the MINIMUM: a mean would let
# three confident joins hide one bad one.
merged_results = []
for idxs in groups:
    cur = unique[idxs[0]]
    for k in idxs[1:]:
        cur = merge_two(cur, unique[k])
    # recomputed from the FINAL membership, so a stage-3 split drops the confidences of the
    # pairs it separated instead of crediting them to the group that survived
    cs = [CONF[p] for p in itertools.combinations(idxs, 2)
          if SAME.get(p) and CONF.get(p) is not None]
    # None rather than 1.0 for a reaction that never merged: there is no merge to be
    # confident about, and that must stay distinguishable from "merged, but shakily"
    cur['annotation_result']['merge_confidence'] = min(cs) if cs else None
    # Unweighted mean of every variant's confidence.
    vals = [unique[k]['annotation_result'].get('confidence') for k in idxs]
    vals = [v for v in vals if isinstance(v, (int, float))]
    cur['annotation_result']['confidence'] = round(sum(vals) / len(vals), 3) if vals else None
    merged_results.append(cur)

# 4) consolidate subsections of every reaction that actually merged something —
# a single-variant reaction has nothing to deduplicate, so it is left untouched
to_fix = [e for e in merged_results if len(e['annotation_result'].get('merged_names') or []) > 1]
if to_fix:
    print(f"[consolidate] deduplicating subsections of {len(to_fix)}/{len(merged_results)} "
          f"multi-variant reactions, {MAX_WORKERS} concurrent...", flush=True)
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fixed = list(pool.map(lambda e: consolidate_sections(e['annotation_result']), to_fix))
    for e, a in zip(to_fix, fixed):
        before = e['annotation_result']
        removed = {f: len(before.get(f) or []) - len(a.get(f) or [])
                   for f in ('input', 'output', 'regulatedBy', 'relationships', 'evidence', 'summation')}
        e['annotation_result'] = a
        if any(v > 0 for v in removed.values()):
            print(f"   {before.get('name','')}: "
                  + ', '.join(f"-{v} {f}" for f, v in removed.items() if v > 0), flush=True)
    print(f"[consolidate] done in {time.time()-t1:.0f}s", flush=True)
    if consolidate_failures:
        from collections import Counter
        print(f"[warn] {len(consolidate_failures)}/{len(to_fix)} reaction(s) were left "
              f"un-consolidated and may still contain duplicate entries:", flush=True)
        for (name, detail), count in Counter(consolidate_failures).most_common():
            print(f"         {count}x {name}: {detail}", flush=True)

# finally, one summation section per reaction, uniformly shaped, whatever happened above
for e in merged_results:
    normalize_summation(e['annotation_result'])

with open(OUT, 'w') as f:
    json.dump(merged_results, f, indent=2)

print(f"[done] {n} -> {len(merged_results)} reactions | {len(pairs)} pairs judged | {time.time()-t0:.0f}s", flush=True)
print(f"[usage] {usage_line()}", flush=True)
print("[done] merged reactions:", flush=True)
for i, e in enumerate(merged_results, 1):
    a = e['annotation_result']
    names = a.get('merged_names') or [a.get('name','')]
    print(f"   {i:2d}. {a.get('name','')}"
          + (f"   (merged {len(names)} variants)" if len(names) > 1 else ""), flush=True)
print(f"[done] saved -> {OUT}", flush=True)
