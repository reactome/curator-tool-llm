"""Manual check for what ACTUALLY goes into extraction — from a PDF or from PubMed XML.

Usage:
  python check_sections.py                        # defaults to PINK1.pdf
  python check_sections.py PINK1.pdf              # a file name in data/papers/
  python check_sections.py /abs/path/to/paper.pdf
  python check_sections.py PMC4003245             # PubMed Central, JATS XML
  python check_sections.py 24751536               # PMID -> resolved to a PMCID
  python check_sections.py PINK1.pdf --unwrap-pdf # preview the gated PDF unwrap
  python check_sections.py PMC4003245 --out /tmp/x.log

Output goes to the terminal AND to results/<stem>_sections.log (override with --out), so
the chunks can be read afterwards instead of being re-generated.

Everything below the source load reuses the PIPELINE's own code, so what this prints is
what run_extraction.py feeds the model rather than a reimplementation that can drift:
  - fetcher.load_source()      same fetch + section detection + flatten_jats_text
  - run_extraction.splitter    the same splitter OBJECT, not a copy of its parameters

PART 1 — SECTION SURVEY. For a PDF, chunk the whole paper and run the per-chunk
  section-title check (one LLM call per chunk, ~1 min). For XML, JATS tags its sections
  explicitly, so no LLM detection is needed and the route is reported instead.
PART 2 — THE EXACT TEXT HANDED TO THE SPLITTER (already flattened, for XML), so the
  start and end of the section can be verified by eye.
PART 3 — THAT TEXT SPLIT BY THE PIPELINE'S SPLITTER, with per-chunk character, word and
  token counts. The splitter budgets CHARACTERS, so the token column is the only place a
  chunk's real size is visible.

Token counts come from Anthropic's count_tokens endpoint, i.e. the tokenizer the model
actually uses. That is one request per chunk plus one for the section — they are not
billed as input tokens, but they do count against request rate limits.
"""
import os, sys, json, argparse
PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)
import anthropic
import PubMedFetcher as fetcher
# Import the pipeline's splitter rather than re-declaring chunk_size/chunk_overlap/
# length_function here: a local copy drifts the moment run_extraction.py is tuned, and
# then this tool reports chunks the extraction never saw. run_extraction's CLI is behind
# an __main__ guard, so importing it only builds its client and graph.
import run_extraction as pipeline

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-5'

# count_tokens prices a whole request, so even a 1-character message costs 7 tokens of
# message scaffolding. Subtract that to report the text's own size.
_TOK_OVERHEAD = 6
_tok_warned = []


def ntok(text):
    """Claude's token count for `text` alone, or None if the endpoint is unreachable."""
    try:
        r = client.messages.count_tokens(
            model=MODEL_NAME, messages=[{'role': 'user', 'content': text}])
        return max(0, r.input_tokens - _TOK_OVERHEAD)
    except Exception as e:
        if not _tok_warned:
            _tok_warned.append(1)
            print(f"  [warn] count_tokens failed ({type(e).__name__}: {e}) — "
                  f"token columns omitted", flush=True)
        return None


TOKEN_TARGET = 300      # the chunk size this pipeline is described as using


def resolve_spec(arg):
    """A PMID/PMCID goes to PubMed Central; anything else must be a PDF on disk."""
    if fetcher.is_pmid(arg) or fetcher.is_pmcid(arg):
        return arg, 'xml'
    if os.path.isfile(arg):
        return arg, 'pdf'
    candidate = os.path.join(PROJECT_ROOT, 'data', 'papers', arg)
    if os.path.isfile(candidate):
        return candidate, 'pdf'
    if not arg.lower().endswith('.pdf') and os.path.isfile(candidate + '.pdf'):
        return candidate + '.pdf', 'pdf'
    raise FileNotFoundError(
        f"not a PMID, not a PMCID, and no such PDF: {arg!r} (looked in cwd and data/papers/)")


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('source', nargs='?', default='PINK1.pdf',
                    help='PDF path, a file name inside data/papers/, a PMID, or a PMCID '
                         '(default: PINK1.pdf)')
parser.add_argument('--unwrap-pdf', action='store_true',
                    help="preview unwrap_pdf_text() on the PDF path. OFF by default to match "
                         "the pipeline, which keeps the PDF path byte-identical to earlier runs")
parser.add_argument('--out', help='write this run to a file as well as the terminal '
                                  '(default: results/<stem>_sections.log)')
args = parser.parse_args()


class _Tee:
    """Send every print() to the terminal AND the log file, so a run can be watched live
    and still be read afterwards without re-running it."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# Named like the pipeline's own logs, and stem-derived so a PMCID run cannot land on top
# of a PDF run's output. Re-running the SAME source does overwrite, as a log should.
OUT_PATH = args.out or os.path.join(
    PROJECT_ROOT, 'results', f'{fetcher.output_stem(args.source)}_sections.log')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
_logf = open(OUT_PATH, 'w')
sys.stdout = _Tee(sys.__stdout__, _logf)
print(f"[check_sections] writing to {OUT_PATH}")

spec, kind = resolve_spec(args.source)
print(f"[check_sections] {kind.upper()} source: {spec}")
if kind == 'xml' and args.unwrap_pdf:
    print("[check_sections] --unwrap-pdf ignored: it applies to the PDF path only")

# Load exactly the way run_extraction.py does. Any fetcher warnings (e.g. a section that
# spans Results and Discussion) are printed by the fetcher itself as it loads.
source_id, results_text, how = fetcher.load_source(
    spec, client=client, model=MODEL_NAME,
    unwrap_pdf=(args.unwrap_pdf and kind == 'pdf'))

# ── PART 1: section survey ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 1 — SECTION SURVEY")
print("=" * 70)
if kind == 'pdf':
    import fitz
    from FullTextPDFSections import map_section_titles
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    doc = fitz.open(spec)
    full_text = ''.join(p.get_text() for p in doc); doc.close()
    # This splitter is deliberately NOT the pipeline's: the title check reads across
    # chunk seams, so it overlaps to keep a heading visible when a boundary splits it.
    survey = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400,
                                            length_function=len,
                                            separators=['\n\n', '\n', '. ', ' ', ''])
    survey_chunks = survey.split_text(full_text)
    print(f"whole paper: {len(full_text.split()):,} words -> {len(survey_chunks)} survey chunks")
    print("one LLM call per chunk (~1 min)...", flush=True)
    section_map = map_section_titles(survey_chunks, client, MODEL_NAME)
    for r in section_map:
        if r['is_section_title']:
            print(f"  chunk {r['chunk_index']:2d}:  is_section_title=True  -> title: {r['title']}")
        else:
            print(f"  chunk {r['chunk_index']:2d}:  no section title found")
    print(f"\n  ALL SECTIONS FOUND: {[r['title'] for r in section_map if r['is_section_title']]}")
else:
    print("JATS tags its sections explicitly, so no per-chunk LLM detection runs here.")
    print(f"  detection route: {how}")
    print("  route meanings: jats:sec-type -> matched sec-type=\"results\"")
    print("                  jats:title    -> matched the section <title>")
    print("                  jats-body:*   -> no tagged section; fell back to the PDF-era")
    print("                                   text heuristics over the whole <body>")
    print("Any '[jats] section spans Results and Discussion' warning above is the fetcher's:")
    print("it means the section carries Discussion prose and figure legends too.")

# ── PART 2: the exact text handed to the splitter ─────────────────────────────
print("\n" + "=" * 70)
print("PART 2 — EXACT TEXT HANDED TO THE SPLITTER")
print("=" * 70)
meta = {'source_id': source_id, 'section': 'Results', 'route': how,
        'n_words': len(results_text.split()), 'n_chars': len(results_text)}
sec_tok = ntok(results_text)
if sec_tok is not None:
    meta['n_tokens'] = sec_tok
if kind == 'pdf':
    meta['unwrap_pdf'] = bool(args.unwrap_pdf)
else:
    meta['flattened'] = True          # load_source always flattens the XML path
print(json.dumps(meta, indent=2))
print("\n--- full text, verbatim (check where the section starts and ends) ---")
print(results_text)

# ── PART 3: the pipeline's own chunking ───────────────────────────────────────
chunks = pipeline.splitter.split_text(results_text)
print("\n" + "=" * 70)
print(f"PART 3 — {len(chunks)} CHUNKS FROM run_extraction.splitter")
print("=" * 70)
print("The splitter's budget is CHARACTERS (length_function=len), so a chunk's token")
print("count is not bounded by it — the column below is where the real size shows up.")
print(f"counting tokens for {len(chunks)} chunks...", flush=True)
toks = [ntok(c) for c in chunks]
if any(t is None for t in toks):
    toks = None
print()

for i, c in enumerate(chunks):
    bits = f"{len(c):,} chars, {len(c.split())} words"
    if toks:
        bits += f", {toks[i]} tokens"
    print(f"--- chunk {i} ({bits}) ---")
    print(c)
    print()

if toks:
    srt = sorted(toks)
    over = [i for i, t in enumerate(toks) if t > TOKEN_TARGET]
    print("=" * 70)
    print(f"token spread: min {min(toks)}, median {srt[len(srt)//2]}, max {max(toks)}")
    print(f"over {TOKEN_TARGET} tokens: {len(over)}/{len(toks)} chunk(s) {over}")
    print(f"orphans under 100 tokens: {[i for i, t in enumerate(toks) if t < 100]}")

print(f"\n[check_sections] written to {OUT_PATH}")
sys.stdout = sys.__stdout__
_logf.close()
