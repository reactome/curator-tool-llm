"""Full-text reaction extraction — generalizable to any PDF.

Results-section only, condition-aware + evidence + normalized reactions, 2-prev + 1-next
memory (LangGraph), Sonnet 4.6, no chunk overlap.

Sources may be local PDFs or PubMed IDs — PMIDs are resolved to PMCIDs and
fetched as JATS XML, which tags the Results section explicitly instead of
requiring the heuristic/LLM detection a flat PDF needs.

Usage:
  python run_extraction.py                            # every *.pdf in data/papers/
  python run_extraction.py PINK1.pdf                  # one local paper
  python run_extraction.py PINK1.pdf ZNFX1.pdf        # several
  python run_extraction.py 38234567                   # one PMID via PubMed Central
  python run_extraction.py --gene PINK1 38234567 29168502
  python run_extraction.py --pmid-file pmids.txt --gene PINK1
  python run_extraction.py PINK1.pdf --tag v2           # separate file, keeps the old one

Writes results/<stem>[_<tag>]_extraction.json per paper (incremental, per
chunk). Refuses to overwrite an existing extraction unless --overwrite is given.
"""
import os, sys, json, time, glob, argparse, importlib.util, threading
from concurrent.futures import ThreadPoolExecutor
PROJECT_ROOT = os.environ.get('CURATOR_LLM_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import anthropic, requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
# PDF loading (fitz + extract_results_section) now lives in load_source(), so the
# same call handles a local PDF or a PMID.
import PubMedFetcher as fetcher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curator_llm.services.evidence_attach import split_evidence
from curator_llm.services.direct_evidence import split_cited_only

PAPERS_DIR = os.path.join(PROJECT_ROOT, 'data', 'papers')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0)
from curator_llm.model_settings import anthropic_kwargs, model_name
MODEL_NAME = model_name('sections')       # also what the Results-section lookup in load_source() uses
MODEL_KW = anthropic_kwargs('extraction')  # model + thinking for the extraction calls (curator_llm/model_settings.py)
PREV_WINDOW = 2   # previous chunks' reaction-notes kept as backward context

# Token accounting across every API call this run makes, so a pipeline run can be
# costed. Note a Results-section LLM fallback inside FullTextPDFSections is not
# counted here — only the extraction calls made below.
USAGE = {'calls': 0, 'in': 0, 'out': 0, 'cache_read': 0, 'cache_write': 0}
_usage_lock = threading.Lock()
# Chunks extracted concurrently (1 = the original serial mode, each chunk seeing notes on the previous two).
WORKERS = int(os.getenv('EXTRACT_WORKERS', '16'))


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

spec = importlib.util.spec_from_file_location('p', os.path.join(PROJECT_ROOT, 'reactome_llm', 'FullTextPDFPrompts.py'))
pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0, length_function=len,
                                          separators=['\n\n', '\n', '. ', ' ', ''])

# A model call that fails for a passing reason (connection, timeout, rate limit, overload) is tried again after these
# waits, on top of the SDK's own retries. A chunk that still fails makes the paper INCOMPLETE: no status "complete" is
# written, so a resumed run extracts the paper again instead of keeping it with reactions missing.
RETRY_WAITS = [float(x) for x in os.getenv('EXTRACT_RETRY_WAITS', '5,20,60').split(',') if x.strip()]
FAILED_CHUNKS = set()          # chunk indexes of the paper being extracted whose model call failed for good
_failed_lock = threading.Lock()


class ChunkFailed(Exception):
    pass


def _transient(e) -> bool:
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError,
                      anthropic.InternalServerError)):
        return True
    return getattr(e, 'status_code', None) in (408, 409, 429, 500, 502, 503, 504, 529)


def _call(head, tail):
    last = None
    for attempt, wait in enumerate([0.0] + RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            # the head (instructions) is the same for every chunk: mark it for prompt caching
            return client.messages.create(**MODEL_KW, max_tokens=16000, messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': head, 'cache_control': {'type': 'ephemeral'}},
                {'type': 'text', 'text': tail}]}])
        except Exception as e:
            last = e
            print(f'    API call failed (try {attempt + 1}): {type(e).__name__}: {e}', flush=True)
            if not _transient(e):                    # a bad request, say: trying again cannot help
                break
    raise ChunkFailed(f'{type(last).__name__}: {last}')


def extract_reactions(current_chunk, prev_contexts, next_context, raw_prev=False):
    """The chunk's reactions, or None when the model's reply cannot be read even on a second try (the chunk is then
    taken as done, with nothing in it). Raises ChunkFailed when the model could not be reached."""
    head, tail = pm.build_extraction_prompt(current_chunk, prev_contexts=prev_contexts, next_context=next_context,
                                            raw_prev=raw_prev, split=True)
    for attempt in (1, 2):
        msg = _call(head, tail)
        track_usage(msg)
        raw = ''.join(b.text for b in msg.content if getattr(b, 'type', None) == 'text').strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        if not raw.startswith('{'):
            s, e = raw.find('{'), raw.rfind('}')
            if s != -1 and e != -1: raw = raw[s:e+1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f'    JSON parse failed (try {attempt}): {e}{"  [TRUNCATED]" if msg.stop_reason=="max_tokens" else ""}',
                  flush=True)
    return None


def _extract_or_mark(idx, *args, **kwargs):
    """extract_reactions, recording a chunk whose model call failed for good (it then counts as having no reactions)."""
    try:
        return extract_reactions(*args, **kwargs), False
    except ChunkFailed as e:
        print(f'    chunk {idx + 1}: giving up after retries ({e}); the paper will be marked incomplete', flush=True)
        with _failed_lock:
            FAILED_CHUNKS.add(idx)
        return None, True

def make_note(idx, reactions):
    if not reactions: return ''
    lines = []
    for r in reactions:
        ins = ', '.join(r.get('input', []) or []); outs = ', '.join(r.get('output', []) or [])
        cond = r.get('condition') or ''
        lines.append(f"{r.get('name','')} [in: {ins} | out: {outs}]" + (f" {{condition: {cond}}}" if cond else ""))
    return f"Chunk {idx}: " + '; '.join(lines)

class PipelineState(TypedDict):
    source: str
    out: str
    chunks: List[str]
    current_chunk_index: int
    reactions_found: List[Dict]
    recent_notes: List[str]     # previous chunks' notes, MOST RECENT FIRST, up to PREV_WINDOW

def process_chunk(state: PipelineState) -> PipelineState:
    idx = state['current_chunk_index']; chunk = state['chunks'][idx]
    prev_contexts = state['recent_notes']
    next_context = state['chunks'][idx + 1] if idx + 1 < len(state['chunks']) else None

    result, _ = _extract_or_mark(idx, chunk, prev_contexts, next_context)
    new_reactions = [split_evidence(r) for r in (result.get('reactions', []) if result else [])]
    note = make_note(idx, new_reactions)
    recent = (([note] if note else []) + state['recent_notes'])[:PREV_WINDOW]

    status = 'FAILED' if result is None else f'{len(new_reactions)} reaction(s)'
    print(f"  chunk {idx+1}/{len(state['chunks'])}: {status}  |  {usage_line()}", flush=True)

    all_reactions = state['reactions_found'] + [
        {'source': state['source'], 'chunk_index': idx,
         'word_count': len(chunk.split()), 'annotation_result': r}
        for r in new_reactions
    ]
    with open(state['out'], 'w') as f:
        json.dump(all_reactions, f, indent=2)

    return {**state, 'current_chunk_index': idx + 1,
            'reactions_found': all_reactions, 'recent_notes': recent}

def extract_concurrently(source_id, out, chunks, workers):
    """Every chunk independently, `workers` at a time. Backward context is the RAW text of the previous two
    chunks (known up front) rather than notes on their reactions, which would make chunk N wait for N-1.
    The first chunk goes alone so its call writes the prompt cache that the rest then read."""
    done = {}
    lock = threading.Lock()

    def work(idx):
        prev = [chunks[k] for k in (idx - 1, idx - 2) if k >= 0]
        nxt = chunks[idx + 1] if idx + 1 < len(chunks) else None
        result, _ = _extract_or_mark(idx, chunks[idx], prev, nxt, raw_prev=True)
        new = [split_evidence(r) for r in (result.get('reactions', []) if result else [])]
        with lock:
            done[idx] = new
            flat = [{'source': source_id, 'chunk_index': i, 'word_count': len(chunks[i].split()),
                     'annotation_result': r} for i in sorted(done) for r in done[i]]
            with open(out, 'w') as f:
                json.dump(flat, f, indent=2)
            status = 'FAILED' if result is None else f'{len(new)} reaction(s)'
            print(f"  chunk {idx+1}/{len(chunks)}: {status}  |  {usage_line()}", flush=True)

    if chunks:
        work(0)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, range(1, len(chunks))))
    return [{'source': source_id, 'chunk_index': i, 'word_count': len(chunks[i].split()), 'annotation_result': r}
            for i in sorted(done) for r in done[i]]

def should_continue(state):
    return 'done' if state['current_chunk_index'] >= len(state['chunks']) else 'continue'

builder = StateGraph(PipelineState)
builder.add_node('process_chunk', process_chunk)
builder.add_edge(START, 'process_chunk')
builder.add_conditional_edges('process_chunk', should_continue, {'continue': 'process_chunk', 'done': END})
graph = builder.compile(checkpointer=InMemorySaver())

def cited_only_path(out):
    """Where the reactions a research paper only cites are put aside (next to the extraction file)."""
    return out[:-len('_extraction.json')] + '_cited_only.json'


def output_path(paper, gene=None, tag=None):
    # gene is only a filename label — a PMID stem carries no gene on its own.
    stem = fetcher.output_stem(paper, gene=gene)
    # --tag keeps a re-run in its own file so two extractions can be compared
    label = f'{stem}' + (f'_{tag}' if tag else '')
    return os.path.join(RESULTS_DIR, f'{label}_extraction.json')


def status_path(out):
    """How the paper's extraction ended (complete / incomplete / no_text), next to the extraction file. A resumed run
    reuses the extraction only when this says complete and the fingerprint matches."""
    return out[:-len('_extraction.json')] + '_extraction.status.json'


def fingerprint(paper, review=False):
    """What the extraction depends on: the paper (a PDF's content, else its id), the model and its settings, the prompt,
    serial or concurrent chunks, and review vs research. If any of them changes, a resumed run extracts again."""
    import hashlib
    h = hashlib.sha256()
    path = paper if os.path.isfile(paper) else os.path.join(PAPERS_DIR, paper)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
    else:
        h.update(str(paper).strip().encode())
    h.update(json.dumps([MODEL_KW, pm.build_extraction_prompt('', split=True)[0], WORKERS == 1, bool(review)],
                        sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


def write_status(out, state, fp, **extra):
    tmp = status_path(out) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'state': state, 'fingerprint': fp, 'finished_at': time.time(), **extra}, f, indent=2)
    os.replace(tmp, status_path(out))


def read_status(out):
    try:
        with open(status_path(out)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def extract_paper(paper, gene=None, tag=None, overwrite=False, pmcid=None, review=False, resume=False):
    out = output_path(paper, gene, tag)
    fp = fingerprint(paper, review)
    if resume:
        st = read_status(out)
        if st and st.get('state') == 'complete' and st.get('fingerprint') == fp and os.path.exists(out):
            print(f"[resume] {paper}: using the complete extraction from an earlier run "
                  f"({st.get('reactions', '?')} reaction(s)) -> {out}", flush=True)
            return
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f'{out} exists — pass --tag <name> to write a separate '
                              f'file, or --overwrite to replace it')
    if os.path.exists(status_path(out)):
        os.remove(status_path(out))          # an aborted run then leaves no "complete" behind for a half-written file
    with _failed_lock:
        FAILED_CHUNKS.clear()
    # pmcid comes from the up-front triage, so PMC is not asked to resolve it twice
    try:
        source_id, results_text, how = fetcher.load_source(paper, client=client, model=MODEL_NAME,
                                                           whole_paper_fallback=True, pmcid=pmcid)
    except requests.RequestException as e:   # the network, not the paper: try again on the next run
        write_status(out, 'incomplete', fp, reason=f'could not fetch the paper ({type(e).__name__}: {e})')
        raise
    except ValueError as e:                   # no usable text (not open access, no body): trying again will not help
        write_status(out, 'no_text', fp, reason=str(e))
        raise
    chunks = splitter.split_text(results_text)
    print(f"[setup] {source_id}: Results={len(results_text.split()):,}w "
          f"(via {how}) -> {len(chunks)} chunks", flush=True)

    t0 = time.time()
    if WORKERS > 1:
        found = extract_concurrently(source_id, out, chunks, WORKERS)
        n_chunks = len(chunks)
    else:
        config = {'configurable': {'thread_id': label}, 'recursion_limit': len(chunks) + 5}
        init = {'source': source_id, 'out': out, 'chunks': chunks, 'current_chunk_index': 0,
                'reactions_found': [], 'recent_notes': []}
        final = graph.invoke(init, config=config)
        found, n_chunks = final['reactions_found'], final['current_chunk_index']
    # Reactome needs direct evidence: from a research paper keep only what the paper itself shows. A review reports
    # cited work by nature, so nothing is left out for one. What is left out is kept next to the extraction file.
    left_out = []
    if not review:
        found, left_out = split_cited_only(found)
        with open(out, 'w') as f:
            json.dump(found, f, indent=2)
    with open(cited_only_path(out), 'w') as f:      # always written, so an earlier run's list never lingers
        json.dump(left_out, f, indent=2)
    if left_out:
        print(f"[cited] left out {len(left_out)} reaction(s) the paper only cites from earlier work "
              f"(a research paper needs direct evidence; pass --review for a review article): "
              + '; '.join(r['annotation_result'].get('name', '') for r in left_out), flush=True)
    with _failed_lock:
        failed = sorted(FAILED_CHUNKS)
    if failed:
        write_status(out, 'incomplete', fp, chunks=len(chunks), failed_chunks=failed, reactions=len(found),
                     reason=f'{len(failed)} of {len(chunks)} chunk(s) could not be extracted (the model could not be '
                            f'reached); run again to finish')
        print(f"[incomplete] {source_id}: {len(failed)} of {len(chunks)} chunk(s) failed: {[i + 1 for i in failed]}; "
              f"run again to extract this paper completely", flush=True)
    else:
        write_status(out, 'complete', fp, chunks=len(chunks), reactions=len(found))
    print(f"[done] {source_id}: {n_chunks} chunks, {len(found)} reaction(s), "
          f"{time.time()-t0:.0f}s ({WORKERS} worker(s)) -> {out}", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract reactions from PDFs or PubMed IDs.')
    ap.add_argument('sources', nargs='*',
                    help='PDF filenames in data/papers/, PMIDs, or PMCIDs '
                         '(default: every *.pdf in data/papers/)')
    ap.add_argument('--gene', help='gene label for output filenames (PMIDs carry no gene)')
    ap.add_argument('--pmid-file', help='file with one PMID per line; blank lines and # comments ignored')
    ap.add_argument('--tag', help='suffix for the output filename, e.g. --tag v2 -> '
                                 'results/<stem>_v2_extraction.json (keeps a re-run '
                                 'separate from an earlier extraction)')
    ap.add_argument('--overwrite', action='store_true',
                    help='allow replacing an existing extraction file (refused by default)')
    ap.add_argument('--resume', action='store_true',
                    help='keep a paper\'s extraction from an earlier run when it finished completely with the same '
                         'paper, model, prompt and settings (see <stem>_extraction.status.json); extract the others')
    ap.add_argument('--workers', type=int, help='chunks extracted concurrently (default $EXTRACT_WORKERS or 16; '
                    '1 = the original serial mode, where each chunk sees notes on the previous two)')
    ap.add_argument('--review', action='store_true',
                    help='the papers are review articles: keep reactions they only cite from earlier work (a research '
                         'paper keeps only reactions its own results show; the rest go to <stem>_cited_only.json)')
    a = ap.parse_args()
    if a.workers:
        WORKERS = a.workers

    papers = list(a.sources)
    if a.pmid_file:
        with open(a.pmid_file) as f:
            papers += [ln.split('#')[0].strip() for ln in f if ln.split('#')[0].strip()]
    if not papers:
        papers = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(PAPERS_DIR, '*.pdf')))]
    papers = list(dict.fromkeys(papers))          # dedupe, keep the given order

    # Triage every PMID/PMCID in ONE ID Converter request before extracting anything:
    # a paper PMC has no full text for can never enter this pipeline, so report it as a
    # miss up front instead of discovering it paper-by-paper. Local PDFs bypass this.
    ids = [p for p in papers if fetcher.is_pmid(p) or fetcher.is_pmcid(p)]
    try:
        triage = fetcher.triage_pmids(ids) if ids else {'hits': {}, 'misses': {}}
    except requests.RequestException as e:   # no network: every PMID is incomplete, to be tried again
        for p in ids:
            write_status(output_path(p, a.gene, a.tag), 'incomplete', fingerprint(p, a.review),
                         reason=f'could not reach PubMed Central ({type(e).__name__}: {e})')
        print(f"[skip] PubMed Central could not be reached: {e}", flush=True)
        triage = {'hits': {}, 'misses': {p: 'pmc-unreachable' for p in ids}}
    misses = dict(triage['misses'])
    for p, why in triage['misses'].items():
        if why != 'pmc-unreachable':          # PMC has no full text for it: trying again will not help
            write_status(output_path(p, a.gene, a.tag), 'no_text', fingerprint(p, a.review), reason=str(why))
    todo = [p for p in papers if p not in misses]

    print(f"[run] extracting {len(todo)} paper(s): {todo}", flush=True)
    for paper in todo:
        print(f"\n########## {paper} ##########", flush=True)
        try:
            extract_paper(paper, gene=a.gene, tag=a.tag, overwrite=a.overwrite,
                          pmcid=triage['hits'].get(paper), review=a.review, resume=a.resume)
        except FileExistsError as e:
            print(f"[skip] {paper}: {e}", flush=True)
            misses[paper] = 'output-exists'
        except (ValueError, requests.RequestException) as e:
            # A PMCID only promises a record, not usable text: PMC may serve no <body>
            # (paywalled), no Results section, or the network may fail. Those surface
            # here rather than in triage, so they join the same miss report.
            print(f"[skip] {paper}: {type(e).__name__}: {e}", flush=True)
            misses[paper] = f'no-usable-full-text ({type(e).__name__})'

    print(f"\n[all done] {len(papers) - len(misses)}/{len(papers)} paper(s) extracted "
          f"|  {usage_line()}", flush=True)
    if misses:
        print(f"[miss] {len(misses)} paper(s) produced no extraction:", flush=True)
        for p in papers:
            if p in misses:
                print(f"    {p}: {misses[p]}", flush=True)
        print("[miss] articles missing from PMC are the candidates for the "
              "PDF-upload route", flush=True)
