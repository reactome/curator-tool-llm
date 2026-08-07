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

Writes results/<stem>_2prev1next[_<tag>]_extraction.json per paper (incremental, per
chunk). Refuses to overwrite an existing extraction unless --overwrite is given.
"""
import os, sys, json, time, glob, argparse, importlib.util
PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
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

PAPERS_DIR = os.path.join(PROJECT_ROOT, 'data', 'papers')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-5'
PREV_WINDOW = 2   # previous chunks' reaction-notes kept as backward context

# Token accounting across every API call this run makes, so a pipeline run can be
# costed. Note a Results-section LLM fallback inside FullTextPDFSections is not
# counted here — only the extraction calls made below.
USAGE = {'calls': 0, 'in': 0, 'out': 0, 'cache_read': 0, 'cache_write': 0}


def track_usage(msg):
    u = getattr(msg, 'usage', None)
    if not u:
        return
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

def extract_reactions(current_chunk, prev_contexts, next_context):
    prompt = pm.build_extraction_prompt(current_chunk, prev_contexts=prev_contexts, next_context=next_context)
    try:
        msg = client.messages.create(model=MODEL_NAME, max_tokens=16000,
                                     messages=[{'role': 'user', 'content': prompt}])
    except Exception as e:
        print(f'    API call failed: {type(e).__name__}: {e}', flush=True); return None
    track_usage(msg)
    raw = ''.join(b.text for b in msg.content if getattr(b, 'type', None) == 'text').strip()
    raw = raw.replace('```json', '').replace('```', '').strip()
    if not raw.startswith('{'):
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1: raw = raw[s:e+1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f'    JSON parse failed: {e}{"  [TRUNCATED]" if msg.stop_reason=="max_tokens" else ""}', flush=True)
        return None

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

    result = extract_reactions(chunk, prev_contexts, next_context)
    new_reactions = result.get('reactions', []) if result else []
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

def should_continue(state):
    return 'done' if state['current_chunk_index'] >= len(state['chunks']) else 'continue'

builder = StateGraph(PipelineState)
builder.add_node('process_chunk', process_chunk)
builder.add_edge(START, 'process_chunk')
builder.add_conditional_edges('process_chunk', should_continue, {'continue': 'process_chunk', 'done': END})
graph = builder.compile(checkpointer=InMemorySaver())

def extract_paper(paper, gene=None, tag=None, overwrite=False, pmcid=None):
    # gene is only a filename label — a PMID stem carries no gene on its own.
    stem = fetcher.output_stem(paper, gene=gene)
    # --tag keeps a re-run in its own file so two extractions can be compared
    label = f'{stem}_2prev1next' + (f'_{tag}' if tag else '')
    out = os.path.join(RESULTS_DIR, f'{label}_extraction.json')
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f'{out} exists — pass --tag <name> to write a separate '
                              f'file, or --overwrite to replace it')
    # pmcid comes from the up-front triage, so PMC is not asked to resolve it twice
    source_id, results_text, how = fetcher.load_source(paper, client=client, model=MODEL_NAME,
                                                       pmcid=pmcid)
    chunks = splitter.split_text(results_text)
    print(f"[setup] {source_id}: Results={len(results_text.split()):,}w "
          f"(via {how}) -> {len(chunks)} chunks", flush=True)

    config = {'configurable': {'thread_id': label}, 'recursion_limit': len(chunks) + 5}
    init = {'source': source_id, 'out': out, 'chunks': chunks, 'current_chunk_index': 0,
            'reactions_found': [], 'recent_notes': []}
    t0 = time.time()
    final = graph.invoke(init, config=config)
    print(f"[done] {source_id}: {final['current_chunk_index']} chunks, {len(final['reactions_found'])} reaction(s), "
          f"{time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract reactions from PDFs or PubMed IDs.')
    ap.add_argument('sources', nargs='*',
                    help='PDF filenames in data/papers/, PMIDs, or PMCIDs '
                         '(default: every *.pdf in data/papers/)')
    ap.add_argument('--gene', help='gene label for output filenames (PMIDs carry no gene)')
    ap.add_argument('--pmid-file', help='file with one PMID per line; blank lines and # comments ignored')
    ap.add_argument('--tag', help='suffix for the output filename, e.g. --tag v2 -> '
                                 'results/<stem>_2prev1next_v2_extraction.json (keeps a re-run '
                                 'separate from an earlier extraction)')
    ap.add_argument('--overwrite', action='store_true',
                    help='allow replacing an existing extraction file (refused by default)')
    a = ap.parse_args()

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
    triage = fetcher.triage_pmids(ids) if ids else {'hits': {}, 'misses': {}}
    misses = dict(triage['misses'])
    todo = [p for p in papers if p not in misses]

    print(f"[run] extracting {len(todo)} paper(s): {todo}", flush=True)
    for paper in todo:
        print(f"\n########## {paper} ##########", flush=True)
        try:
            extract_paper(paper, gene=a.gene, tag=a.tag, overwrite=a.overwrite,
                          pmcid=triage['hits'].get(paper))
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
