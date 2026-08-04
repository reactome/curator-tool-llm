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

Writes results/<stem>_2prev1next_extraction.json per paper (incremental, per chunk).
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
    print(f"  chunk {idx+1}/{len(state['chunks'])}: {status}", flush=True)

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

def extract_paper(paper, gene=None):
    # gene is only a filename label — a PMID stem carries no gene on its own.
    stem = fetcher.output_stem(paper, gene=gene)
    out = os.path.join(RESULTS_DIR, f'{stem}_2prev1next_extraction.json')
    source_id, results_text, how = fetcher.load_source(paper, client=client, model=MODEL_NAME)
    chunks = splitter.split_text(results_text)
    print(f"[setup] {source_id}: Results={len(results_text.split()):,}w "
          f"(via {how}) -> {len(chunks)} chunks", flush=True)

    config = {'configurable': {'thread_id': f'{stem}-2prev1next'}, 'recursion_limit': len(chunks) + 5}
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
    a = ap.parse_args()

    papers = list(a.sources)
    if a.pmid_file:
        with open(a.pmid_file) as f:
            papers += [ln.split('#')[0].strip() for ln in f if ln.split('#')[0].strip()]
    if not papers:
        papers = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(PAPERS_DIR, '*.pdf')))]

    print(f"[run] extracting {len(papers)} paper(s): {papers}", flush=True)
    skipped = []
    for paper in papers:
        print(f"\n########## {paper} ##########", flush=True)
        try:
            extract_paper(paper, gene=a.gene)
        except (ValueError, requests.RequestException) as e:
            # No PMCID, no <body> (paywalled), no Results section, or a network
            # failure — skip this paper and keep going through the batch.
            print(f"[skip] {paper}: {type(e).__name__}: {e}", flush=True)
            skipped.append(paper)
    print(f"\n[all done]" + (f" — skipped {len(skipped)}: {skipped}" if skipped else ""), flush=True)
