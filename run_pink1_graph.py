"""Standalone run of the LangGraph memory pipeline on PINK1 only.
Mirrors the notebook cells: chunk -> extract_reactions -> graph (cumulative memory).
Prints per-chunk progress and saves extraction_results to JSON.
"""
import os, sys, json, time, importlib.util

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import fitz  # PyMuPDF
import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-4-6'

# incremental output file (written after every chunk)
OUT = os.path.join(PROJECT_ROOT, 'results', 'pink1_graph_extraction.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---- prompts module ----
prompts_path = os.path.join(PROJECT_ROOT, 'reactome_llm', 'FullTextPDFPrompts.py')
spec = importlib.util.spec_from_file_location('prompts', prompts_path)
prompts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts_mod)

# ---- extract PINK1 text ----
pdf_path = os.path.join(PROJECT_ROOT, 'data', 'papers', 'PINK1.pdf')
doc = fitz.open(pdf_path)
full_text = ''.join(page.get_text() for page in doc)
doc.close()
print(f"[setup] PINK1.pdf: {len(full_text.split()):,} words", flush=True)

# ---- chunk (cell 5) ----
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 100
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_TOKENS * 4,
    chunk_overlap=CHUNK_OVERLAP_TOKENS * 4,
    length_function=len,
    separators=['\n\n', '\n', '. ', ' ', ''],
)
chunks = splitter.split_text(full_text)
print(f"[setup] {len(chunks)} chunks ({CHUNK_SIZE_TOKENS}-token, {CHUNK_OVERLAP_TOKENS} overlap)", flush=True)

# ---- extract_reactions (cell 6, max_tokens raised to 16000) ----
def extract_reactions(text_chunk):
    prompt = prompts_mod.build_extraction_prompt(text_chunk)
    try:
        msg = client.messages.create(
            model=MODEL_NAME, max_tokens=16000, temperature=0.1,
            messages=[{'role': 'user', 'content': prompt}],
        )
    except Exception as e:
        print(f'    API call failed: {type(e).__name__}: {e}', flush=True)
        return None
    raw = msg.content[0].text.strip()
    cleaned = raw.replace('```json', '').replace('```', '').strip()
    if not cleaned.startswith('{'):
        s, e = cleaned.find('{'), cleaned.rfind('}')
        if s != -1 and e != -1:
            cleaned = cleaned[s:e+1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        trunc = msg.stop_reason == 'max_tokens'
        print(f'    JSON parse failed: {e}{"  [TRUNCATED - hit max_tokens]" if trunc else ""}', flush=True)
        return None

# ---- graph (memory cell + cumulative context) ----
class PipelineState(TypedDict):
    source: str
    chunks: List[str]
    current_chunk_index: int
    reactions_found: List[Dict]
    context_summary: str

def process_chunk(state: PipelineState) -> PipelineState:
    idx = state["current_chunk_index"]
    chunk = state["chunks"][idx]
    context = state.get("context_summary", "")
    context_prefix = f"Reactions found in previous chunks:\n{context}\n\n" if context else ""

    result = extract_reactions(context_prefix + chunk)
    new_reactions = result.get("reactions", []) if result else []

    new_context = context
    if new_reactions:
        lines = []
        for r in new_reactions:
            ins = ', '.join(r.get('input', []) or [])
            outs = ', '.join(r.get('output', []) or [])
            lines.append(f"{r.get('name', '')} [in: {ins} | out: {outs}]")
        new_context += f"\nChunk {idx}: " + '; '.join(lines)

    status = "FAILED" if result is None else f"{len(new_reactions)} reaction(s)"
    print(f"  chunk {idx+1}/{len(state['chunks'])}: {status}", flush=True)

    # --- previous version (no incremental save) ---
    # return {
    #     **state,
    #     "current_chunk_index": idx + 1,
    #     "reactions_found": state["reactions_found"] + [
    #         {"source": state["source"], "chunk_index": idx,
    #          "word_count": len(chunk.split()), "annotation_result": r}
    #         for r in new_reactions
    #     ],
    #     "context_summary": new_context,
    # }

    all_reactions = state["reactions_found"] + [
        {"source": state["source"], "chunk_index": idx,
         "word_count": len(chunk.split()), "annotation_result": r}
        for r in new_reactions
    ]
    # incremental save after every chunk — survives a mid-run stop
    with open(OUT, 'w') as f:
        json.dump(all_reactions, f, indent=2)

    return {
        **state,
        "current_chunk_index": idx + 1,
        "reactions_found": all_reactions,
        "context_summary": new_context,
    }

def should_continue(state: PipelineState) -> str:
    return "done" if state["current_chunk_index"] >= len(state["chunks"]) else "continue"

builder = StateGraph(PipelineState)
builder.add_node("process_chunk", process_chunk)
builder.add_edge(START, "process_chunk")
builder.add_conditional_edges("process_chunk", should_continue,
                              {"continue": "process_chunk", "done": END})
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ---- run PINK1 ----
fname = "PINK1.pdf"
config = {"configurable": {"thread_id": fname}, "recursion_limit": len(chunks) + 5}
initial_state = {"source": fname, "chunks": chunks, "current_chunk_index": 0,
                 "reactions_found": [], "context_summary": ""}

print(f"[run] starting graph on {fname} ({len(chunks)} chunks)...", flush=True)
t0 = time.time()
final_state = graph.invoke(initial_state, config=config)
dt = time.time() - t0

extraction_results = final_state["reactions_found"]
n_checkpoints = len(list(checkpointer.list(config)))
print(f"[done] {final_state['current_chunk_index']} chunks processed in {dt:.0f}s", flush=True)
print(f"[done] {len(extraction_results)} reaction(s) extracted, {n_checkpoints} checkpoint(s)", flush=True)

out = os.path.join(PROJECT_ROOT, 'results', 'pink1_graph_extraction.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(extraction_results, f, indent=2)
print(f"[done] saved -> {out}", flush=True)
