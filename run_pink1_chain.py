"""Pathway chaining over the merged PINK1 reactions — LLM owns the ordering.
The LLM arranges reactions into the order they occur BIOLOGICALLY (not article order),
using deterministic output->input matches only as supporting evidence, and returns
immediate precedingEvent links. Output is presented/saved in that biological order.
"""
import os, json
PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)
import anthropic

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-5'
IN  = os.path.join(PROJECT_ROOT, 'results', 'pink1_2prev1next_merged.json')
OUT = os.path.join(PROJECT_ROOT, 'results', 'pink1_2prev1next_pathway.json')

m = json.load(open(IN))
rx = [e['annotation_result'] for e in m]
names = [a.get('name', f'reaction_{i}') for i, a in enumerate(rx)]
n = len(rx)

def norm(x): return ' '.join((x or '').lower().split())
outs  = [set(norm(o) for o in (a.get('output') or [])) for a in rx]
ins   = [set(norm(i) for i in (a.get('input')  or [])) for a in rx]
conds = [a.get('condition') or '' for a in rx]

# --- Condition bucketing (generalizable): let the LLM cluster this paper's distinct
#     condition strings into a minimal set of coarse buckets. No hard-coded, paper-specific
#     keywords — the buckets are derived from whatever conditions the article produced.
distinct_conds = sorted({c.strip() for c in conds if c.strip()})
bucket_of = {}
if distinct_conds:
    bprompt = ("These are the distinct experimental/biological conditions extracted from one "
               "paper's reactions. Group them into the MINIMAL set of coarse, mutually-exclusive "
               "biological conditions: merge conditions that describe the same underlying state or "
               "experimental context (even if worded differently); keep genuinely different "
               "contexts separate.\n\nReturn ONLY JSON, no markdown:\n"
               '{"mapping": {"<exact condition string>": "<short bucket label>"}}\n'
               "Map EVERY condition below to a bucket label, using the exact condition text as the key.\n\n"
               "Conditions:\n" + "\n".join("- " + c for c in distinct_conds))
    try:
        bmsg = client.messages.create(model=MODEL_NAME, max_tokens=3000,
                                      messages=[{'role': 'user', 'content': bprompt}])
        braw = ''.join(b.text for b in bmsg.content if getattr(b, 'type', None) == 'text').strip()
        braw = braw.replace('```json', '').replace('```', '').strip()
        if not braw.startswith('{'):
            s, e = braw.find('{'), braw.rfind('}')
            if s != -1 and e != -1: braw = braw[s:e+1]
        bucket_of = {k: str(v) for k, v in json.loads(braw).get('mapping', {}).items()}
    except Exception as ex:
        print(f"[buckets] clustering failed ({type(ex).__name__}: {ex}) — each condition its own bucket", flush=True)

buckets = [bucket_of.get(c.strip(), '') for c in conds]
_blabels = sorted(set(b for b in buckets if b))
print(f"[buckets] {len(distinct_conds)} distinct conditions -> {len(_blabels)} buckets: {_blabels}", flush=True)

def same_condition(i, j):
    # gate on the LLM-derived coarse bucket; a blank bucket links to anything (shared/upstream)
    return (not buckets[i]) or (not buckets[j]) or (buckets[i] == buckets[j])

# deterministic output->input matches, gated by condition — evidence for the LLM (not the final order)
det = set()
for i in range(n):
    for j in range(n):
        if i != j and outs[i] & ins[j] and same_condition(i, j):
            det.add((i, j))
print(f"[evidence] {len(det)} condition-consistent output->input matches passed to the LLM", flush=True)

listing = "\n".join(
    f"{i}: {names[i]}   [bucket: {buckets[i] or 'unspecified'}]  [condition: {rx[i].get('condition') or 'unspecified'}]"
    f"\n     in:  {', '.join(rx[i].get('input',[]) or []) or 'none'}"
    f"\n     out: {', '.join(rx[i].get('output',[]) or []) or 'none'}"
    for i in range(n)
)
known = "\n".join(f"{i} -> {j}" for (i, j) in sorted(det)) or "(none)"

prompt = f"""You are a Reactome biocurator assembling a biochemical pathway from extracted reactions.

Below are {n} reactions (indexed), each with input and output molecular entities.

Your task: arrange them into the order they occur BIOLOGICALLY in the cell — earliest cause
first, following cause -> effect. This is NOT the order they appeared in the paper; reorder
freely based on the biology. A molecular entity must be produced by an earlier reaction before
a later reaction can consume it. (For example, mitochondrial import happens before the imported
protein can be cleaved.)

A reaction A immediately precedes reaction B if A's OUTPUT is consumed as B's INPUT, even when
the entities are worded differently (e.g. "ubiquitin (phosphorylated at Ser65)" ==
"phospho-ubiquitin").

CRITICAL — link only within a CONDITION BUCKET. Each reaction is tagged with a coarse
[bucket] that groups the fine-grained conditions into the same biological context. Only
create a precedingEvent link between two reactions in the SAME bucket — never link across
different buckets (they are mutually exclusive arms, e.g. a healthy-state reaction vs a
damaged-state reaction vs an in-vitro assay). A reaction whose bucket is "unspecified" is a
shared/upstream step and may link across buckets.

The pathway is a BRANCHED graph, not a single line: one reaction may immediately precede
SEVERAL reactions (a fork), and several reactions may converge on ONE downstream reaction.
Emit every real fork and convergence. Independent experimental observations (mutant/in-vitro
assays) need not lie on the main chain.

Reactions:
{listing}

Exact output->input matches already found (A -> B, by index) as supporting evidence:
{known}

Return ONLY JSON, no markdown:
{{
  "order": [<every reaction index, from earliest to latest biological step, each exactly once>],
  "edges": [[<before_index>, <after_index>], ...]
}}
"edges" are IMMEDIATE step-to-step precedingEvent links only (no transitive/skip links)."""

order, llm_edges = list(range(n)), set()
try:
    msg = client.messages.create(model=MODEL_NAME, max_tokens=6000,
                                 messages=[{'role': 'user', 'content': prompt}])
    raw = ''.join(b.text for b in msg.content if getattr(b, 'type', None) == 'text').strip()
    txt = raw.replace('```json', '').replace('```', '').strip()
    if not txt.startswith('{'):
        s, e = txt.find('{'), txt.rfind('}')
        if s != -1 and e != -1: txt = txt[s:e+1]
    obj = json.loads(txt)
    # validate order: every index exactly once (repair if the model slips)
    seen, cleaned = set(), []
    for k in obj.get('order', []):
        k = int(k)
        if 0 <= k < n and k not in seen:
            seen.add(k); cleaned.append(k)
    cleaned += [k for k in range(n) if k not in seen]   # append any omitted
    order = cleaned
    for pair in obj.get('edges', []):
        a, b = int(pair[0]), int(pair[1])
        if 0 <= a < n and 0 <= b < n and a != b:
            llm_edges.add((a, b))
    print(f"[llm] biological order received; {len(llm_edges)} precedingEvent edges", flush=True)
except Exception as ex:
    print(f"[llm] FAILED ({type(ex).__name__}: {ex}) — falling back to evidence order", flush=True)

# keep only condition-consistent edges (defensive: drop cross-condition links the LLM may add)
edges = {(a, b) for (a, b) in (det | llm_edges) if same_condition(a, b)}
preds = {i: [p for p in order if (p, i) in edges] for i in range(n)}
succ  = {i: [s for s in order if (i, s) in edges] for i in range(n)}

print("\n=== reactions in BIOLOGICAL order ===", flush=True)
for rank, i in enumerate(order, 1):
    ps = preds[i]; c = rx[i].get('condition') or 'unspecified'
    pstr = "  <after: " + " ; ".join(names[p] for p in ps) + ">" if ps else "  <ENTRY POINT>"
    print(f"{rank:2d}. [{c}] {names[i]}{pstr}", flush=True)

# indented branch tree — forks (one -> several) and merges (…seen) are visible
print("\n=== branch tree ===", flush=True)
def render(i, depth, seen):
    line = "  " * depth + "- " + names[i]
    if i in seen:
        print(line + "  (…converges here)", flush=True); return
    print(line, flush=True); seen.add(i)
    for s in succ[i]:
        render(s, depth + 1, seen)
seen = set()
for r in [i for i in order if not preds[i]]:
    print(f"[{rx[r].get('condition') or 'unspecified'}]", flush=True)
    render(r, 0, seen)
for i in order:              # anything left (feedback cycles)
    if i not in seen:
        render(i, 0, seen)

# save in biological order with a precedingEvent field
out = []
for i in order:
    a = dict(rx[i]); a['precedingEvent'] = [names[p] for p in preds[i]]
    out.append({'source': m[i].get('source'), 'annotation_result': a})
json.dump(out, open(OUT, 'w'), indent=2)
print(f"\n[done] saved biologically-ordered pathway -> {OUT}", flush=True)
