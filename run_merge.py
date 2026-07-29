"""Exhaustive semantic merger for extracted reactions from any PDF.
Dedups to unique signatures, prints them, then LLM-judge merges semantically
identical reactions (union-find, exhaustive pairwise). Saves merged set.

Usage:
    python run_merge.py                                     # defaults to PINK1
    python run_merge.py znfx1                               # gene prefix -> results/znfx1_2prev1next_extraction.json
    python run_merge.py results/znfx1_2prev1next_extraction.json   # explicit input path
    python run_merge.py znfx1 --out results/znfx1_custom_merged.json
"""
import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)
import anthropic

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-5'


def resolve_paths(target, out=None):
    """Accept either an extraction-file path or a gene prefix and return (IN, OUT)."""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    cand = target if os.path.isabs(target) else os.path.join(PROJECT_ROOT, target)
    if target.endswith('.json') or os.path.isfile(cand):
        in_path = cand
    else:
        # treat as a gene prefix, e.g. "znfx1" -> results/znfx1_2prev1next_extraction.json
        in_path = os.path.join(results_dir, f'{target.lower()}_2prev1next_extraction.json')
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
    reg_str = ', '.join(f"{r.get('regulationType')} by {r.get('regulator')}" for r in regs) if regs else 'none'
    return (f"name: {a.get('name','')}\n"
            f"type: {a.get('reactionType','')}\n"
            f"input: {', '.join(a.get('input',[]) or []) or 'none'}\n"
            f"output: {', '.join(a.get('output',[]) or []) or 'none'}\n"
            f"catalyst: {ca.get('catalyst') or 'none'} ({ca.get('molecularFunction') or 'none'})\n"
            f"regulation: {reg_str}\n"
            f"compartment: {a.get('compartment') or 'none'}\n"
            f"relationships: {' | '.join(a.get('relationships',[]) or []) or 'none'}")

def reactions_are_duplicate(a1, a2):
    prompt = f"""You are a Reactome biocurator deduplicating extracted reactions.
Two records are DUPLICATES if they describe the SAME underlying biochemical event — the same
molecular transformation of the same participants — EVEN IF they differ in:
  - wording or synonyms
  - level of detail / granularity (e.g. one adds a qualifier like "leading to degradation")
  - reactionType label (e.g. one tagged "blackBoxEvent", the other "transition")
They are DIFFERENT reactions only if they involve different participants or products, a
different catalyst, a different compartment, or a different molecular action (e.g.
phosphorylation vs ubiquitination), or if they merely share one participant.

Reaction A:
{_reaction_brief(a1)}

Reaction B:
{_reaction_brief(a2)}

Return ONLY JSON: {{"same": <true|false>, "reason": "<one sentence>"}}"""
    try:
        msg = client.messages.create(model=MODEL_NAME, max_tokens=200,
                                     messages=[{'role': 'user', 'content': prompt}])
        # Skip thinking/other blocks; take the first text block. With extended
        # thinking on, content[0] is a ThinkingBlock (no .text) and would crash.
        txt = next(b.text for b in msg.content if getattr(b, 'type', None) == 'text')
        txt = txt.strip().replace('```json','').replace('```','').strip()
        return bool(json.loads(txt).get('same', False))
    except Exception as ex:
        print(f"    dup-judge soft-fail ({type(ex).__name__}: {ex}) -> NOT duplicate", flush=True)
        return False

def merge_two(e1, e2):
    a1, a2 = e1['annotation_result'], e2['annotation_result']
    def union(l1, l2):
        seen, out = set(), []
        for x in (l1 or []) + (l2 or []):
            if _norm(x) not in seen:
                seen.add(_norm(x)); out.append(x)
        return out
    def as_list(s):
        return [] if not s else (s if isinstance(s, list) else [s])
    ca1, ca2 = a1.get('catalystActivity',{}) or {}, a2.get('catalystActivity',{}) or {}
    merged_ca = ca1 if sum(1 for v in ca1.values() if v) >= sum(1 for v in ca2.values() if v) else ca2
    seen_t, summ = set(), []
    for s in as_list(a1.get('summation')) + as_list(a2.get('summation')):
        if isinstance(s, dict) and _norm(s.get('text')) and _norm(s.get('text')) not in seen_t:
            seen_t.add(_norm(s.get('text'))); summ.append(s)
    names1 = a1.get('merged_names') or [a1.get('name', '')]
    names2 = a2.get('merged_names') or [a2.get('name', '')]
    merged = {
        'name': a1.get('name') or a2.get('name',''),
        'reactionType': a1.get('reactionType') or a2.get('reactionType',''),
        'input': union(a1.get('input',[]), a2.get('input',[])),
        'output': union(a1.get('output',[]), a2.get('output',[])),
        'catalystActivity': merged_ca,
        'regulatedBy': (a1.get('regulatedBy',[]) or []) + (a2.get('regulatedBy',[]) or []),
        'compartment': a1.get('compartment') or a2.get('compartment'),
        'condition': a1.get('condition') or a2.get('condition'),
        'summation': summ,
        'relationships': union(a1.get('relationships',[]), a2.get('relationships',[])),
        # combine the source-text excerpts from both copies so the merged reaction cites every place it was found
        'evidence': union(a1.get('evidence',[]), a2.get('evidence',[])),
        'confidence': round(((a1.get('confidence') or 0)+(a2.get('confidence') or 0))/2, 3),
        'merged_names': sorted(set(names1 + names2)),   # full cluster membership
    }
    return {'source': e1['source'], 'annotation_result': merged}

# 2) exhaustive union-find over unique set — judge ALL pairs CONCURRENTLY
MAX_WORKERS = 12          # parallel judge calls (independent, so safe to run concurrently)
n = len(unique)
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def unite(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra

pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
         if unique[i]['source'] == unique[j]['source']]
print(f"[merge] exhaustive judge over {n} reactions ({len(pairs)} pairs), {MAX_WORKERS} concurrent...", flush=True)
t0 = time.time(); done = [0]

def judge(pair):
    i, j = pair
    same = reactions_are_duplicate(unique[i]['annotation_result'], unique[j]['annotation_result'])
    done[0] += 1
    if done[0] % 100 == 0:
        print(f"  judged ~{done[0]}/{len(pairs)} pairs | {time.time()-t0:.0f}s", flush=True)
    return (i, j, same)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    verdicts = list(pool.map(judge, pairs))

# apply the "same" verdicts to union-find AFTER all judgments (order-independent)
for i, j, same in verdicts:
    if same:
        unite(i, j)
print(f"[merge] judged {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)

clusters = {}
for i in range(n):
    clusters.setdefault(find(i), []).append(i)
merged_results = []
for idxs in clusters.values():
    cur = unique[idxs[0]]
    for k in idxs[1:]:
        cur = merge_two(cur, unique[k])
    merged_results.append(cur)

with open(OUT, 'w') as f:
    json.dump(merged_results, f, indent=2)

print(f"[done] {n} -> {len(merged_results)} reactions | {len(pairs)} pairs judged | {time.time()-t0:.0f}s", flush=True)
print("[done] merged reactions:", flush=True)
for i, e in enumerate(merged_results, 1):
    a = e['annotation_result']
    names = a.get('merged_names') or [a.get('name','')]
    print(f"   {i:2d}. {a.get('name','')}"
          + (f"   (merged {len(names)} variants)" if len(names) > 1 else ""), flush=True)
print(f"[done] saved -> {OUT}", flush=True)
