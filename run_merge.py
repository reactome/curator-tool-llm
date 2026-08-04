"""Exhaustive semantic merger for extracted reactions from any PDF.
Dedups to unique signatures, prints them, then LLM-judge merges semantically
identical reactions (union-find, exhaustive pairwise). Saves merged set.

Usage:
    python run_merge.py                                     # defaults to PINK1
    python run_merge.py znfx1                               # gene prefix -> results/znfx1_2prev1next_extraction.json
    python run_merge.py results/znfx1_2prev1next_extraction.json   # explicit input path
    python run_merge.py znfx1 --out results/znfx1_custom_merged.json
"""
import os, sys, json, re, time, argparse
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
            f"condition: {a.get('condition') or 'none'}\n"
            f"compartment: {a.get('compartment') or 'none'}\n"
            f"relationships: {' | '.join(a.get('relationships',[]) or []) or 'none'}")

# pairs whose verdict we never got — they default to NOT duplicate, so the
# final count may under-merge. Reported at the end rather than lost in the log.
failures = []

def reactions_are_duplicate(a1, a2):
    prompt = f"""You are a Reactome biocurator deduplicating extracted reactions.
Two records are DUPLICATES if they describe the SAME underlying biochemical event — the same
molecular transformation of the same participants — EVEN IF they differ in:
  - wording or synonyms
  - level of detail / granularity (e.g. one adds a qualifier like "leading to degradation")
  - reactionType label (e.g. one tagged "blackBoxEvent", the other "transition")
They are DIFFERENT reactions — do NOT merge them — if ANY of the following differs:
  - the participants (input) or the products (output)
  - the catalyst
  - the regulators: a different regulator, or a different regulationType on the same regulator
  - the biological condition under which the reaction occurs
  - the compartment
  - the molecular action (e.g. phosphorylation vs ubiquitination)
They are also DIFFERENT if they merely share one participant.

A field that is "none" in one record and filled in the other is MISSING DATA, not a
difference — ignore it when comparing. Likewise these are the SAME, not different:
  - a domain or subunit vs its parent protein as catalyst (e.g. "ZNFX1 RZ domain" = "ZNFX1")
  - one record listing an accessory cofactor (E1/E2 enzyme, ATP, ubiquitin) the other omits
Judge the CORE transformation: which entity acts on which, and what change it undergoes.

When in doubt, do NOT merge. Two records left separate can still be merged by a curator
later; two distinct reactions merged together destroy one of them irrecoverably.

Reaction A:
{_reaction_brief(a1)}

Reaction B:
{_reaction_brief(a2)}

Answer with the "same" key FIRST so the verdict survives truncation.
Keep "reason" under 15 words.
Return ONLY JSON: {{"same": <true|false>, "reason": "<short phrase>"}}"""
    for attempt in (1, 2):
        try:
            # no temperature: claude-sonnet-5 rejects it ("`temperature` is deprecated for this model")
            msg = client.messages.create(model=MODEL_NAME, max_tokens=1000,
                                         messages=[{'role': 'user', 'content': prompt}])
            # Skip thinking/other blocks; take the first text block. With extended
            # thinking on, content[0] is a ThinkingBlock (no .text) and would crash.
            txt = next((b.text for b in msg.content if getattr(b, 'type', None) == 'text'), '')
            txt = txt.strip().replace('```json','').replace('```','').strip()
            if not txt:
                raise ValueError('no text block in response')
            try:
                return bool(json.loads(txt).get('same', False))
            except json.JSONDecodeError:
                # truncated or malformed JSON — the verdict is still readable
                m = re.search(r'"same"\s*:\s*(true|false)', txt, re.I)
                if m:
                    return m.group(1).lower() == 'true'
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
            return False

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
        restated by every variant in the cluster is ONE regulation, not N."""
        seen, out = set(), []
        for r in (l1 or []) + (l2 or []):
            if not isinstance(r, dict):
                continue
            k = (_key(r.get('regulationType')), _key(r.get('regulator')))
            if any(k) and k not in seen:
                seen.add(k); out.append(r)
        return out
    ca1, ca2 = a1.get('catalystActivity',{}) or {}, a2.get('catalystActivity',{}) or {}
    merged_ca = ca1 if sum(1 for v in ca1.values() if v) >= sum(1 for v in ca2.values() if v) else ca2
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

# _key() only collapses cosmetic duplicates. Wording-level duplicates survive it
# ("Parkin (mitochondria)" vs "Parkin (mitochondrial)"), as do restatements of the same
# fact across a cluster's summations, relationships and evidence. One LLM pass per merged
# cluster reduces every subsection to a single non-repeating section. Field shapes are
# unchanged — summation stays a list, just with one consolidated entry.
consolidate_failures = []

def consolidate_sections(a):
    """Collapse wording-level duplicates within one merged reaction's subsections.
    Returns a new annotation dict; returns `a` unchanged on any failure."""
    payload = {k: a.get(k) for k in
               ('name', 'input', 'output', 'catalystActivity', 'regulatedBy',
                'relationships', 'summation', 'evidence')}
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
{{"input": [...], "output": [...], "regulatedBy": [{{"regulationType": "...", "regulator": "..."}}],
  "relationships": [...], "evidence": [...], "summation": {{"text": "<single combined summary>"}}}}"""
    for attempt in (1, 2):
        try:
            msg = client.messages.create(model=MODEL_NAME, max_tokens=8000,
                                         messages=[{'role': 'user', 'content': prompt}])
            txt = next((b.text for b in msg.content if getattr(b, 'type', None) == 'text'), '')
            txt = txt.strip().replace('```json', '').replace('```', '').strip()
            if not txt:
                raise ValueError('no text block in response')
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
        print(
            f"\nMERGED:\n"
            f"  A: {unique[i]['annotation_result']['name']}\n"
            f"  B: {unique[j]['annotation_result']['name']}",
            flush=True
        )
        unite(i, j)
print(f"[merge] judged {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)

if failures:
    from collections import Counter
    print(f"[warn] {len(failures)}/{len(pairs)} pair(s) never got a verdict and were treated as "
          f"NOT duplicate — the merge is under-merged:", flush=True)
    for (name, detail), count in Counter(failures).most_common():
        print(f"         {count}x {name}: {detail}", flush=True)
    if len(failures) > len(pairs) * 0.1:
        print("[warn] that is a systemic failure, not noise — fix it and re-run; "
              "the output below is not trustworthy", flush=True)

clusters = {}
for i in range(n):
    clusters.setdefault(find(i), []).append(i)
merged_results = []
for idxs in clusters.values():
    cur = unique[idxs[0]]
    for k in idxs[1:]:
        cur = merge_two(cur, unique[k])
    merged_results.append(cur)

# 3) consolidate subsections of every reaction that actually merged something —
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
