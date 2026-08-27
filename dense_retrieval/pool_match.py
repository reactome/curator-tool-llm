"""Match by construction, then rank only inside the pool.

The ranking pipeline fails because it ranks 20,003 reactions on a signal that
cannot tell reactions apart, so the correct answer loses to something that reads
similarly. This inverts the order:

    merged reaction
        |
        v
    deterministic slot match on input / output / catalyst   <- set intersection
        |                                                       no scores
        v
    CANDIDATE POOL  (usually 0-30 reactions, all genuinely compatible)
        |
        v
    cross-encoder over the pool (cosine reported alongside) <- ranks only
        |
        v
    LLM adjudication                                        <- decides
        |
        v
    one Reactome match, or none

Nothing is scored until the pool exists, so a wrong reaction cannot outrank the
right one by reading like it — it never enters the pool.

The LLM is what makes 'none' reachable. Cosine can order a pool but cannot
decline it, so without adjudication a reaction Reactome does not curate is
reported as its nearest neighbour. Answering 'none of these' is most of the
value: it is what separates a novel finding from a bad match.

Usage:
    python pool_match.py results/pink1_2prev1next_merged.json --gene PINK1
    python pool_match.py results/x_merged.json --gene ZNFX1 --no-llm
"""
import os, re, sys, csv, json, time, argparse
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import numpy as np

ap = argparse.ArgumentParser(description='Deterministic candidate pool, then cosine within it.')
ap.add_argument('merged', help='merged (or extraction) JSON to match')
ap.add_argument('--gene', required=True, help='gene whose aliases are layered on top')
ap.add_argument('--bi-encoder-model', default='all-MiniLM-L6-v2')
ap.add_argument('--cross-encoder-model', default='ncbi/MedCPT-Cross-Encoder')
ap.add_argument('--no-cross-encoder', action='store_true',
                help='order the pool by cosine alone')
ap.add_argument('--no-llm', action='store_true',
                help='stop after cosine; reports the nearest pool member, no API calls')
ap.add_argument('--llm-candidates', type=int, default=12,
                help='pool entries shown to the LLM, best cosine first (default: 12)')
ap.add_argument('--min-pool', type=int, default=5,
                help='relax tiers until the pool holds at least this many (default: 5)')
ap.add_argument('--max-pool', type=int, default=40,
                help='report a pool larger than this as too broad (default: 40)')
ap.add_argument('--neo4j-db', default='graph.db')
args = ap.parse_args()

import ReactomeNeo4jUtils as neo4jutils
neo4jutils.URI = os.getenv('REACTOME_NEO4J_URI')
neo4jutils.AUTH = (os.getenv('REACTOME_NEO4J_USER'), os.getenv('REACTOME_NEO4J_PWD'))
neo4jutils.DB = args.neo4j_db
import ReactionMatcher as RM

stem = re.sub(r'_(merged|extraction)\.json$', '', os.path.basename(args.merged))

# ── corpus: every human reaction, with its three slots ──────────────────────
CORPUS_QUERY = """
MATCH (r:ReactionLikeEvent)-[:species]->(:Species {displayName:'Homo sapiens'})
OPTIONAL MATCH (r)-[:input]->(i:PhysicalEntity)
OPTIONAL MATCH (r)-[:output]->(o:PhysicalEntity)
OPTIONAL MATCH (r)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(c:PhysicalEntity)
RETURN r.dbId AS db_id, r.stId AS st_id, r.displayName AS reaction,
       collect(DISTINCT i.displayName) AS inputs,
       collect(DISTINCT o.displayName) AS outputs,
       collect(DISTINCT c.displayName) AS catalysts
"""
from neo4j import GraphDatabase
t0 = time.time()
with GraphDatabase.driver(neo4jutils.URI, auth=neo4jutils.AUTH) as driver:
    with driver.session(database=neo4jutils.DB) as s:
        corpus = [dict(r) for r in s.run(CORPUS_QUERY)]
corpus = [c for c in corpus if not RM.is_deprecated(c['reaction'])]
for c in corpus:
    c['text'] = RM._gt_text(c)
print(f"[corpus] {len(corpus):,} human reactions in {time.time()-t0:.1f}s", flush=True)

idx = RM.fetch_global_aliases(neo4jutils)
# Layers the gene's own aliases into `idx` in place; the rows are kept only for
# the pathways they sit in (see pathway scope below).
gene_rows, _ = RM.fetch_ground_truth(neo4jutils, args.gene, verbose=False,
                                     base_idx=idx)
print(f"[alias] {len(idx):,} surface forms", flush=True)

# Cofactors are in half of Reactome, so intersecting on them pools everything.
_GENERIC = RM._STOPWORDS | {'ubiquitin chain', 'mitochondrial proteins'}


def slots(inputs, outputs, catalysts):
    def norm(names):
        out = set()
        for n in (names or []):
            out |= {e for e in RM.resolve_entity(n, idx) if e and e not in _GENERIC}
            out |= {idx.canon(m) for m in RM.split_entity(n, idx)[1]
                    if idx.knows(m) and idx.canon(m) not in _GENERIC}
        return out
    cat = norm(catalysts)
    return {'input': norm(inputs), 'output': norm(outputs), 'catalyst': cat}


c_slots = [slots(c['inputs'], c['outputs'], c['catalysts']) for c in corpus]

# ── pathway scope ───────────────────────────────────────────────────────────
# Shared molecules alone pool too widely: ubiquitin is a participant in hundreds
# of reactions across unrelated biology. The pathways the gene's own reactions
# sit in are a cheap prior for where its paper's reactions belong, so in-scope
# candidates are preferred and the rest are only reached if the pool is thin.
# It is a preference, not a filter — a genuinely cross-pathway match is still
# findable, just ranked behind.
PATHWAY_SCOPE_QUERY = """
MATCH (p:Pathway)-[:hasEvent*1..6]->(r:ReactionLikeEvent)-[:species]->
      (:Species {displayName:'Homo sapiens'})
WHERE p.displayName IN $pathways
RETURN DISTINCT r.stId AS st_id
"""
scope_paths = sorted({p for row in (gene_rows or []) for p in (row.get('pathways') or []) if p})
in_scope = set()
if scope_paths:
    with GraphDatabase.driver(neo4jutils.URI, auth=neo4jutils.AUTH) as driver:
        with driver.session(database=neo4jutils.DB) as s:
            in_scope = {r['st_id'] for r in s.run(PATHWAY_SCOPE_QUERY, pathways=scope_paths)}
scoped = [c['st_id'] in in_scope for c in corpus]
print(f"[scope] {len(scope_paths)} pathway(s) containing {args.gene} -> "
      f"{sum(scoped):,}/{len(corpus):,} reactions in scope", flush=True)

# ── the pool ────────────────────────────────────────────────────────────────
# Tiers, strictest first. The pool is whatever the strictest non-empty tier
# yields, so a query with a precise slot match never gets diluted by loose ones,
# and a vague query still gets something rather than nothing.
TIERS = [
    ('catalyst+input+output', ('catalyst', 'input', 'output')),
    ('catalyst+input',        ('catalyst', 'input')),
    ('catalyst+output',       ('catalyst', 'output')),
    ('input+output',          ('input', 'output')),
    ('catalyst',              ('catalyst',)),
    ('input',                 ('input',)),
]


def pool_for(q):
    """(tier label, [corpus indices]), strictest tier first.

    Keeps relaxing tiers until the pool reaches --min-pool, rather than stopping
    at the first non-empty one — a tiny strict pool can exclude the answer, which
    only qualifies a tier down. The label names the strictest tier that
    contributed, so a precise pool is still distinguishable from a loose one.
    """
    hits, seen, label = [], set(), 'none'
    for name, fields in TIERS:
        added = [j for j, cs in enumerate(c_slots)
                 # every named slot must be non-empty on both sides AND overlap
                 if all(q[f] and cs[f] and (q[f] & cs[f]) for f in fields)
                 and j not in seen]
        if added:
            if label == 'none':
                label = name
            seen.update(added)
            # In-scope first WITHIN the tier. Scope orders candidates, it never
            # drops them: a tier is always taken whole, or a correct match that
            # happens to sit outside the gene's pathways is cut before the LLM
            # ever sees it.
            hits.extend(sorted(added, key=lambda j: not scoped[j]))
        # size checked only between tiers, so a tier is never truncated
        if len(hits) >= args.min_pool:
            break
    return label, hits


merged = json.load(open(os.path.join(PROJECT_ROOT, args.merged)))
queries = [e.get('annotation_result') or {} for e in merged]
q_slots = []
for a in queries:
    ca = a.get('catalystActivity') or {}
    q_slots.append(slots(a.get('input'), a.get('output'),
                         [ca['catalyst']] if ca.get('catalyst') else []))

pools = [pool_for(q) for q in q_slots]

# ── cosine inside the pool ──────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(args.bi_encoder_model)
cache = os.path.join(PROJECT_ROOT, 'data', 'cache',
                     f"corpus_{args.bi_encoder_model.replace('/', '_')}_{len(corpus)}.npy")
os.makedirs(os.path.dirname(cache), exist_ok=True)
if os.path.exists(cache):
    emb_corpus = np.load(cache)
else:
    print(f"[embed] {len(corpus):,} reactions...", flush=True)
    emb_corpus = model.encode([c['text'] for c in corpus], batch_size=256)
    np.save(cache, emb_corpus)
emb_q = model.encode([RM.extracted_reaction_text(a) for a in queries])

cos_of = []                      # per query: {corpus idx: cosine}
for i, (_, hits) in enumerate(pools):
    sims = cosine_similarity(emb_q[i:i+1], emb_corpus[hits])[0] if hits else []
    cos_of.append(dict(zip(hits, (float(s) for s in sims))))

# ── cross-encoder over the pool ─────────────────────────────────────────────
# The bi-encoder embeds each side independently; the cross-encoder reads the pair
# together and is the better judge. It could never run over 20,003 candidates
# (one forward pass per pair), but a pool of 5-15 is nothing. Its output is an
# unbounded logit, not a similarity, so both numbers are reported and cosine is
# kept as the comparable-across-rows score.
_CROSS_ENCODER = None


def _get_cross_encoder(model_name):
    """Lazily construct and cache the cross-encoder (loading it is expensive)."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        import torch
        import transformers
        from sentence_transformers import CrossEncoder
        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            _CROSS_ENCODER = CrossEncoder(model_name)
        finally:
            transformers.logging.set_verbosity(prev)
        # force identity so predict() returns raw logits, not Sigmoid
        _CROSS_ENCODER.activation_fn = torch.nn.Identity()
    return _CROSS_ENCODER


cross_of = [{} for _ in queries]
if not args.no_cross_encoder:
    print(f"[cross] {args.cross_encoder_model} over "
          f"{sum(len(h) for _, h in pools):,} pooled pair(s)...", flush=True)
    ce = _get_cross_encoder(args.cross_encoder_model)
    t0 = time.time()
    for i, (_, hits) in enumerate(pools):
        if not hits:
            continue
        q = RM.extracted_reaction_text(queries[i])
        scores = ce.predict([(q, corpus[j]['text']) for j in hits])
        cross_of[i] = {j: float(s) for j, s in zip(hits, scores)}
    print(f"[cross] done in {time.time()-t0:.0f}s", flush=True)

# Pool order: in-scope candidates first, then by cross-encoder (else cosine).
# Scope is applied HERE rather than when building the pool, because only the top
# --llm-candidates are shown; a reaction outside the gene's pathways stays in the
# pool and remains reachable, it just does not displace an in-pathway one.
ranked = []
for i, (_, hits) in enumerate(pools):
    key = cross_of[i] if cross_of[i] else cos_of[i]
    ranked.append(sorted(((j, cos_of[i][j]) for j in hits),
                         key=lambda t: (not scoped[t[0]], -key[t[0]])))

print(f"\n{'='*104}\nPOOL SIZES\n{'='*104}")
_order_by = 'COSINE' if args.no_cross_encoder else 'CROSS-ENCODER'
print(f"{'EXTRACTED REACTION':<46} {'TIER':<22} {'POOL':>5}  TOP OF POOL (by {_order_by})")
print('-' * 104)
for i, a in enumerate(queries):
    tier, hits = pools[i]
    best = f"{ranked[i][0][1]:.3f} {corpus[ranked[i][0][0]]['reaction'][:26]}" if ranked[i] else '—'
    flag = '  <- too broad' if len(hits) > args.max_pool else ''
    print(f"{a.get('name','')[:44]:<46} {tier:<22} {len(hits):>5}  {best}{flag}")

# ── LLM adjudication ────────────────────────────────────────────────────────
# Cosine orders the pool but cannot decline it: without this step every non-empty
# pool yields a best match, and a reaction Reactome does not curate is reported as
# its nearest neighbour. Deciding "none of these" is most of the value here.
def make_row(i, j, reason):
    """One output row. `status` is explicit so an empty match column is never
    ambiguous between 'Reactome does not have this' and 'nothing was tried'."""
    tier, hits = pools[i]
    cos = dict(ranked[i])
    status = ('EMPTY-POOL' if not hits else 'MATCH' if j is not None else 'NO-MATCH')
    xe = cross_of[i]
    return {
        'status': status,
        'extracted_reaction': queries[i].get('name', ''),
        'reactome_match': corpus[j]['reaction'] if j is not None else '',
        'stid': corpus[j]['st_id'] if j is not None else '',
        'cosine': f'{cos[j]:.4f}' if j is not None else '',
        'cross_encoder': f'{xe[j]:.3f}' if j is not None and j in xe else '',
        'tier': tier,
        'pool_size': len(hits),
        'reason': reason,
    }


rows = []
if args.no_llm:
    for i, a in enumerate(queries):
        top = ranked[i][0] if ranked[i] else None
        rows.append(make_row(i, top[0] if top else None,
                             '(--no-llm: nearest pool member, not adjudicated)'))
else:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0)
    MODEL = 'claude-sonnet-5'

    PROMPT = """You are a Reactome biocurator deciding whether an extracted reaction \
already exists in Reactome.

Every candidate below already shares participants with the extracted reaction, so do \
not re-check that. Decide whether ONE of them is the SAME BIOCHEMICAL EVENT.

Match on the CHEMISTRY and the IDENTITY OF THE MOLECULES. Reactome and a paper \
describe the same event at different levels of detail, and these differences are NOT \
grounds for rejection:

- Complexed vs free form. Reactome usually names a participant in the complex or \
  modified state it occupies ('X:Y', 'mod-X'); a paper names the molecule. Same \
  molecule, same reaction.
- Specific vs general substrate. Reactome often curates one reaction over a substrate \
  SET where the paper names one member of it. Match them.
- Compartment or localisation wording, and stoichiometry of cofactors.
- Wild-type vs a mutant or mimetic used as a tool to demonstrate the same chemistry. \
  Reactome curates the biology, not the construct.
- Naming: gene symbol vs common name, phospho-site notation, residue numbering.

Be lenient about FORM, strict about the EVENT. Two reactions are different events when:

- The chemistry differs. Phosphorylation, ubiquitination, binding, dissociation, \
  cleavage and translocation are each distinct. A reaction that binds a molecule is \
  not the reaction that moves it, even when one causes the other, and a step that \
  precedes or enables the extracted event is not that event.
- The molecule acted on is genuinely a different one.
- The extracted statement is not a reaction: a phenotype, a requirement, an \
  inhibition, an activity measurement, or an observation about where something is.

Answer null when nothing matches. That is the expected answer for a real event Reactome \
does not curate, which is a useful finding rather than a failure.

EXTRACTED REACTION
  name     : {name}
  inputs   : {inputs}
  outputs  : {outputs}
  catalyst : {catalyst}
  summary  : {summation}

CANDIDATES
{candidates}

Reply with JSON only, "match" FIRST so the decision survives a truncated reply:
{{"match": <candidate number or null>, "reason": "<at most 25 words>"}}"""

    def adjudicate(i):
        a = queries[i]
        cands = ranked[i][:args.llm_candidates]
        if not cands:
            return None, 'no structurally compatible candidate'
        listing = '\n'.join(
            f"  {n+1}. {corpus[j]['reaction']}\n"
            f"     in: {', '.join(corpus[j]['inputs'][:4]) or 'none'}"
            f" | out: {', '.join(corpus[j]['outputs'][:4]) or 'none'}"
            f" | cat: {', '.join(corpus[j]['catalysts'][:2]) or 'none'}"
            for n, (j, _) in enumerate(cands))
        ca = a.get('catalystActivity') or {}
        # The extraction writes summation as a list of {text: ...}, a bare dict, or
        # a plain string depending on the paper. Take whichever shape it is.
        summ = a.get('summation') or []
        if isinstance(summ, dict):
            summ = [summ]
        elif isinstance(summ, str):
            summ = [{'text': summ}]
        prompt = PROMPT.format(
            name=a.get('name', ''),
            inputs=', '.join(a.get('input') or []) or 'none',
            outputs=', '.join(a.get('output') or []) or 'none',
            catalyst=ca.get('catalyst') or 'none',
            summation=(summ[0].get('text', '') if summ else '')[:700],
            candidates=listing)

        def to_idx(n):
            return cands[n-1][0] if isinstance(n, int) and 1 <= n <= len(cands) else None

        for _ in (1, 2):
            try:
                msg = client.messages.create(model=MODEL, max_tokens=1500,
                                             messages=[{'role': 'user', 'content': prompt}])
                txt = next((b.text for b in msg.content
                            if getattr(b, 'type', None) == 'text'), '')
                try:
                    s, e = txt.find('{'), txt.rfind('}')
                    d = json.loads(txt[s:e+1])
                    return to_idx(d.get('match')), d.get('reason', '')
                except (json.JSONDecodeError, ValueError):
                    # A reply cut off mid-reason still carries the decision, since
                    # "match" is emitted first. Recover it rather than reporting a
                    # non-match, which would read as a judgement.
                    m = re.search(r'"match"\s*:\s*(null|\d+)', txt)
                    if m:
                        g = m.group(1)
                        return (None if g == 'null' else to_idx(int(g))), '(reason truncated)'
                    continue
            except Exception as ex:
                return None, f'API error: {type(ex).__name__}'
        return None, 'unparseable LLM reply'

    print(f"\n{'='*104}\nLLM ADJUDICATION ({MODEL})\n{'='*104}")
    for i, a in enumerate(queries):
        j, reason = adjudicate(i)
        cos = dict(ranked[i])
        print(f"\n{a.get('name','')[:96]}")
        xe = cross_of[i]
        xs = f" xe {xe[j]:6.2f}" if j is not None and j in xe else ''
        print(f"   MATCH  cos {cos[j]:.3f}{xs}  {corpus[j]['reaction'][:62]}"
              if j is not None else "   NO MATCH")
        print(f"   why: {reason[:150]}")
        rows.append(make_row(i, j, reason))

out = os.path.join(PROJECT_ROOT, 'results', f'{stem}_pool_match.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

md = os.path.join(PROJECT_ROOT, 'results', f'{stem}_pool_match.md')
with open(md, 'w') as f:
    f.write(f'# Pool match — {stem}\n\n')
    for status, title in (('MATCH', 'Matched to a curated reaction'),
                          ('NO-MATCH', 'No curated reaction matched'),
                          ('EMPTY-POOL', 'No structurally compatible candidate')):
        group = [r for r in rows if r['status'] == status]
        if not group:
            continue
        f.write(f'## {title} ({len(group)})\n\n')
        for r in group:
            f.write(f"**{r['extracted_reaction']}**\n\n")
            if r['reactome_match']:
                f.write(f"- -> `{r['stid']}` {r['reactome_match']}  "
                        f"(cosine {r['cosine']}, cross-encoder {r['cross_encoder'] or 'n/a'})\n")
            f.write(f"- pool: {r['pool_size']} candidate(s) via {r['tier']}\n")
            f.write(f"- {r['reason']}\n\n")

n_match = sum(1 for r in rows if r['reactome_match'])
n_empty = sum(1 for _, h in pools if not h)
n_broad = sum(1 for _, h in pools if len(h) > args.max_pool)
print(f"\n{'='*104}")
print(f"{n_match}/{len(rows)} matched | {n_empty} had an empty pool | "
      f"{n_broad} pool(s) above --max-pool {args.max_pool}")
print(f"[write] {os.path.relpath(out, PROJECT_ROOT)}")
print(f"[write] {os.path.relpath(md, PROJECT_ROOT)}   <- readable version")
