"""Is the embedding itself useful? Measured alone, with nothing downstream.

    merged reaction -> normalized text -> embedding -> cosine vs ALL Reactome
                                                             -> ranked candidates

No structural filter, no cross-encoder, no LLM. Two outputs.

TABLE 1 — retrieval
    extracted | correct Reactome | rank | cosine | top wrong score
    If the correct reaction ranks 2nd at 0.769 while a wrong one takes 1st at
    0.810, the embedding is not merely imprecise, it is confidently wrong.

TABLE 2 — why, by slot
    A reaction is (action, catalyst, substrate). Cosine returns one number for
    all three at once, so it cannot say WHICH slot disagreed:

        PINK1 phosphorylates ubiquitin   vs   PINK1 phosphorylates PRKN
          action    phosphorylation  OK         action    phosphorylation  OK
          catalyst  PINK1            OK         catalyst  PINK1            OK
          substrate ubiquitin        OK         substrate PRKN             WRONG

    Two of three slots agree, so cosine scores it high. But reaction identity
    needs all three, and the substrate mismatch has to be disqualifying rather
    than a third of a penalty. Table 2 counts how often a high-scoring wrong
    hit is exactly this shape.

'Correct' is not hand-labelled. A Reactome reaction is the same reaction when
all three slots agree, which is a rule rather than an opinion. Pass --gold to
override it with curator labels.

Usage:
    python embedding_benchmark.py
    python embedding_benchmark.py --merged results/pmc4003245_..._merged.json
    python embedding_benchmark.py --bi-encoder-model NeuML/pubmedbert-base-embeddings
"""
import os, re, sys, csv, json, time, argparse
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import numpy as np

ap = argparse.ArgumentParser(description='Clean embedding-only benchmark.')
ap.add_argument('--merged', default='results/pink1_2prev1next_merged.json')
ap.add_argument('--gene', default='PINK1', help='gene whose alias index normalizes entities')
ap.add_argument('--bi-encoder-model', default='all-MiniLM-L6-v2')
ap.add_argument('--gold', help='JSON of {extracted name: Reactome stId or null} overriding '
                               'the slot rule')
ap.add_argument('--top-k', type=int, default=10, help='rows per query in the CSV (default: 10)')
ap.add_argument('--neo4j-db', default='graph.db')
args = ap.parse_args()

import ReactomeNeo4jUtils as neo4jutils
neo4jutils.URI = os.getenv('REACTOME_NEO4J_URI')
neo4jutils.AUTH = (os.getenv('REACTOME_NEO4J_USER'), os.getenv('REACTOME_NEO4J_PWD'))
neo4jutils.DB = args.neo4j_db
import ReactionMatcher as RM

stem = re.sub(r'_(merged|extraction)\.json$', '', os.path.basename(args.merged))

# ── slots ───────────────────────────────────────────────────────────────────
# The chemistry, keyed off the verb. Reactome's reactionType is a schema class
# ('transition', 'blackBoxEvent') and says nothing about what happens, so both
# sides are read the same way: from the reaction name.
_ACTIONS = [
    ('phosphorylation', r'phosphorylat|autophosphorylat|phospho-|kinase'),
    ('dephosphorylation', r'dephosphorylat|phosphatase'),
    ('ubiquitination', r'ubiquitinat|ubiquitylat|polyubiquitinat|ubiquitinates|transfers ub'),
    ('deubiquitination', r'deubiquitinat|deubiquitylat'),
    ('binding', r'\bbind|\bbinds\b|interact|associat|forms? a complex'),
    ('dissociation', r'dissociat|releases?\b|unbind'),
    ('translocation', r'translocat|recruit|import|export|is recruited|relocali'),
    ('cleavage', r'cleav|proteolys|degrad|hydroly'),
    ('acetylation', r'acetylat'),
    ('methylation', r'methylat'),
    ('sumoylation', r'sumoylat'),
    ('expression', r'gene expression|transcription|is expressed'),
]


def action_of(text):
    """The chemistry a reaction name describes, or None if no verb is recognised."""
    t = (text or '').lower()
    for label, pat in _ACTIONS:
        if re.search(pat, t):
            return label
    return None


# Cofactors and generic nouns are in almost every reaction, so they identify
# nothing. RM already keeps this list for structural matching.
_GENERIC = RM._STOPWORDS | {'ubiquitin chain', 'mitochondrial proteins', 'substrate'}


def _norm(names, idx):
    """Canonical entities in a participant list, modifications included when the
    modifier is itself a real entity.

    'Ub-MOM proteins' splits to base {mom proteins} + modification {ub}. For a
    reaction that phosphorylates the ubiquitin, dropping the modifier loses the
    actual substrate, so a modifier the alias index recognises is kept as a
    participant. A bare 'p-S65' is not an entity and stays out.
    """
    out = set()
    for n in (names or []):
        out |= {e for e in RM.resolve_entity(n, idx) if e and e not in _GENERIC}
        mods = RM.split_entity(n, idx)[1]
        out |= {idx.canon(m) for m in mods
                if idx.knows(m) and idx.canon(m) not in _GENERIC}
    return out


def _substrate(inputs, cat, idx):
    """Inputs minus the catalyst. An autocatalytic reaction has nothing left —
    PINK1 autophosphorylation lists only PINK1 and ATP — and there the catalyst
    IS the substrate, so returning an empty set would wrongly read as 'unknown'.
    """
    return (_norm(inputs, idx) - cat) or cat


def extracted_slots(a, idx):
    ca = a.get('catalystActivity') or {}
    cat = _norm([ca.get('catalyst')] if ca.get('catalyst') else [], idx)
    return {'action': action_of(a.get('name')),
            'catalyst': cat,
            'substrate': _substrate(a.get('input'), cat, idx)}


def corpus_slots(c, idx):
    cat = _norm(c['catalysts'], idx)
    return {'action': action_of(c['reaction']),
            'catalyst': cat,
            'substrate': _substrate(c['inputs'], cat, idx)}


def compare(qs, cs):
    """Per-slot verdicts. A slot with nothing on either side is not evidence."""
    v = {}
    v['action'] = None if not (qs['action'] and cs['action']) else qs['action'] == cs['action']
    for k in ('catalyst', 'substrate'):
        v[k] = None if not (qs[k] and cs[k]) else bool(qs[k] & cs[k])
    return v


def is_same_reaction(v):
    """Same reaction only on positive evidence for the chemistry AND the molecule.

    'No slot contradicts' is far too weak — under it, 'Parkin translocates to
    damaged mitochondria' matches 'PTEN translocates to mitochondrial outer
    membrane' on the shared verb alone. Action and substrate must both be
    positively confirmed; the catalyst may be unknown (Reactome often records no
    catalyst for a binding or transport step) but must not conflict.
    """
    return (v['action'] is True and v['substrate'] is True
            and v['catalyst'] is not False)


# ── corpus ──────────────────────────────────────────────────────────────────
CORPUS_QUERY = """
MATCH (r:ReactionLikeEvent)-[:species]->(:Species {displayName:'Homo sapiens'})
OPTIONAL MATCH (r)-[:input]->(i:PhysicalEntity)
OPTIONAL MATCH (r)-[:output]->(o:PhysicalEntity)
OPTIONAL MATCH (r)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(c:PhysicalEntity)
RETURN r.dbId AS db_id, r.stId AS st_id, r.displayName AS reaction,
       collect(DISTINCT i.displayName)[..6] AS inputs,
       collect(DISTINCT o.displayName)[..6] AS outputs,
       collect(DISTINCT c.displayName)[..3] AS catalysts
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

# Reactome's global synonym vocabulary first, then the gene's own naming on top
# so a gene-specific form wins any conflict.
idx = RM.fetch_global_aliases(neo4jutils)
_, gene_idx = RM.fetch_ground_truth(neo4jutils, args.gene, verbose=False)
for surface, canon in gene_idx.items():
    idx.add_group(canon, [surface])
print(f"[alias] {len(idx):,} surface forms total "
      f"(+{len(gene_idx)} from {args.gene})", flush=True)

# ── embed and rank ──────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(args.bi_encoder_model)
cache = os.path.join(PROJECT_ROOT, 'data', 'cache',
                     f"corpus_{args.bi_encoder_model.replace('/', '_')}_{len(corpus)}.npy")
os.makedirs(os.path.dirname(cache), exist_ok=True)
if os.path.exists(cache):
    emb_corpus = np.load(cache)
    print(f"[embed] reused {os.path.relpath(cache, PROJECT_ROOT)}", flush=True)
else:
    print(f"[embed] {args.bi_encoder_model} over {len(corpus):,} reactions...", flush=True)
    t0 = time.time()
    emb_corpus = model.encode([c['text'] for c in corpus], batch_size=256)
    np.save(cache, emb_corpus)
    print(f"[embed] {time.time()-t0:.0f}s -> cached", flush=True)

merged = json.load(open(os.path.join(PROJECT_ROOT, args.merged)))
queries = [e.get('annotation_result') or {} for e in merged]
emb_q = model.encode([RM.extracted_reaction_text(a) for a in queries])
sim = cosine_similarity(emb_q, emb_corpus)
order = np.argsort(-sim, axis=1)

c_slots = [corpus_slots(c, idx) for c in corpus]
gold = json.load(open(os.path.join(PROJECT_ROOT, args.gold))) if args.gold else None

# ── TABLE 1 — retrieval ─────────────────────────────────────────────────────
print(f"\n{'='*118}\nTABLE 1 — RETRIEVAL   (embedding: {args.bi_encoder_model})\n{'='*118}")
print(f"{'EXTRACTED REACTION':<40} {'CORRECT REACTOME':<38} {'RANK':>5} {'COSINE':>7} "
      f"{'TOPWRONG':>9} {'DELTA':>7}")
print('-' * 118)

rows, ranks, n_conf_wrong, cases = [], [], 0, []
for qi, a in enumerate(queries):
    qs = extracted_slots(a, idx)
    if gold is not None and a.get('name') in gold:
        stid = gold[a['name']]
        correct = [j for j, c in enumerate(corpus) if stid and c['st_id'] == stid]
    else:
        correct = [j for j in range(len(corpus)) if is_same_reaction(compare(qs, c_slots[j]))]

    ranked = list(order[qi])
    best_rank = next((r for r, j in enumerate(ranked) if j in set(correct)), None)
    top_wrong = next(sim[qi][j] for j in ranked if j not in set(correct))
    if correct:
        cj = min(correct, key=lambda j: ranked.index(j))
        name, cos_c = corpus[cj]['reaction'], sim[qi][cj]
        rk = f"{best_rank+1}" if best_rank is not None else '>corpus'
        delta = cos_c - top_wrong
        ranks.append(best_rank + 1)
        if delta < 0:
            n_conf_wrong += 1
            cases.append((a, qs, corpus[ranked[0]], c_slots[ranked[0]],
                          corpus[cj], sim[qi][ranked[0]], cos_c))
    else:
        name, cos_c, rk, delta = '— not in Reactome —', float('nan'), '—', float('nan')
    print(f"{a.get('name','')[:38]:<40} {name[:36]:<38} {rk:>5} "
          f"{cos_c:>7.4f} {top_wrong:>9.4f} {delta:>+7.4f}"
          if correct else
          f"{a.get('name','')[:38]:<40} {name[:36]:<38} {rk:>5} {'—':>7} "
          f"{top_wrong:>9.4f} {'—':>7}")
    rows.append({'extracted_reaction': a.get('name', ''), 'correct_reactome': name,
                 'correct_stid': corpus[cj]['st_id'] if correct else '',
                 'rank': rk, 'cosine': f'{cos_c:.4f}' if correct else '',
                 'top_wrong_score': f'{top_wrong:.4f}',
                 'delta': f'{delta:+.4f}' if correct else '',
                 'top1_reactome': corpus[ranked[0]]['reaction']})

out_csv = os.path.join(PROJECT_ROOT, 'results', f'{stem}_embedding_benchmark.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('-' * 118)
n_gold = len(ranks)
if n_gold:
    r = np.array(ranks)
    print(f"queries whose reaction Reactome actually has : {n_gold}/{len(queries)}")
    for k in (1, 5, 10, 25):
        print(f"  Recall@{k:<3} {int((r <= k).sum())}/{n_gold}  ({100*(r <= k).mean():.0f}%)")
    print(f"  median rank {int(np.median(r))}, worst {r.max()}")
    print(f"  CONFIDENTLY WRONG (a wrong reaction outranks the correct one): "
          f"{n_conf_wrong}/{n_gold}")
print(f"[write] {os.path.relpath(out_csv, PROJECT_ROOT)}")

# ── TABLE 2 — why, by slot ──────────────────────────────────────────────────
print(f"\n{'='*118}\nTABLE 2 — WHY: the winning wrong answer, slot by slot\n{'='*118}")
tick = {True: 'OK   ', False: 'WRONG', None: '?    '}
for a, qs, wrong, ws, right, cos_w, cos_r in cases[:8]:
    v = compare(qs, ws)
    print(f"\nQUERY  {a.get('name','')}")
    print(f"  cosine put   {wrong['reaction'][:60]:<62} {cos_w:.4f}")
    print(f"  above        {right['reaction'][:60]:<62} {cos_r:.4f}")
    for slot in ('action', 'catalyst', 'substrate'):
        q = qs[slot] if slot == 'action' else ', '.join(sorted(qs[slot])) or '-'
        w_ = ws[slot] if slot == 'action' else ', '.join(sorted(ws[slot])) or '-'
        print(f"     {slot:<10} {tick[v[slot]]}  extracted={str(q)[:30]:<32} wrong-hit={str(w_)[:30]}")

if cases:
    n_sub = sum(1 for *_x, in [()] ) if False else sum(
        1 for a, qs, w, ws, r_, cw, cr in cases
        if compare(qs, ws)['substrate'] is False
        and compare(qs, ws)['action'] is not False)
    print(f"\n{'-'*118}")
    print(f"Of the {len(cases)} confidently-wrong hits, {n_sub} share the extracted")
    print(f"reaction's ACTION but not its SUBSTRATE — right chemistry, wrong molecule.")
    print("Cosine averages the slots, so two agreements outweigh one disagreement.")
    print("Reaction identity is a conjunction: the substrate mismatch alone should")
    print("disqualify, which is a rule cosine has no way to express.")
