"""The pure experiment: can an embedding identify the same biochemical reaction?

Everything downstream of the embedding is deliberately removed. No structural
filter, no cross-encoder, no candidate pool:

    merged reaction -> text -> embedding -> cosine vs ALL Reactome reactions -> rank

and three questions are asked of that ranking.

  1. RETRIEVAL   Is the correct Reactome reaction in the top 1 / 5 / 10?
  2. SEPARATION  Is the correct reaction's cosine substantially ABOVE
                 biologically unrelated reactions, or is everything ~0.85?
  3. THRESHOLD   Is there a cosine above which two reactions are reliably the
                 same reaction?

(1) and (3) need gold labels, so this runs in two passes:

    # pass 1 - rank, and emit a labelling sheet
    python cosine_retrieval_test.py
    #   -> results/<stem>_cosine_ranking.csv    every scored pair, top-N
    #   -> results/<stem>_cosine_labels.csv     same rows, blank `same` column

    # fill in the `same` column (y / n), then
    # pass 2 - scored against those labels
    python cosine_retrieval_test.py --labels results/<stem>_cosine_labels.csv

SEPARATION needs no labels and is always reported: it compares each query's top
hit against the corpus-wide score distribution for that same query. A top hit
that sits only fractionally above the median of 20,000 unrelated reactions is
the 0.85-vs-collagen problem, restated quantitatively.
"""
import os, re, sys, csv, json, time, argparse
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import numpy as np

ap = argparse.ArgumentParser(description='Pure cosine retrieval over all of Reactome.')
ap.add_argument('--merged', default='results/pink1_2prev1next_merged.json',
                help='reactions to query with — a merged file or a raw extraction')
ap.add_argument('--top-k', type=int, default=10,
                help='rows per query written to the ranking/labelling CSVs (default: 10)')
ap.add_argument('--labels', help='filled-in labelling CSV; enables RETRIEVAL and THRESHOLD')
ap.add_argument('--bi-encoder-model', default='all-MiniLM-L6-v2')
ap.add_argument('--neo4j-db', default='graph.db')
ap.add_argument('--out-prefix', help='output stem (default: derived from --merged)')
args = ap.parse_args()

import ReactomeNeo4jUtils as neo4jutils
neo4jutils.URI = os.getenv('REACTOME_NEO4J_URI')
neo4jutils.AUTH = (os.getenv('REACTOME_NEO4J_USER'), os.getenv('REACTOME_NEO4J_PWD'))
neo4jutils.DB = args.neo4j_db
import ReactionMatcher as RM

stem = args.out_prefix or re.sub(r'_(merged|extraction)\.json$', '',
                                 os.path.basename(args.merged))
RANK_CSV = os.path.join(PROJECT_ROOT, 'results', f'{stem}_cosine_ranking.csv')
LABEL_CSV = os.path.join(PROJECT_ROOT, 'results', f'{stem}_cosine_labels.csv')

# ── the whole corpus ────────────────────────────────────────────────────────
# Deliberately a light query: displayName plus direct participants, no complex
# expansion. This only builds embedding text, and 20k deep traversals would cost
# minutes for no gain.
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
print(f"[corpus] {len(corpus):,} human reactions in {time.time()-t0:.1f}s "
      f"(deprecated dropped)", flush=True)

for c in corpus:
    c['text'] = RM._gt_text(c)

# ── embed both sides and rank ───────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(args.bi_encoder_model)

# The corpus is the same for every paper, so embedding it once and reusing turns
# a 2-minute startup into a 1-second one. Keyed on model + corpus size so a model
# swap or a database change misses the cache rather than silently reusing it.
cache = os.path.join(PROJECT_ROOT, 'data', 'cache',
                     f"corpus_{args.bi_encoder_model.replace('/', '_')}_{len(corpus)}.npy")
os.makedirs(os.path.dirname(cache), exist_ok=True)
if os.path.exists(cache):
    emb_corpus = np.load(cache)
    print(f"[embed] reused {os.path.relpath(cache, PROJECT_ROOT)}", flush=True)
else:
    print(f"[embed] {args.bi_encoder_model} over {len(corpus):,} reactions...", flush=True)
    t0 = time.time()
    emb_corpus = model.encode([c['text'] for c in corpus], batch_size=256,
                              show_progress_bar=False)
    np.save(cache, emb_corpus)
    print(f"[embed] done in {time.time()-t0:.0f}s -> cached", flush=True)

merged = json.load(open(os.path.join(PROJECT_ROOT, args.merged)))
queries = [e.get('annotation_result') or {} for e in merged]
q_text = [RM.extracted_reaction_text(a) for a in queries]
emb_q = model.encode(q_text)

sim = cosine_similarity(emb_q, emb_corpus)          # (n_query, n_corpus)
order = np.argsort(-sim, axis=1)
K = args.top_k

# ── the table ───────────────────────────────────────────────────────────────
os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
COLS = ['query_idx', 'extracted_reaction', 'rank', 'cosine', 'reactome_reaction',
        'reactome_stid']
with open(RANK_CSV, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(COLS)
    for qi, a in enumerate(queries):
        for r, j in enumerate(order[qi][:K]):
            w.writerow([qi, a.get('name', ''), r + 1, f'{sim[qi][j]:.4f}',
                        corpus[j]['reaction'], corpus[j]['st_id']])
print(f"[write] {RANK_CSV}")

if not args.labels and not os.path.exists(LABEL_CSV):
    with open(LABEL_CSV, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLS + ['same'])
        for qi, a in enumerate(queries):
            for r, j in enumerate(order[qi][:K]):
                w.writerow([qi, a.get('name', ''), r + 1, f'{sim[qi][j]:.4f}',
                            corpus[j]['reaction'], corpus[j]['st_id'], ''])
    print(f"[write] {LABEL_CSV}  <- fill the `same` column with y/n")

# ── 2. SEPARATION (needs no labels) ─────────────────────────────────────────
# For each query, where does its BEST hit sit relative to the 20k reactions it
# is definitely not? If top-1 is only a fraction of a standard deviation above
# the corpus mean, cosine is not discriminating, it is just measuring "both of
# these are sentences about biochemistry".
print(f"\n{'='*104}\n2. SEPARATION — top hit vs the corpus it is not\n{'='*104}")
print(f"{'extracted reaction':<52} {'top1':>7} {'p50':>7} {'p99':>7} "
      f"{'gap1-10':>8} {'z(top1)':>8}")
print('-' * 104)
z_all, gap_all = [], []
for qi, a in enumerate(queries):
    row = sim[qi]
    top1, p50, p99 = row[order[qi][0]], np.median(row), np.percentile(row, 99)
    gap = top1 - row[order[qi][9]] if len(order[qi]) > 9 else float('nan')
    z = (top1 - row.mean()) / (row.std() or 1e-9)
    z_all.append(z); gap_all.append(gap)
    print(f"{a.get('name','')[:50]:<52} {top1:>7.4f} {p50:>7.4f} {p99:>7.4f} "
          f"{gap:>8.4f} {z:>8.2f}")
print('-' * 104)
print(f"{'MEAN':<52} {np.mean([sim[i][order[i][0]] for i in range(len(queries))]):>7.4f} "
      f"{np.median(sim):>7.4f} {np.percentile(sim, 99):>7.4f} "
      f"{np.nanmean(gap_all):>8.4f} {np.mean(z_all):>8.2f}")
print("\nz(top1) is how many standard deviations the best hit sits above the mean")
print("of all 20k. Above ~6 the top hit is genuinely distinctive; near 3-4 the")
print("ranking is weak and a fixed cosine threshold cannot work.")

# ── 1 & 3. RETRIEVAL and THRESHOLD (need labels) ────────────────────────────
if not args.labels:
    print(f"\n{'='*104}\n1. RETRIEVAL and 3. THRESHOLD — skipped, no labels\n{'='*104}")
    print(f"Mark the `same` column in {os.path.relpath(LABEL_CSV, PROJECT_ROOT)}")
    print(f"then re-run with --labels {os.path.relpath(LABEL_CSV, PROJECT_ROOT)}")
    sys.exit(0)

gold = {}                                   # query_idx -> {stid marked same}
with open(os.path.join(PROJECT_ROOT, args.labels)) as f:
    for row in csv.DictReader(f):
        if (row.get('same') or '').strip().lower() in ('y', 'yes', '1', 'true'):
            gold.setdefault(int(row['query_idx']), set()).add(row['reactome_stid'])

print(f"\n{'='*104}\n1. RETRIEVAL — is the correct reaction in the top k?\n{'='*104}")
hits = {1: 0, 5: 0, 10: 0}
labelled = [qi for qi in range(len(queries)) if gold.get(qi)]
for qi in labelled:
    ranked = [corpus[j]['st_id'] for j in order[qi][:max(hits)]]
    best = next((r + 1 for r, s in enumerate(ranked) if s in gold[qi]), None)
    for k in hits:
        hits[k] += bool(best and best <= k)
    print(f"  rank {str(best or '>%d' % max(hits)):>4}   {queries[qi].get('name','')[:82]}")
n = len(labelled) or 1
print(f"\n  labelled queries with a correct match: {len(labelled)}")
for k in sorted(hits):
    print(f"  Recall@{k:<3} {hits[k]}/{len(labelled)}  ({100*hits[k]/n:.0f}%)")
if not labelled:
    print("  (no rows marked y — nothing to score)")

print(f"\n{'='*104}\n3. THRESHOLD — does one cosine cutoff separate same from different?\n{'='*104}")
pos, neg = [], []
with open(os.path.join(PROJECT_ROOT, args.labels)) as f:
    for row in csv.DictReader(f):
        lab = (row.get('same') or '').strip().lower()
        if lab in ('y', 'yes', '1', 'true'):
            pos.append(float(row['cosine']))
        elif lab in ('n', 'no', '0', 'false'):
            neg.append(float(row['cosine']))
if not pos or not neg:
    print("  need both y and n labels to find a cutoff")
else:
    print(f"  same      n={len(pos):<4} cosine {min(pos):.4f} - {max(pos):.4f}  "
          f"median {np.median(pos):.4f}")
    print(f"  different n={len(neg):<4} cosine {min(neg):.4f} - {max(neg):.4f}  "
          f"median {np.median(neg):.4f}")
    best = max(
        ((t, 2 * sum(p >= t for p in pos) /
             (2 * sum(p >= t for p in pos) + sum(q >= t for q in neg)
              + sum(p < t for p in pos) or 1))
         for t in np.arange(0.30, 1.00, 0.005)), key=lambda x: x[1])
    t, f1 = best
    tp, fp, fn = sum(p >= t for p in pos), sum(q >= t for q in neg), sum(p < t for p in pos)
    print(f"\n  best cutoff {t:.3f} -> F1 {f1:.2f}  (tp {tp}, fp {fp}, fn {fn})")
    print(f"  overlap: {sum(q >= min(pos) for q in neg)} different pairs score at or above")
    print(f"  the LOWEST same pair — that overlap is the ceiling on any threshold.")
