"""Score merged reactions against the Reactome Neo4j ground truth for a gene.

Takes the output of run_merge.py and matches each extracted reaction to the
curated Reactome reaction describing the same event, reporting precision on the
extracted side and recall on the curated side.

Matching is structure-first (see reactome_llm/ReactionMatcher.py):

    normalize entities -> structural filter -> cosine shortlist
                       -> top-k -> cross-encoder rerank

Reactions that share no participants cannot be the same event, so structure
decides which curated reactions are candidates at all, and the text models only
rank what survives. An earlier version compared prose embeddings alone and
returned precision = recall = 1.0 for every input — including when scored
against an unrelated gene's reactions — because any two biomedical sentences
embed to ~0.85+ cosine, well above any usable threshold.

Scores in the output are unbounded cross-encoder logits, NOT cosine similarity.
--threshold defaults to a placeholder; calibrate it against unrelated genes
before quoting precision/recall.

Usage:
    python cosine_similarity_score.py results/pink1_2prev1next_merged.json --gene PINK1
    python cosine_similarity_score.py znfx1 --neo4j-db graph.db
    python cosine_similarity_score.py pink1 --gene-only        # don't widen ground truth
    python cosine_similarity_score.py pink1 --no-structural    # old all-pairs behaviour
    python cosine_similarity_score.py pink1 --top-k 2          # lean harder on cosine
"""
import os, sys, json, re, argparse

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

import numpy as np
import pandas as pd

CROSS_ENCODER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
# Matches TextEmbedder.SENTENCE_TRANSFORMER_MODEL so the project uses one bi-encoder.
BI_ENCODER_MODEL_NAME = 'all-MiniLM-L6-v2'
_CROSS_ENCODER = None


def _get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL_NAME):
    """Lazily construct and cache the cross-encoder (loading it is expensive; reuse across calls)."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        import torch
        import transformers
        from sentence_transformers import CrossEncoder
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            _CROSS_ENCODER = CrossEncoder(model_name)
        finally:
            transformers.logging.set_verbosity(prev_verbosity)
        # See GOTCHA in the module docstring — force identity to return raw logits, not Sigmoid.
        _CROSS_ENCODER.activation_fn = torch.nn.Identity()
    return _CROSS_ENCODER


def cross_encoder_rerank(query_text: str, abstracts: list) -> list:
    """Rank `abstracts` against a SINGLE `query_text`.

    Each dict must carry its abstract text under 'Summary'; the score is written back as
    'cross_score'. Returns the list sorted by descending relevance.
    """
    cross_encoder = _get_cross_encoder()
    pairs = [(query_text, a["Summary"]) for a in abstracts]
    scores = cross_encoder.predict(pairs)
    for a, s in zip(abstracts, scores):
        a["cross_score"] = float(s)
    return sorted(abstracts, key=lambda a: a["cross_score"], reverse=True)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Score merged reactions against Reactome Neo4j ground truth.')
parser.add_argument('target', nargs='?', default='pink1',
                    help='gene prefix (e.g. znfx1) or path to a *_merged.json file (default: pink1)')
parser.add_argument('--gene', help='gene symbol for the Neo4j lookup (default: inferred from filename)')
parser.add_argument('--out', help='output prefix (default: input with _merged -> _scores)')
parser.add_argument('--threshold', type=float, default=0.0,
                    help='cross-encoder LOGIT at or above which a pair counts as a match (default: 0.0). '
                         'Not a cosine value — logits are unbounded and this default is a placeholder; '
                         'calibrate it against unrelated genes before trusting precision/recall.')
parser.add_argument('--max-pathways', type=int, default=10,
                    help='max Neo4j pathways to pull reactions from, 0 = all (default: 10)')
parser.add_argument('--neo4j-uri', help='override REACTOME_NEO4J_URI')
parser.add_argument('--neo4j-db', help='override REACTOME_NEO4J_DATABASE')
parser.add_argument('--embed-model', default=CROSS_ENCODER_MODEL_NAME,
                    help=f'cross-encoder used for pairwise matching (default: {CROSS_ENCODER_MODEL_NAME})')
parser.add_argument('--bi-encoder-model', default=BI_ENCODER_MODEL_NAME,
                    help=f'bi-encoder for the cheap ranking stage that shortlists candidates '
                         f'for the cross-encoder (default: {BI_ENCODER_MODEL_NAME})')
parser.add_argument('--top-k', type=int, default=25,
                    help='candidates kept after cosine ranking and passed to the '
                         'cross-encoder (default: 25). Cosine only sorts the right '
                         'reaction into roughly the top ten, not to rank 1, so a '
                         'tight shortlist discards correct answers before the '
                         'cross-encoder can see them. Reranking is the cheap stage '
                         'here, so err wide.')
parser.add_argument('--min-shared', type=int, default=1,
                    help='entities an extracted and curated reaction must share to be '
                         'structurally compatible (default: 1)')
parser.add_argument('--require-catalyst', action='store_true',
                    help='only consider curated reactions with a matching catalyst')
parser.add_argument('--no-structural', action='store_true',
                    help='skip structural filtering and score every pair (the old behaviour)')
parser.add_argument('--gene-only', action='store_true',
                    help='restrict ground truth to --gene instead of widening it to every '
                         'curated gene the extraction mentions')
parser.add_argument('--rank-by-text', action='store_true',
                    help='rank candidates by cross-encoder logit alone instead of putting '
                         'structural evidence first (for comparison; measurably worse)')
parser.add_argument('--keep-deprecated', action='store_true',
                    help="keep Reactome 'Clone of ...' / 'replaced by ...' records")
args = parser.parse_args()


def resolve_paths(target, out=None):
    """Accept either a merged-file path or a gene prefix; return (IN, OUT_PREFIX)."""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    cand = target if os.path.isabs(target) else os.path.join(PROJECT_ROOT, target)
    if target.endswith('.json') or os.path.isfile(cand):
        in_path = cand
    else:
        in_path = os.path.join(results_dir, f'{target.lower()}_merged.json')
    if out:
        out_prefix = out if os.path.isabs(out) else os.path.join(PROJECT_ROOT, out)
    elif '_merged' in in_path:
        out_prefix = in_path.replace('_merged', '_scores').replace('.json', '')
    else:
        out_prefix = os.path.splitext(in_path)[0] + '_scores'
    return in_path, out_prefix


def _looks_like_id(s):
    """True for PubMed-style identifiers, which are never gene symbols."""
    return bool(re.fullmatch(r'(PMID:?\d+|PMC\d+|\d{4,9})', s, re.I))


def infer_gene(in_path, entries):
    """Gene symbol from --gene, else the 'source' field (PINK1.pdf), else the filename prefix.

    PubMed sources ('PMID:38234567') carry no gene, so they are skipped in favour
    of the filename prefix that `run_extraction.py --gene` writes.
    """
    if args.gene:
        return args.gene.upper()
    for e in entries:
        src = e.get('source') or ''
        if src and not _looks_like_id(src):
            return os.path.splitext(os.path.basename(src))[0].upper()
    # results/znfx1_merged.json -> ZNFX1
    stem = os.path.basename(in_path).split('_')[0]
    gene = re.sub(r'\.json$', '', stem).upper()
    if _looks_like_id(gene):
        sys.exit(f"[error] cannot infer a gene from {os.path.basename(in_path)} — "
                 f"pass --gene, or re-run extraction with --gene so the filename carries it")
    return gene


IN, OUT_PREFIX = resolve_paths(args.target, args.out)
if not os.path.isfile(IN):
    sys.exit(f"[error] merged file not found: {IN}")
merged = json.load(open(IN))
GENE = infer_gene(IN, merged)

print(f"[setup] input : {IN}", flush=True)
print(f"[setup] gene  : {GENE}", flush=True)
print(f"[setup] output: {OUT_PREFIX}.csv / .json", flush=True)
print(f"[setup] {len(merged)} merged reaction(s) to score", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Neo4j ground truth
# ──────────────────────────────────────────────────────────────────────────
REACTOME_LLM_PATH = os.path.join(PROJECT_ROOT, 'reactome_llm')
if REACTOME_LLM_PATH not in sys.path:
    sys.path.insert(0, REACTOME_LLM_PATH)
import ReactomeNeo4jUtils as neo4jutils

# module reads env at import time; re-apply here so CLI overrides take effect
neo4jutils.URI = args.neo4j_uri or os.getenv('REACTOME_NEO4J_URI')
neo4jutils.AUTH = (os.getenv('REACTOME_NEO4J_USER'), os.getenv('REACTOME_NEO4J_PWD'))
neo4jutils.DB = args.neo4j_db or os.getenv('REACTOME_NEO4J_DATABASE')
if not neo4jutils.URI:
    sys.exit('[error] no Neo4j URI — set REACTOME_NEO4J_URI in .env or pass --neo4j-uri')
print(f"[setup] neo4j : {neo4jutils.URI} db={neo4jutils.DB}", flush=True)


def get_ground_truth_reactions(gene):
    """One row per curated Neo4j reaction the gene participates in.

    Returns list of {'reaction', 'pathway', 'roles', 'text'} — 'text' is what
    gets embedded, phrased the way the notebook phrased reaction roles.
    """
    try:
        pathways = neo4jutils.query_pathways_for_gene(gene)
    except Exception as ex:
        # a dead server / bad auth raises deep inside the bolt driver; the traceback
        # is 40 frames of driver internals and says nothing actionable
        sys.exit(f"[error] Neo4j query failed ({type(ex).__name__}): {str(ex).splitlines()[0]}\n"
                 f"        uri={neo4jutils.URI} db={neo4jutils.DB}\n"
                 f"        start the Reactome database (or point --neo4j-uri at a running one) and re-run")
    if not pathways:
        return []
    print(f"[neo4j] {len(pathways)} pathway(s) for {gene}", flush=True)

    limit = args.max_pathways if args.max_pathways > 0 else len(pathways)
    if len(pathways) > limit:
        print(f"[neo4j] using the first {limit} pathway(s); "
              f"{len(pathways) - limit} skipped (raise with --max-pathways 0 for all)", flush=True)
    pathways = pathways[:limit]

    by_reaction = {}   # reaction displayName -> {pathways: set, roles: set}
    for p in pathways:
        try:
            roles_df = neo4jutils.query_reaction_roles_of_pathway(p['pathway'], [gene])
        except Exception as ex:
            print(f"    role query failed for {p['pathway']}: {type(ex).__name__}: {ex}", flush=True)
            continue
        if roles_df is None or roles_df.empty:
            continue
        for _, row in roles_df.iterrows():
            rec = by_reaction.setdefault(row['reaction'], {'pathways': set(), 'roles': set()})
            rec['pathways'].add(row['pathway'])
            rec['roles'].add(row['role'])

    gt = []
    for reaction, rec in sorted(by_reaction.items()):
        roles = sorted(rec['roles'])
        paths = sorted(rec['pathways'])
        gt.append({
            'reaction': reaction,
            'pathway': '; '.join(paths),
            'roles': ', '.join(roles),
            'text': (f"{reaction}. {gene} acts as "
                     f"{' and '.join(roles)} in this reaction, "
                     f"within pathway {paths[0]}."),
        })
    return gt


import ReactionMatcher as RM

# Participant-aware ground truth. get_ground_truth_reactions() above returns only
# displayNames, which is what forced the whole comparison to be prose-vs-prose;
# this pulls each reaction's inputs, outputs and catalysts as graph nodes.
# A gene's own reactions supply only its own naming, so 'ubiquitin' never reaches
# 'Ub' and the structural filter reports sh=0 on real matches. Seeding with
# Reactome's whole entity vocabulary fixes that; gene-level aliases added on top
# still win, since the global pass does not overwrite.
BASE_ALIASES = RM.fetch_global_aliases(neo4jutils)
ground_truth, ALIASES = RM.fetch_ground_truth(
    neo4jutils, GENE, drop_deprecated=not args.keep_deprecated,
    base_idx=BASE_ALIASES)

# Widen the candidate pool to every gene the extraction actually mentions. A
# PINK1 paper is half about Parkin; if Reactome's Parkin reactions are never
# candidates, a correct extraction is scored against the wrong target.
if ground_truth and not args.gene_only:
    tokens = set()
    for e in merged:
        a = e.get('annotation_result') or {}
        ents, _ = RM.extracted_entities(a, ALIASES)
        tokens |= ents
    found = RM.resolve_gene_symbols(neo4jutils, tokens)
    # Reactome stores case variants as distinct geneName entries (PRKN, Prkn),
    # which would fetch the same reactions two or three times over.
    by_upper = {}
    for s in sorted(found):
        by_upper.setdefault(s.upper(), s)
    extra = sorted(v for k, v in by_upper.items() if k != GENE.upper())
    if extra:
        print(f"[neo4j] extraction also mentions {len(extra)} curated gene(s): "
              f"{', '.join(extra[:12])}{' ...' if len(extra) > 12 else ''}", flush=True)
        ground_truth, ALIASES = RM.fetch_ground_truth_multi(
            neo4jutils, [GENE] + extra, drop_deprecated=not args.keep_deprecated,
            verbose=False, base_idx=BASE_ALIASES)

if not ground_truth:
    sys.exit(f"[error] no curated reactions found in Neo4j for {GENE} — "
             f"check the gene symbol (--gene) and the Neo4j connection")
print(f"[neo4j] {len(ground_truth)} curated reaction(s) for {GENE}, "
      f"{len(ALIASES)} entity alias(es) indexed", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Text for embedding
# ──────────────────────────────────────────────────────────────────────────
def extracted_to_text(a):
    """Flatten one merged annotation_result into a single embeddable string.
    """
    parts = []
    if a.get('name'):
        parts.append(a['name'])
        
    ins = ', '.join(a.get('input') or [])
    outs = ', '.join(a.get('output') or [])
    if ins:
        parts.append(f"Inputs: {ins}.")
    if outs:
        parts.append(f"Outputs: {outs}.")
    ca = a.get('catalystActivity') or {}
    if ca.get('catalyst'):
        parts.append(f"Catalyzed by {ca['catalyst']}"
                     + (f" ({ca['molecularFunction']})." if ca.get('molecularFunction') else "."))
    for r in (a.get('regulatedBy') or []):
        if r.get('regulator'):
            parts.append(f"{r.get('regulationType', 'regulated')} by {r['regulator']}.")
    if a.get('compartment'):
        parts.append(f"Compartment: {a['compartment']}.")
    summ = a.get('summation')
    for s in (summ if isinstance(summ, list) else [summ] if summ else []):
        if isinstance(s, dict) and s.get('text'):
            parts.append(s['text'])
        elif isinstance(s, str):
            parts.append(s)
    parts.extend(a.get('relationships') or [])
    return ' '.join(p for p in parts if p)


extracted = []
for e in merged:
    a = e.get('annotation_result') or {}
    ents, cats = RM.extracted_entities(a, ALIASES)
    extracted.append({
        'source': e.get('source', ''),
        'name': a.get('name', ''),
        'reactionType': a.get('reactionType', ''),
        'n_variants': len(a.get('merged_names') or [a.get('name', '')]),
        'llm_confidence': a.get('confidence'),
        # Compact text, shaped like the ground-truth text. The long blob from
        # extracted_to_text() adds generic domain vocabulary that raises
        # similarity without adding discrimination.
        'text': RM.extracted_reaction_text(a),
        'entities': ents,
        'catalysts': cats,
    })


# ──────────────────────────────────────────────────────────────────────────
# Score:  structural filter -> cosine rank -> top-k -> cross-encoder rerank
#
# Structure decides WHICH curated reactions are even candidates; the text models
# only rank what survives. Reactions that share no participants cannot be the
# same event no matter how alike they read.
# ──────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print(f"[score] loading bi-encoder {args.bi_encoder_model}...", flush=True)
bi_encoder = SentenceTransformer(args.bi_encoder_model)
emb_ext = bi_encoder.encode([x['text'] for x in extracted])
emb_gt = bi_encoder.encode([g['text'] for g in ground_truth])
cos = cosine_similarity(emb_ext, emb_gt)

print(f"[score] loading cross-encoder {args.embed_model} (cached after first run)...", flush=True)
_get_cross_encoder(args.embed_model)

# -inf marks "never considered", so it can be told apart from "scored badly".
sim = np.full((len(extracted), len(ground_truth)), -np.inf, dtype=float)
struct = np.zeros_like(sim)          # shared-entity count, for reporting
cat_hit = np.zeros_like(sim, dtype=bool)
n_filtered = n_reranked = 0

for i, x in enumerate(extracted):
    # 1. structural filter
    if args.no_structural:
        cands = [(j, 0, False) for j in range(len(ground_truth))]
    else:
        cands = RM.structural_candidates(
            x['entities'], x['catalysts'], ground_truth,
            min_shared=args.min_shared, require_catalyst=args.require_catalyst,
            query_gene=GENE)
    for j, n_shared, cm in cands:
        struct[i][j], cat_hit[i][j] = n_shared, cm
    if not cands:
        n_filtered += 1
        continue

    # 2. cosine rank over survivors -> 3. keep top-k
    # Shortlist on the same evidence the final ranking uses. Sorting by cosine
    # alone silently drops structurally strong candidates: only reranked pairs
    # get a finite score, so a candidate cosine cuts here can never win later,
    # however many participants it shares. That is what matched
    # 'PINK1 phosphorylates Parkin at Ser65' to 'PINK1 is autophosphorylated'
    # while the PRKN reaction sat outside the shortlist.
    idxs = [j for j, _, _ in cands]
    idxs.sort(key=lambda j: (struct[i][j], bool(cat_hit[i][j]), cos[i][j]),
              reverse=True)
    top = idxs[:args.top_k]

    # 4. cross-encoder rerank of the shortlist
    shortlist = [{'idx': j, 'Summary': ground_truth[j]['text']} for j in top]
    for c in cross_encoder_rerank(x['text'], shortlist):
        sim[i][c['idx']] = c['cross_score']
    n_reranked += len(top)

    if (i + 1) % 10 == 0 or i + 1 == len(extracted):
        print(f"  scored {i+1}/{len(extracted)} extracted reactions", flush=True)

kept = int(np.isfinite(sim).sum())
total = sim.size
print(f"[score] structural filter: {kept}/{total} pairs survived "
      f"({100*kept/total:.0f}%), {n_reranked} cross-encoded", flush=True)
if n_filtered:
    print(f"[score] {n_filtered} extracted reaction(s) had NO structurally compatible "
          f"curated reaction — reported as unmatched", flush=True)
print("[score] NOTE: scores are unbounded cross-encoder logits, not cosine "
      "similarities — --threshold is a placeholder and needs calibrating.", flush=True)

T = args.threshold

# precision side — each extracted reaction vs its best curated match.
# A row whose whole sim slice is -inf was filtered out structurally: it shares no
# participants with any curated reaction, so it has no best match at all. Reporting
# argmax there would name a reaction that was never even considered.
def _rank_key(i, j):
    """Ordering for candidate curated reactions.

    Structural evidence outranks text: a shared catalyst, then the number of
    shared participants, and only then the cross-encoder logit. Ranking on the
    logit alone re-creates the original failure — measured here, it put
    'PINK1 phosphorylates Parkin at Ser65' against 'PINK1 is autophosphorylated'
    even though only one curated reaction contains PRKN at all.
    Pass --rank-by-text to score the way the cross-encoder alone would.
    """
    if args.rank_by_text:
        return (sim[i][j],)
    # Shared participants outrank a catalyst match: catalyst-only agreement
    # (sh=0) just means "same enzyme acts here too", which was matching
    # 'Parkin translocates...' to 'PINK1 is autophosphorylated'.
    return (struct[i][j], bool(cat_hit[i][j]), sim[i][j])


per_extracted = []
for i, x in enumerate(extracted):
    scored = np.isfinite(sim[i])
    has_match = bool(scored.any())
    ranked = sorted((k for k in range(len(ground_truth)) if scored[k]),
                    key=lambda k: _rank_key(i, k), reverse=True)
    j = ranked[0] if has_match else None
    order = ranked[:3]
    per_extracted.append({
        'gene': GENE,
        'source': x['source'],
        'extracted_reaction': x['name'],
        'reaction_type': x['reactionType'],
        'n_variants_merged': x['n_variants'],
        'llm_confidence': x['llm_confidence'],
        'n_structural_candidates': int(scored.sum()),
        'best_neo4j_match': ground_truth[j]['reaction'] if has_match else '',
        'best_neo4j_pathway': ground_truth[j]['pathway'] if has_match else '',
        'best_neo4j_roles': ground_truth[j]['roles'] if has_match else '',
        'shared_entities': int(struct[i][j]) if has_match else 0,
        'catalyst_match': bool(cat_hit[i][j]) if has_match else False,
        'similarity': round(float(sim[i][j]), 4) if has_match else None,
        'matched': bool(has_match and sim[i][j] >= T),
        'top3_matches': '; '.join(f"{ground_truth[k]['reaction']} ({sim[i][k]:.3f})"
                                  for k in order),
    })

# recall side — each curated reaction vs its best extracted match
per_ground_truth = []
for j, g in enumerate(ground_truth):
    scored = np.isfinite(sim[:, j])
    has_match = bool(scored.any())
    i = max((k for k in range(len(extracted)) if scored[k]),
            key=lambda k: _rank_key(k, j), default=None) if has_match else None
    per_ground_truth.append({
        'gene': GENE,
        'neo4j_reaction': g['reaction'],
        'neo4j_pathway': g['pathway'],
        'neo4j_roles': g['roles'],
        'best_extracted_match': extracted[i]['name'] if has_match else '',
        'shared_entities': int(struct[i][j]) if has_match else 0,
        'catalyst_match': bool(cat_hit[i][j]) if has_match else False,
        'similarity': round(float(sim[:, j][i]), 4) if has_match else None,
        'recovered': bool(has_match and sim[:, j][i] >= T),
    })

df_ext = pd.DataFrame(per_extracted)
df_gt = pd.DataFrame(per_ground_truth)

n_matched = int(df_ext['matched'].sum())
n_recovered = int(df_gt['recovered'].sum())
precision = n_matched / len(df_ext) if len(df_ext) else 0.0
recall = n_recovered / len(df_gt) if len(df_gt) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

summary = {
    'gene': GENE,
    'input_file': IN,
    'embed_model': args.embed_model,
    'threshold': T,
    'n_extracted': len(df_ext),
    'n_neo4j_reactions': len(df_gt),
    'n_extracted_matched': n_matched,
    'n_neo4j_recovered': n_recovered,
    'precision_at_threshold': round(precision, 4),
    'recall_at_threshold': round(recall, 4),
    'f1_at_threshold': round(f1, 4),
    'mean_best_similarity_extracted': round(float(df_ext['similarity'].mean()), 4),
    'mean_best_similarity_neo4j': round(float(df_gt['similarity'].mean()), 4),
}


# ──────────────────────────────────────────────────────────────────────────
# Report & save
# ──────────────────────────────────────────────────────────────────────────
def _fmt(v):
    """Score column — blank when a reaction had no structural candidate at all."""
    return f"{v:7.3f}" if v is not None else "      -"


print(f"\n=== extracted -> best Neo4j match ({GENE}) ===", flush=True)
for r in per_extracted:
    flag = 'MATCH' if r['matched'] else ('  -  ' if r['similarity'] is None else '     ')
    cat = 'cat' if r['catalyst_match'] else '   '
    print(f"  {flag} {_fmt(r['similarity'])} {cat} sh={r['shared_entities']}  "
          f"{r['extracted_reaction'][:56]:<56} -> "
          f"{r['best_neo4j_match'][:52] or '(no structural candidate)'}", flush=True)

print(f"\n=== Neo4j curated -> best extracted match ({GENE}) ===", flush=True)
for r in per_ground_truth:
    flag = 'FOUND' if r['recovered'] else ('  -  ' if r['similarity'] is None else '     ')
    cat = 'cat' if r['catalyst_match'] else '   '
    print(f"  {flag} {_fmt(r['similarity'])} {cat} sh={r['shared_entities']}  "
          f"{r['neo4j_reaction'][:56]:<56} -> "
          f"{r['best_extracted_match'][:52] or '(not extracted)'}", flush=True)

print(f"\n=== summary (threshold {T}) ===", flush=True)
for k, v in summary.items():
    print(f"  {k}: {v}", flush=True)

os.makedirs(os.path.dirname(OUT_PREFIX), exist_ok=True)
df_ext.to_csv(f'{OUT_PREFIX}.csv', index=False)
df_gt.to_csv(f'{OUT_PREFIX}_recall.csv', index=False)
with open(f'{OUT_PREFIX}.json', 'w') as f:
    json.dump({'summary': summary,
               'per_extracted': per_extracted,
               'per_ground_truth': per_ground_truth,
               'similarity_matrix': sim.round(4).tolist(),
               'extracted_reactions': [x['name'] for x in extracted],
               'neo4j_reactions': [g['reaction'] for g in ground_truth]}, f, indent=2)

print(f"\n[done] saved -> {OUT_PREFIX}.csv", flush=True)
print(f"[done] saved -> {OUT_PREFIX}_recall.csv", flush=True)
print(f"[done] saved -> {OUT_PREFIX}.json", flush=True)
