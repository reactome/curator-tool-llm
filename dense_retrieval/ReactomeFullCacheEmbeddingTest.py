"""Bi-encoder full-pool positive-control retrieval eval (NO cross-encoder).

For each of 10 seeded has-data genes:
  1. pull the gene's full reaction list from Neo4j;
  2. deterministically pick ONE reaction  (random.Random(f"{SEED}|{gene}|reaction_pick").choice)
     and reuse it across all 3 draws;
  3. true positives = that reaction's cited PMIDs that have an abstract in Mongo (X papers);
  4. query = the reaction's DB summary text -- if none, SKIP the gene (no stochastic LLM
     fallback) and take the next gene in the seeded order.

Negative universe (built once): every distinct PMID cited by ANY reaction in the graph,
intersected with Mongo-available abstracts. Per (gene, draw) we sample (1000 - X) negatives
from it (seeded f"{SEED}|{gene}|{draw}", excluding the gene's own X TPs) -> a 1000-paper pool.
The reaction summary and every pool abstract are embedded with the all-MiniLM-L6-v2 bi-encoder
(abstract vectors cached globally by PMID), ranked by cosine similarity, and scored:
  hit_rate      = (# TPs ranked in the top X) / X
  correct_ranks = 1-indexed rank (1..1000) of each individual TP in the full sorted pool.

Output: one CSV row per (gene, draw) -> data/fullpool_eval_<date>.csv
  gene, reaction, reaction_stid, n_true_positives, pool_size, draw, draw_seed,
  hit_rate, correct_ranks (JSON list)

Usage (repo root, paperqa env):
    python reactome_llm/ReactomeFullCacheEmbeddingTest.py            # full 10-gene x 3-draw run
    python reactome_llm/ReactomeFullCacheEmbeddingTest.py --benchmark  # universe count + embed-rate only
"""
import os, sys, csv, json, time, random, argparse, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import neo4j
from neo4j import GraphDatabase
from pymongo import MongoClient

from ReactomeNeo4jUtils import URI, AUTH, DB
import ReactomePubMed as rpm
from TextEmbedder import create_sentence_transformer

BASE_SEED = 20260805
N_GENES = 10
N_DRAWS = 3
POOL_SIZE = 1000
DATE = datetime.date.today().isoformat()
OUT = os.path.join("data", f"fullpool_eval_{DATE}.csv")
FIELDS = ["gene", "reaction", "reaction_stid", "n_true_positives", "pool_size",
          "draw", "draw_seed", "hit_rate", "correct_ranks"]


# ------------------------------------------------------------------ Neo4j / Mongo helpers

def _driver():
    return GraphDatabase.driver(URI, auth=AUTH)


def _mongo():
    cli = MongoClient(rpm.pubmed_mongo_uri)
    return cli[rpm.pubmed_mongo_db][rpm.pubmed_mongo_collection]


def reactome_wide_pmids() -> list:
    """Every distinct PMID cited by ANY reaction in the graph (strings)."""
    q = """
        MATCH (:ReactionLikeEvent)-[:literatureReference]->(lit:LiteratureReference)
        WHERE lit.pubMedIdentifier IS NOT NULL
        RETURN DISTINCT lit.pubMedIdentifier AS pmid
    """
    with _driver() as d:
        df = d.execute_query(q, db=DB, result_transformer_=neo4j.Result.to_df)
    return [str(p) for p in df["pmid"].to_list()] if not df.empty else []


def mongo_available(pmids: list, col) -> set:
    """Subset of `pmids` that have an abstract document in Mongo (bulk $in, chunked)."""
    avail, pmids = set(), list(pmids)
    for i in range(0, len(pmids), 50000):
        chunk = pmids[i:i + 50000]
        for doc in col.find({"pmid": {"$in": chunk}}, {"pmid": 1, "_id": 0}):
            avail.add(str(doc["pmid"]))
    return avail


def all_annotated_genes() -> list:
    """All human gene symbols participating in at least one reaction (deterministic order)."""
    q = """
        MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
        WHERE g.geneName IS NOT NULL AND ewas.speciesName = "Homo sapiens"
        MATCH (r:ReactionLikeEvent)
              -[:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*]->(ewas)
        RETURN DISTINCT g.geneName[0] AS gene
    """
    with _driver() as d:
        df = d.execute_query(q, db=DB, result_transformer_=neo4j.Result.to_df)
    return sorted(df["gene"].to_list()) if not df.empty else []


def reactions_for_gene(gene: str) -> list:
    """All (stid, name) reactions the gene participates in, sorted by stId (deterministic)."""
    q = """
        MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
        WHERE g.geneName[0] = $gene
        MATCH (r:ReactionLikeEvent)
              -[:input|catalystActivity|regulatedBy|physicalEntity|hasComponent|hasMember|hasCandidate*]->(ewas)
        RETURN DISTINCT r.stId AS stid, r.displayName AS name
    """
    with _driver() as d:
        df = d.execute_query(q, db=DB, gene=gene, result_transformer_=neo4j.Result.to_df)
    if df.empty:
        return []
    return sorted(((str(r["stid"]), r["name"]) for _, r in df.iterrows()), key=lambda t: t[0])


def reaction_pmids(stid: str) -> list:
    q = """
        MATCH (r:ReactionLikeEvent {stId: $stid})-[:literatureReference]->(lit:LiteratureReference)
        WHERE lit.pubMedIdentifier IS NOT NULL
        RETURN DISTINCT lit.pubMedIdentifier AS pmid
    """
    with _driver() as d:
        df = d.execute_query(q, db=DB, stid=stid, result_transformer_=neo4j.Result.to_df)
    return [str(p) for p in df["pmid"].to_list()] if not df.empty else []


def reaction_summary(stid: str):
    q = """
        MATCH (r:ReactionLikeEvent {stId: $stid})
        OPTIONAL MATCH (r)-[:summation]->(s:Summation)
        RETURN s.text AS text
    """
    with _driver() as d, d.session(database=DB) as sess:
        rec = sess.run(q, stid=stid).single()
    return (rec["text"] if rec else None) or None


# ------------------------------------------------------------------ embedding

class Embedder:
    """all-MiniLM-L6-v2 with a global PMID->vector cache; batched encode for uncached texts."""

    def __init__(self, col):
        self.model = create_sentence_transformer()
        self.col = col
        self.vec = {}          # pmid -> np.ndarray (normalized)
        self.text_cache = {}   # pmid -> abstract str

    def _texts(self, pmids):
        missing = [p for p in pmids if p not in self.text_cache]
        for i in range(0, len(missing), 50000):
            chunk = missing[i:i + 50000]
            for doc in self.col.find({"pmid": {"$in": chunk}},
                                     {"pmid": 1, "abstract": 1, "_id": 0}):
                self.text_cache[str(doc["pmid"])] = doc.get("abstract") or ""
        return {p: self.text_cache.get(p, "") for p in pmids}

    def ensure(self, pmids):
        """Embed (and cache) any pmids not yet vectorized, in one batched encode."""
        todo = [p for p in pmids if p not in self.vec]
        if not todo:
            return
        texts = self._texts(todo)
        mat = self.model.encode([texts[p] for p in todo], batch_size=128,
                                show_progress_bar=False, normalize_embeddings=True)
        for p, v in zip(todo, np.asarray(mat, dtype=np.float32)):
            self.vec[p] = v

    def embed_query(self, text):
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def rank(self, query_vec, pool_pmids):
        """Return pool_pmids sorted by descending cosine to query (vectors are L2-normalized)."""
        self.ensure(pool_pmids)
        mat = np.stack([self.vec[p] for p in pool_pmids])   # (N, d), normalized
        sims = mat @ query_vec                              # cosine == dot for unit vectors
        order = np.argsort(-sims, kind="stable")
        return [pool_pmids[i] for i in order]


# ------------------------------------------------------------------ gene selection + scoring

def pick_valid_genes(col, need):
    """Seeded walk over all annotated genes; keep the first `need` with a deterministically
    chosen reaction that has both a DB summary and >=1 Mongo-available cited PMID."""
    genes = all_annotated_genes()
    random.Random(f"{BASE_SEED}|genes").shuffle(genes)
    chosen, skipped = [], []
    for gene in genes:
        if len(chosen) >= need:
            break
        rxns = reactions_for_gene(gene)
        if not rxns:
            continue
        stid, name = random.Random(f"{BASE_SEED}|{gene}|reaction_pick").choice(rxns)
        summ = reaction_summary(stid)
        if not summ:
            skipped.append((gene, "no_summary"))
            continue
        tp_all = reaction_pmids(stid)
        tp = sorted(mongo_available(tp_all, col) & set(tp_all))
        if not tp:
            skipped.append((gene, "no_tp_abstracts"))
            continue
        if len(tp) >= POOL_SIZE:
            skipped.append((gene, f"too_many_tp({len(tp)})"))
            continue
        chosen.append({"gene": gene, "stid": stid, "reaction": name,
                       "summary": summ, "tp": tp})
    return chosen, skipped


def score_pool(ranked, tp_set, x):
    ranks = {p: i + 1 for i, p in enumerate(ranked)}
    correct_ranks = sorted(ranks[p] for p in tp_set if p in ranks)
    hits = sum(1 for r in correct_ranks if r <= x)
    return hits / x, correct_ranks


# ------------------------------------------------------------------ main / benchmark

def benchmark():
    col = _mongo()
    print(f"[{datetime.datetime.now():%H:%M:%S}] querying Reactome-wide cited PMIDs ...", flush=True)
    universe_all = reactome_wide_pmids()
    universe = sorted(mongo_available(universe_all, col))
    print(f"Reactome-wide distinct cited PMIDs : {len(universe_all)}", flush=True)
    print(f"  with abstract in Mongo (universe): {len(universe)}", flush=True)

    # embedding throughput on a real sample
    emb = Embedder(col)
    sample = universe[:300]
    emb._texts(sample)
    t0 = time.perf_counter()
    _ = emb.model.encode([emb.text_cache.get(p, "") for p in sample], batch_size=128,
                         show_progress_bar=False, normalize_embeddings=True)
    dt = time.perf_counter() - t0
    rate = len(sample) / dt if dt else float("nan")
    print(f"Embed rate (batched, {len(sample)} abstracts): {rate:.0f} abstracts/sec "
          f"({dt:.1f}s for {len(sample)})", flush=True)

    # distinct negatives actually embedded across 30 draws (expected union), pool ~1000 each
    U = len(universe)
    exp_distinct = U * (1 - (1 - POOL_SIZE / U) ** (N_GENES * N_DRAWS))
    print(f"Expected distinct negatives embedded across {N_GENES*N_DRAWS} draws: "
          f"~{exp_distinct:,.0f}  (one-time embed ~{exp_distinct/rate/60:.1f} min at that rate)", flush=True)
    print(f"Full-universe embed (upper bound {U:,}): ~{U/rate/60:.1f} min", flush=True)
    print("Per-draw scoring after embed (sample 1000 + cosine + sort): sub-second.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true",
                    help="only report universe size + embed rate + runtime estimate")
    args = ap.parse_args()
    if args.benchmark:
        benchmark()
        return

    col = _mongo()
    print(f"[{datetime.datetime.now():%H:%M:%S}] building negative universe ...", flush=True)
    universe = sorted(mongo_available(reactome_wide_pmids(), col))
    print(f"Negative universe (Mongo-available Reactome PMIDs): {len(universe)}", flush=True)

    print(f"[{datetime.datetime.now():%H:%M:%S}] selecting {N_GENES} valid genes (seed={BASE_SEED}) ...", flush=True)
    genes, skipped = pick_valid_genes(col, N_GENES)
    for g in genes:
        print(f"    {g['gene']:<10} X={len(g['tp']):<3} {g['stid']}  {g['reaction'][:55]}", flush=True)
    if skipped:
        print(f"    (skipped {len(skipped)}: "
              + ", ".join(f"{gg}:{why}" for gg, why in skipped[:12])
              + (" ..." if len(skipped) > 12 else "") + ")", flush=True)

    emb = Embedder(col)
    os.makedirs("data", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["gene"], str(row["draw"])))
        if done:
            print(f"Resuming: {len(done)} (gene,draw) rows already in {OUT}", flush=True)
    new_file = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); fh.flush()

    for gi, g in enumerate(genes, 1):
        gene, stid, name, summ, tp = g["gene"], g["stid"], g["reaction"], g["summary"], g["tp"]
        x = len(tp)
        tp_set = set(tp)
        qv = emb.embed_query(summ)
        elig = [p for p in universe if p not in tp_set]
        for draw in range(N_DRAWS):
            if (gene, str(draw)) in done:
                print(f"  [{gi}/{len(genes)}] {gene} draw {draw}: already done, skip", flush=True)
                continue
            seed = f"{BASE_SEED}|{gene}|{draw}"
            t0 = time.perf_counter()
            negs = random.Random(seed).sample(elig, POOL_SIZE - x)
            pool = tp + negs
            ranked = emb.rank(qv, pool)
            hit_rate, correct_ranks = score_pool(ranked, tp_set, x)
            w.writerow({
                "gene": gene, "reaction": name, "reaction_stid": stid,
                "n_true_positives": x, "pool_size": len(pool), "draw": draw,
                "draw_seed": seed, "hit_rate": round(hit_rate, 4),
                "correct_ranks": json.dumps(correct_ranks),
            })
            fh.flush(); os.fsync(fh.fileno())
            print(f"  [{gi}/{len(genes)}] {gene} draw {draw}: hit_rate={hit_rate:.2f} "
                  f"({sum(1 for r in correct_ranks if r <= x)}/{x})  "
                  f"ranks={correct_ranks[:8]}{'...' if len(correct_ranks) > 8 else ''}  "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)

    fh.close()
    print(f"\n[{datetime.datetime.now():%H:%M:%S}] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
