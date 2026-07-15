"""Sweep harness: tune the two FETCH counts for a gene-pool + pathway-pool result-set merge,
measuring FINAL-5 on-target% AFTER replicating the pipeline's Stage-2 re-rank.

Objective = fraction of the final `max_papers` (post-rerank) that mention the gene/synonyms.
Compares pure re-rank vs reserving R slots for gene-pool papers.

  Ng = gene-name/synonym pool fetch count   (Ng=0 row == current production, pathway-only)
  Np = partner-enrichment pathway pool fetch count
  R  = slots (of the final 5) reserved for the best-reranked gene-pool papers

LLM-free (rerank uses the local MiniLM embedder). Run from repo root in paperqa env.
"""
import os, sys, re
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.append(os.path.join(_ROOT, "reactome_llm"))
sys.path.append(_ROOT)  # TextEmbedder.py lives at the repo root, not in reactome_llm/

import QueryBuilder as qb
import ReactomeNeo4jUtils as neo
from TextEmbedder import sentence_embed, create_sentence_transformer, cosine_similarity

MAX_PAPERS = 5


def build_pools(gene, max_ng, max_np, ann):
    """Fetch gene-pool (synonym query) once at max_ng and pathway-pool once at max_np.
    Both are relevance-ordered by esearch, so pool[:k] == top-k."""
    def fetch(query, k):
        r = ann._get_pubmed_retriver(top_k_results=k)
        try: r.maxdate = "2026/03/31"
        except Exception: pass
        return [d for d in r.lazy_load(query=query) if d is not None and d.get("Summary")]
    gene_q = qb.build_synonym_search_query(gene)          # gene + UniProt synonyms
    path_q = qb.build_partner_enrichment_query(gene)      # enriched pathway names
    return fetch(gene_q, max_ng), fetch(path_q, max_np), gene_q, path_q


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "RAB6C"
    max_ng = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_np = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    from GenePathwayAnnotator import GenePathwayAnnotator
    ann = GenePathwayAnnotator()

    syns = qb.get_uniprot_synonyms(neo.query_accession_for_gene(gene)) or []
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in [gene] + syns if t) + r")\b", re.I)
    def hits(docs): return sum(1 for d in docs if pat.search((d.get("title","") or "")+" "+(d.get("Summary","") or "")))

    genep, pathp, gene_q, path_q = build_pools(gene, max_ng, max_np, ann)
    gene_uids = {d["uid"] for d in genep}

    # Stage-2 rerank target = exactly what the pipeline uses
    targets = qb.get_reranking_target(gene)
    model = create_sentence_transformer()
    tvecs = [sentence_embed(t, model) for t in targets]
    # embed every candidate once
    vec = {}
    for d in genep + pathp:
        u = d["uid"]
        if u not in vec:
            dv = sentence_embed(d["Summary"], model)
            vec[u] = max(cosine_similarity(dv, tv) for tv in tvecs)

    print(f"gene={gene}  synonyms={syns[:4]}")
    print(f"gene-pool query : {gene_q[:90]}")
    print(f"pathway query   : {path_q[:90]}")
    print(f"rerank target   : {targets[0][:90] if targets else '(none)'}...")
    print(f"available: gene-pool={len(genep)} (on-target {hits(genep)}), pathway-pool={len(pathp)} (on-target {hits(pathp)})")
    print(f"\nFinal-{MAX_PAPERS} on-target count after Stage-2 rerank:")
    print(f"{'Ng':>4} {'Np':>4} {'pool':>5} {'pool_hit':>8} | {'R=0(pure)':>9} {'R=1':>5} {'R=2':>5}")

    def final_hits(pool, R):
        scored = sorted(pool, key=lambda d: vec[d["uid"]], reverse=True)
        if R <= 0:
            final = scored[:MAX_PAPERS]
        else:
            reserved = [d for d in scored if d["uid"] in gene_uids][:R]
            rids = {d["uid"] for d in reserved}
            rest = [d for d in scored if d["uid"] not in rids][:MAX_PAPERS - len(reserved)]
            final = reserved + rest
        return hits(final), len(final)

    for Ng in [0, 10, 20, min(30, max_ng)]:
        for Np in [25, 50, 100, 200]:
            if Np > max_np: continue
            pool, seen = [], set()
            for d in genep[:Ng] + pathp[:Np]:
                if d["uid"] not in seen:
                    seen.add(d["uid"]); pool.append(d)
            ph = hits(pool)
            r0 = final_hits(pool, 0)[0]
            r1 = final_hits(pool, 1)[0]
            r2 = final_hits(pool, 2)[0]
            print(f"{Ng:>4} {Np:>4} {len(pool):>5} {ph:>8} | {r0:>9} {r1:>5} {r2:>5}")


if __name__ == "__main__":
    main()
