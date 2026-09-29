"""ReactomeLiteratureExtractor — the literature retrieval engine (Curator Tool 1).

Given a gene symbol, return the top-N PubMed papers a curator should annotate from,
each with a curator-rubric score. That's the WHOLE job of this file: gene in, scored
PMIDs out. Full-text resolution/extraction and reaction->instance conversion are
deliberately NOT here — they are separate tools.

This is an ENGINE (a plain class you call directly), not an agent "tool". A thin
BaseTool wrapper in ReactomeTools.py exposes `extract()` to the Curator; the agent
tool is ~10 lines and just forwards to this.

Pipeline (all inside `extract()`):
  1. resolve the gene's UniProt accession        (GenePathwayAnnotator -> Reactome graph / UniProt)
  2. build the query pair                          (QueryBuilder: name+synonyms  vs  pathway/partner)
  3. run both PubMed E-Searches, union by PMID     (-> candidate pool)
  4. cross-encoder re-rank the pool                (TextEmbedder, against gene-specific pathway text)
  5. LLM curator-judge: score, drop below floor,   (CuratorRubric.judge_select)
     return the top `max_papers` with scores

Retry / feedback loop: the engine is DETERMINISTIC, but `extract(..., adjustment=...)`
accepts the Reviewer's proposed changes so a re-run can behave differently WITHOUT the
Reviewer reaching inside a run. The Reviewer proposes; the orchestrator caps retries;
the Curator re-invokes with the adjustment. Honored keys (see `_apply_adjustment`):
  additional_terms  -> broaden the query (OR'd into the searches)
  avoid_pmids       -> exclude PMIDs from selection (surfaces the next-best candidates;
                       e.g. the all-misses case: drop the ones with no full text, re-pick)
  min_score         -> lower/raise the rubric floor
  candidate_pool    -> widen the cross-encoder pool the judge sees
  max_papers        -> select more/fewer
Knobs that belong to LATER stages are intentionally NOT acted on here: `prefer_fulltext`
(needs a PMC-availability probe — couples to the full-text tool) and `target_pathways`
(a build_instances/placement concern). They're accepted-and-ignored so the Reviewer can
emit one adjustment object; each stage takes what applies to it.

Why this can be simple: it runs synchronously and OFF any event loop, so unlike the
CrewAI path it can call the blocking LLM description/judge builders inline instead of
precomputing them in worker threads and stashing them on a shared annotator. Steps 2-4
are LLM-free; only step 1 (accession), step 5 (judge), and the optional rerank-context
prose touch an LLM.
"""

import logging
from typing import Any, Dict, List, Optional

from GenePathwayAnnotator import GenePathwayAnnotator
from QueryBuilder import (build_retrieval_query_pair, get_reranking_target,
                          build_judge_context, build_gene_specific_pathway_descriptions,
                          build_gene_specific_enriched_pathway_description,
                          build_query_and_search_terms)
from CuratorRubric import judge_select
from TextEmbedder import _get_cross_encoder
import token_profiler
import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)

# Cross-encoder candidate pool size and rubric floor. 1-2 = "not usable for a specific
# annotation"; 3+ = at least weak background. Mirrors CrewAILiteratureAnnotator's constants.
DEFAULT_CANDIDATE_POOL = 50
DEFAULT_MIN_SCORE = 5
DEFAULT_MAX_PAPERS = 5


class ReactomeLiteratureExtractor:
    """Retrieve + cross-encoder re-rank + LLM curator-judge -> scored top-N PMIDs for a gene."""

    def __init__(self, gene_annotator: Optional[GenePathwayAnnotator] = None) -> None:
        # GenePathwayAnnotator is reused only for its configured PubMed retriever and the
        # deterministic accession resolver — both cheap. A fresh one is fine for standalone use.
        self.gene_annotator = gene_annotator or GenePathwayAnnotator()

    # ------------------------------------------------------------------ public API
    def extract(self, gene: str,
                max_papers: int = DEFAULT_MAX_PAPERS,
                candidate_pool: int = DEFAULT_CANDIDATE_POOL,
                min_score: int = DEFAULT_MIN_SCORE,
                adjustment: Optional[Dict[str, Any]] = None,
                placement: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gene -> the papers to annotate from.

        `adjustment`: the Reviewer's proposed changes for a re-run (or None on the first
        attempt). See the module docstring for honored keys.

        Returns:
            {
              "gene", "accession",
              "name_query", "context_query", "pool_size",   # retrieval provenance
              "candidate_pool",                              # #docs the judge saw
              "papers":  [selected candidate dicts, highest rubric score first],
              "scored":  <full CuratorRubric scoring, for reporting/eval>,
              "dropped_below_threshold": <int>,
              "adjustment_applied": <the adjustment dict, for the Reviewer's audit trail>,
            }
        `papers` may be FEWER than max_papers — an honest signal a gene has little usable
        literature. Empty `papers` (with pool_size 0) means retrieval found no candidates.
        """
        gene = (gene or "").strip().upper()
        max_papers, candidate_pool, min_score, additional_terms, avoid_pmids = \
            self._apply_adjustment(adjustment, max_papers, candidate_pool, min_score)
        logger.info(f"Literature extraction for {gene} (adjustment={adjustment or 'none'})")

        accession = self.gene_annotator.resolve_uniprot_accession(gene)
        logger.info(f"Resolved accession for {gene}: {accession}")

        # Rerank context: gene-specific pathway descriptions (best), else a gate-fail gene
        # background. Passed straight into re-rank + judge (no worker-thread/stash needed here).
        description, pathway_descs = self._build_rerank_context(gene, placement=placement)

        cand = self._retrieve_candidates(gene, candidate_pool, description, pathway_descs,
                                         additional_terms=additional_terms, avoid_pmids=avoid_pmids,
                                         placement=placement)
        candidates = cand["papers"]
        base = {"gene": gene, "accession": accession,
                "name_query": cand["name_query"], "context_query": cand["context_query"],
                "pool_size": cand["pool_size"], "adjustment_applied": adjustment or {}}
        if not candidates:
            logger.warning(f"No candidates retrieved for {gene}")
            return {**base, "candidate_pool": 0, "papers": [], "scored": {},
                    "dropped_below_threshold": 0}

        # Judge prose: reuse whatever rerank context we already computed (no extra LLM call);
        # build_judge_context then appends the gene's explicit Reactome pathways + FI partners.
        if description:
            prose = description
        elif pathway_descs:
            prose = " ".join(str(v) for v in pathway_descs.values())
        else:
            prose = None
        judge_context = build_judge_context(gene, description=prose)

        with token_profiler.label("litextract_judge"):
            judged = judge_select(gene, judge_context, candidates,
                                  max_papers=max_papers, min_score=min_score)

        logger.info(
            f"Judge selected {len(judged['selected'])}/{len(candidates)} candidates for {gene} "
            f"(pool {cand['pool_size']}; dropped {judged['dropped_below_threshold']} below floor {min_score})")

        return {
            **base,
            "candidate_pool": len(candidates),
            "papers": judged["selected"],
            "scored": judged["scored"],
            "dropped_below_threshold": judged["dropped_below_threshold"],
        }

    # ------------------------------------------------------------------ adjustment
    @staticmethod
    def _apply_adjustment(adjustment: Optional[Dict[str, Any]],
                          max_papers: int, candidate_pool: int, min_score: int):
        """Fold the Reviewer's adjustment over the defaults. Honors only the retrieval-relevant
        keys; other keys (prefer_fulltext, target_pathways) are for later stages and ignored here."""
        adj = adjustment or {}
        max_papers = adj.get("max_papers", max_papers)
        candidate_pool = adj.get("candidate_pool", candidate_pool)
        min_score = adj.get("min_score", min_score)
        additional_terms = adj.get("additional_terms", "") or ""
        avoid_pmids = {str(p) for p in (adj.get("avoid_pmids") or [])}
        return max_papers, candidate_pool, min_score, additional_terms, avoid_pmids

    # ------------------------------------------------------------------ steps 2-4 (LLM-free)
    def _retrieve_candidates(self, gene: str, candidate_pool: int,
                             description: Optional[str], pathway_descs: Optional[dict],
                             fetch_papers: int = 200, additional_terms: str = "",
                             avoid_pmids: Optional[set] = None,
                             placement: Optional[dict] = None) -> dict:
        """Stage-1 result-set merge + cross-encoder re-rank -> top-`candidate_pool` paper dicts
        (with abstracts). LLM-FREE. Lifted from the old LiteratureSearchTool so retrieval logic
        lives with the extractor, not in the agent-tool file."""
        name_query, context_query = build_retrieval_query_pair(gene)
        if additional_terms:
            if context_query:
                context_query = f"({context_query}) OR {additional_terms}"
            else:
                name_query = f"({name_query}) OR {additional_terms}"

        pool = self._merge_search(name_query, context_query,
                                  name_fetch=max(1, fetch_papers // 2),
                                  context_fetch=fetch_papers)
        # Drop excluded PMIDs BEFORE re-rank so the next-best candidates take their place — this is
        # what makes a "these all missed full text, try different papers" re-run surface new PMIDs.
        if avoid_pmids:
            before = len(pool)
            pool = [d for d in pool if str(d.get("uid")) not in avoid_pmids]
            logger.info(f"Excluded {before - len(pool)} avoid_pmids from the {before}-doc pool")
        ranked = self._rerank(gene, pool, candidate_pool, description, pathway_descs, placement=placement)
        return {
            "name_query": name_query,
            "context_query": context_query,
            "pool_size": len(pool),
            "papers": self._format_papers(ranked),
        }

    def _merge_search(self, name_query: str, context_query: str,
                      name_fetch: int, context_fetch: int) -> List[dict]:
        """Run the name and context queries as SEPARATE E-Searches and union by PMID. Each search
        gets its own relevance ranking, so gene-specific (name) papers are guaranteed into the pool
        rather than buried below the fetch cap by broad pathway/partner results (the RAB6C fix).
        Name results first so they win the degenerate no-rerank-target fallback in _rerank."""
        seen, pool = set(), []
        for query, cap in ((name_query, name_fetch), (context_query, context_fetch)):
            if not query:
                continue
            retriever = self.gene_annotator._get_pubmed_retriver(top_k_results=cap)
            for d in retriever.lazy_load(query=query):
                if d is None:
                    continue
                uid = d.get("uid")
                if uid in seen:
                    continue
                seen.add(uid)
                pool.append(d)
        return pool

    def _rerank(self, gene: str, pool: List[dict], top_k: int,
                description: Optional[str], pathway_descs: Optional[dict],
                placement: Optional[dict] = None) -> List[dict]:
        """Re-rank pool docs by MAX cross-encoder score against the gene's re-ranking targets,
        returning the top `top_k`. Falls back to pool order if there are no embeddable abstracts
        or no targets. Targets are passed in (description/pathway_descs) rather than read off a
        shared annotator, since this runs off-loop."""
        embeddable = [d for d in pool if d.get("Summary")]
        if not embeddable:
            return pool[:top_k]

        targets = get_reranking_target(gene, description_override=description,
                                       pathway_descriptions=pathway_descs, placement=placement)
        if not targets:
            return pool[:top_k]

        cross_encoder = _get_cross_encoder()
        pairs = [(t, d["Summary"]) for d in embeddable for t in targets]
        scores = cross_encoder.predict(pairs)
        n_t = len(targets)
        for i, d in enumerate(embeddable):
            d["cross_score"] = max(float(s) for s in scores[i * n_t:(i + 1) * n_t])
        embeddable.sort(key=lambda d: d["cross_score"], reverse=True)
        return embeddable[:top_k]

    @staticmethod
    def _format_papers(docs: List[dict]) -> List[dict]:
        """Project raw pool docs (uid/Summary/...) into the public paper shape."""
        return [{
            "pmid": d.get("uid", ""),
            "title": d.get("title", ""),
            "abstract": d.get("Summary", ""),
            "authors": d.get("authors", ""),
            "journal": d.get("journal", ""),
            "year": d.get("year", ""),
            "cross_score": d.get("cross_score"),
        } for d in docs]

    # ------------------------------------------------------------------ rerank context
    def _build_rerank_context(self, gene: str,
                              placement: Optional[dict] = None) -> tuple[Optional[str], dict]:
        """Compute the re-rank/judge target text for `gene`.

        Prefers per-pathway gene-SPECIFIC descriptions (has-data genes; validated to rerank
        specific annotatable papers above broad reviews), then the enriched-pathway description
        (cold-start gate-pass genes), then a gene background (gate-fail). Each builder self-gates
        to {} for the cases it doesn't apply to, so we just try them in order. All guarded — a
        failure degrades to the next fallback, never breaks retrieval.

        Returns (description, pathway_descriptions): at most one is non-empty; both empty means
        get_reranking_target/build_judge_context fall back to their own defaults.
        """
        pathway_descs: dict = {}
        try:
            with token_profiler.label("litextract_desc_per_pathway"):
                pathway_descs = build_gene_specific_pathway_descriptions(gene) or {}
                if not pathway_descs:
                    pathway_descs = build_gene_specific_enriched_pathway_description(
                        gene, placement=placement) or {}
        except Exception as e:
            logger.warning(f"Gene-specific pathway descriptions failed for {gene}: {e}")
            pathway_descs = {}

        description = None
        if not pathway_descs:
            try:
                with token_profiler.label("litextract_desc_gate_fail"):
                    _, description = build_query_and_search_terms(gene)
            except Exception as e:
                logger.warning(f"Gene background generation failed for {gene}: {e}")
                description = None
        return description, pathway_descs


# --------------------------------------------------------------------------- CLI smoke test
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser(description="Retrieve + judge the top papers for a gene.")
    ap.add_argument("gene", help="Gene symbol, e.g. SHANK3")
    ap.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS)
    args = ap.parse_args()

    result = ReactomeLiteratureExtractor().extract(args.gene, max_papers=args.max_papers)
    scored = result.get("scored", {})
    print(f"\n{args.gene}: pool {result['pool_size']} -> {result['candidate_pool']} candidates "
          f"-> {len(result['papers'])} selected "
          f"(dropped {result['dropped_below_threshold']} below floor)")
    print(f"accession: {result['accession']}   pool mean score: {scored.get('mean_score')}")
    by_pmid = {str(pp.get("pmid")): pp.get("score") for pp in scored.get("per_paper", [])}
    for p in result["papers"]:
        print(f"  {p['pmid']:<12} score={by_pmid.get(str(p['pmid']))}  {p['title'][:80]}")
