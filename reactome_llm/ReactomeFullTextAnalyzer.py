"""ReactomeFullTextAnalyzer — full-text resolution + reaction extraction (Curator Tool 2).

Given the papers the literature extractor selected, resolve each PMID to a full-text
file (the curator's local PDF or a downloaded/cached PMC XML) or mark it a miss, then
run the partner's extraction pipeline over the resolved files — plus an ABSTRACT
fallback for the misses — to produce reactions with evidence. Selected papers in,
reactions out.

Standalone and synchronous. It delegates the real work to two existing modules:
  - FullTextResolver.resolve_fulltext  (PMID -> pdf / xml / miss manifest)
  - fulltext_extractor.*               (the partner's subprocess-driven extractor)
so this file is a thin orchestrator, not a re-implementation. It lifts the logic that
used to live in CrewAILiteratureAnnotator._resolve_and_extract_fulltext, minus the
event-loop / worker-thread machinery (nothing here touches an event loop).
"""

import logging
from typing import Any, Dict, List, Optional

import FullTextResolver
import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)


class ReactomeFullTextAnalyzer:
    """Resolve selected PMIDs to full text and extract reactions from them."""

    @staticmethod
    def build_index(papers_dir: Optional[str]) -> Dict[str, str]:
        """Index a curator's local PDF folder -> {pmid: filepath}. Empty when no folder given."""
        return FullTextResolver.build_index(papers_dir) if papers_dir else {}

    def analyze(self, gene: str, papers: List[dict],
                fulltext_index: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Resolve + extract for the selected `papers` (each a dict with at least pmid/abstract).

        Returns:
            {
              "manifest":  {pmid: {source: pdf|xml|miss, path|...}},
              "counts":    {"pdf": n, "xml": n, "miss": n},
              "per_paper": [{pmid, source, n_extracted, n_merged, review_score, reactions, ...}],
              "reactions": [ ...flattened partner reaction dicts across all papers... ],
            }
        Never raises for the extraction step — failures degrade to fewer/no reactions.
        """
        gene = (gene or "").strip().upper()
        empty = {"manifest": {}, "counts": {"pdf": 0, "xml": 0, "miss": 0},
                 "per_paper": [], "reactions": []}

        pmids = [str(p.get("pmid")) for p in papers if p.get("pmid")]
        if not pmids:
            return empty

        try:
            manifest = FullTextResolver.resolve_fulltext(pmids, fulltext_index or {}, gene=gene)
        except Exception as e:
            logger.warning(f"Full-text resolution failed for {gene}: {e}")
            return empty
        counts = {k: sum(1 for v in manifest.values() if v.get("source") == k)
                  for k in ("pdf", "xml", "miss")}

        # The partner's extractor is optional; if it's not importable, we still return the manifest
        # so the Reviewer can see the resolution hit-rate — the pipeline just proceeds with 0 reactions.
        try:
            from fulltext_extractor import extract_review_per_paper, extract_abstracts_for_misses
        except Exception:
            logger.info("fulltext_extractor not present; manifest built but no reactions extracted.")
            return {"manifest": manifest, "counts": counts, "per_paper": [], "reactions": []}

        # Full-text papers: independent extract -> merge -> OpenAI review per paper (parallel, capped).
        per_paper = extract_review_per_paper(manifest, gene) or []
        # Abstract fallback for misses: mine the already-in-hand abstract, labeled provenance
        # "abstract" so a curator can weight it below full text. This is why a gene with strong
        # abstracts but no full text (e.g. TANC1) still yields reactions.
        abstracts_by_pmid = {str(p.get("pmid")): (p.get("abstract") or "")
                             for p in papers if p.get("pmid")}
        abstract_per_paper = extract_abstracts_for_misses(manifest, gene, abstracts_by_pmid) or []

        all_per_paper = per_paper + abstract_per_paper
        # NOTE: no cross-paper dedup yet — duplicate reactions across papers are possible.
        reactions = [rx for p in all_per_paper for rx in (p.get("reactions") or [])]
        logger.info(
            f"{gene}: {len(per_paper)} full-text + {len(abstract_per_paper)} abstract paper(s) "
            f"-> {len(reactions)} reaction(s) total.")
        return {"manifest": manifest, "counts": counts,
                "per_paper": all_per_paper, "reactions": reactions}
