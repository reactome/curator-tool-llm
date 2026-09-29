"""ReactomeFullTextAnalyzer — reaction extraction from resolved full text (Curator Tool 4).

Takes the manifest produced by Tool 3 (ReactomeFullTextResolver) — i.e. the PMIDs already
resolved to PDF / PMC-XML / miss — and runs the partner's extraction pipeline over the
resolved files, plus an ABSTRACT fallback for the misses, to produce reactions with evidence.
Manifest in, reactions out. It does NOT resolve — that's Tool 3's job.

Thin orchestrator over `fulltext_extractor` (the partner's subprocess-driven extractor);
nothing here touches an event loop.
"""

import logging
from typing import Any, Dict, List

import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)


class ReactomeFullTextAnalyzer:
    """Extract reactions from an already-resolved full-text manifest."""

    def analyze(self, gene: str, papers: List[dict],
                manifest: Dict[str, dict]) -> Dict[str, Any]:
        """`papers` supplies the abstracts (for the miss fallback); `manifest` is Tool 3's output.

        Returns:
            {
              "per_paper": [{pmid, source, n_extracted, n_merged, review_score, reactions, ...}],
              "reactions": [ ...flattened partner reaction dicts across all papers... ],
            }
        Never raises — failures degrade to fewer/no reactions.
        """
        gene = (gene or "").strip().upper()
        if not manifest:
            return {"per_paper": [], "reactions": []}

        # The partner's extractor is optional; if it's absent the pipeline proceeds with 0 reactions
        # (the resolver's manifest/counts still tell the Reviewer the full-text hit-rate).
        try:
            from fulltext_extractor import extract_review_per_paper, extract_abstracts_for_misses
        except Exception:
            logger.info("fulltext_extractor not present; no reactions extracted.")
            return {"per_paper": [], "reactions": []}

        # Full-text papers: independent extract -> merge -> OpenAI review per paper (parallel, capped).
        per_paper = extract_review_per_paper(manifest, gene) or []
        # Abstract fallback for misses: mine the already-in-hand abstract, labeled provenance
        # "abstract" so a curator can weight it below full text (why TANC1-type genes still yield some).
        abstracts_by_pmid = {str(p.get("pmid")): (p.get("abstract") or "")
                             for p in papers if p.get("pmid")}
        abstract_per_paper = extract_abstracts_for_misses(manifest, gene, abstracts_by_pmid) or []

        all_per_paper = per_paper + abstract_per_paper
        # NOTE: no cross-paper dedup yet — duplicate reactions across papers are possible.
        reactions = [rx for p in all_per_paper for rx in (p.get("reactions") or [])]
        logger.info(
            f"{gene}: {len(per_paper)} full-text + {len(abstract_per_paper)} abstract paper(s) "
            f"-> {len(reactions)} reaction(s) total.")
        return {"per_paper": all_per_paper, "reactions": reactions}
