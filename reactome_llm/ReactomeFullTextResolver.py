"""ReactomeFullTextResolver — full-text resolution (Curator Tool 3).

Given the papers the literature extractor selected, resolve each PMID to a full-text file:
the curator's local PDF, a downloaded/cached PMC XML, or a miss. Resolution only — no
extraction. The resulting manifest is handed to Tool 4 (ReactomeFullTextAnalyzer), which
does the actual reaction extraction.

Thin wrapper over the FullTextResolver module (same pattern as ReactomeFullTextAnalyzer
wrapping fulltext_extractor). Standalone and synchronous.
"""

import logging
from typing import Any, Dict, List, Optional

import FullTextResolver
import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)


class ReactomeFullTextResolver:
    """Resolve selected PMIDs to full-text files -> a manifest + hit-rate counts."""

    @staticmethod
    def build_index(papers_dir: Optional[str]) -> Dict[str, str]:
        """Index a curator's local PDF folder -> {pmid: filepath}. Empty when no folder given."""
        return FullTextResolver.build_index(papers_dir) if papers_dir else {}

    def resolve(self, gene: str, papers: List[dict],
                fulltext_index: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Returns:
            {
              "manifest": {pmid: {source: pdf|xml|miss, path|...}},
              "counts":   {"pdf": n, "xml": n, "miss": n},
            }
        Never raises — on failure returns an empty manifest so the pipeline continues.
        """
        gene = (gene or "").strip().upper()
        empty = {"manifest": {}, "counts": {"pdf": 0, "xml": 0, "miss": 0}}

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
        logger.info(f"{gene}: resolved {counts['pdf']} PDF · {counts['xml']} XML · {counts['miss']} miss")
        return {"manifest": manifest, "counts": counts}
