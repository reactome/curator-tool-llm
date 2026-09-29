"""ReactomePathwayPlacement — pathway placement (Curator Tool 2).

Deterministic FI-partner enrichment -> a predicted Reactome pathway + a confidence flag.
LLM-FREE. The Curator calls this ONCE per gene and reuses the result for three things:
  1. the retrieval rerank target (cold-start genes — passed into ReactomeLiteratureExtractor),
  2. the CuratorResult signal the Reviewer reads (predicted_pathway + confident),
  3. build_instances grouping (via ReactomeCurator.build_instances_args).

This is the piece that used to be computed redundantly (once inside retrieval's rerank-target
builders, once again in the Curator). Making it its own tool and computing it once removes the
cold-start double-compute.
"""

import logging
from typing import Any, Dict

import ReactomeUtils as utils
import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)


class ReactomePathwayPlacement:
    """FI-partner enrichment -> predicted pathway + confidence, bundled for reuse."""

    def resolve(self, gene: str) -> Dict[str, Any]:
        """Returns a placement bundle:
            {predicted_pathway, confident, status, primary, secondary, message, _raw}
        `_raw` is the untouched suggest_pathway_placement dict — pass it to the retrieval
        rerank-target builders and to build_instances (they expect that raw shape)."""
        gene = (gene or "").strip().upper()
        placement = utils.suggest_pathway_placement(gene)
        confident = utils.is_confident_placement(placement)
        primary = placement.get("primary")
        logger.info(
            f"Placement for {gene}: {placement.get('status')} "
            f"(predicted={primary['pathway_name'] if primary else None}, confident={confident})")
        return {
            "predicted_pathway": primary["pathway_name"] if primary else None,
            "confident": confident,
            "status": placement.get("status"),
            "primary": primary,
            "secondary": placement.get("secondary", []),
            "message": (None if confident else
                        "No confident FI-partner placement; reactions can be returned unplaced "
                        "for a curator to place manually."),
            "_raw": placement,   # raw suggest_pathway_placement result, for rerank + build_instances
        }
