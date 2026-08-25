"""ReactomeCurator — the deterministic Curator (agent 1 of 3).

Given a gene (and optionally the Reviewer's `adjustment` for a re-run), run ONE
deterministic annotation attempt:

    retrieval  (ReactomeLiteratureExtractor)      -> scored top-N papers
    placement  (FI-partner enrichment, LLM-free)  -> predicted pathway + confidence
    full text  (ReactomeFullTextAnalyzer)         -> reactions + evidence

and bundle EVERY signal the Reviewer needs into one CuratorResult. The Curator does no
LLM reasoning of its own — placement is deterministic; the only LLM work is inside the
tools (retrieval curator-judge, the partner's extraction/review).

It's a pure function of (gene, adjustment): the Reviewer proposes an adjustment, the
orchestrator re-invokes run(), and the SAME code produces a different result only
because the arguments changed. The Curator never re-runs itself or talks to the
Reviewer — the orchestrator (run_curator.py) owns that loop and its retry cap.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from GenePathwayAnnotator import GenePathwayAnnotator
from ReactomeLiteratureExtractor import ReactomeLiteratureExtractor
from ReactomeFullTextAnalyzer import ReactomeFullTextAnalyzer
import ReactomeUtils as utils
import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class CuratorResult:
    """One annotation attempt — everything the Reviewer reads to judge sufficiency."""
    gene: str
    accession: Optional[str]
    retrieval: Dict[str, Any]   # queries, pool sizes, selected papers, rubric scores
    placement: Dict[str, Any]   # predicted pathway + confidence (deterministic)
    fulltext: Dict[str, Any]    # resolution counts, per-paper review scores, reactions
    adjustment_applied: Dict[str, Any] = field(default_factory=dict)

    def has_reactions(self) -> bool:
        return bool(self.fulltext.get("reactions"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReactomeCurator:
    def __init__(self, gene_annotator: Optional[GenePathwayAnnotator] = None,
                 enable_full_text: bool = True,
                 fulltext_index: Optional[Dict[str, str]] = None) -> None:
        self.gene_annotator = gene_annotator or GenePathwayAnnotator()
        self.extractor = ReactomeLiteratureExtractor(self.gene_annotator)
        self.analyzer = ReactomeFullTextAnalyzer()
        self.enable_full_text = enable_full_text
        self.fulltext_index = fulltext_index or {}

    def run(self, gene: str, adjustment: Optional[Dict[str, Any]] = None,
            max_papers: int = 5) -> CuratorResult:
        gene = (gene or "").strip().upper()
        logger.info(f"Curator attempt for {gene} (adjustment={adjustment or 'none'})")

        retrieval = self.extractor.extract(gene, max_papers=max_papers, adjustment=adjustment)
        placement = self._placement(gene)

        fulltext = {"manifest": {}, "counts": {"pdf": 0, "xml": 0, "miss": 0},
                    "per_paper": [], "reactions": []}
        if self.enable_full_text and retrieval.get("papers"):
            fulltext = self.analyzer.analyze(gene, retrieval["papers"], self.fulltext_index)

        return CuratorResult(
            gene=gene,
            accession=retrieval.get("accession"),
            retrieval=self._retrieval_summary(retrieval),
            placement=placement,
            fulltext=fulltext,
            adjustment_applied=retrieval.get("adjustment_applied", {}),
        )

    # ------------------------------------------------------------------ helpers
    def _placement(self, gene: str) -> Dict[str, Any]:
        """Deterministic FI-partner pathway placement (no LLM). Bundles the predicted pathway +
        confidence for the Reviewer, and keeps the raw dict / primary candidate for build_instances."""
        placement = utils.suggest_pathway_placement(gene)
        confident = utils.is_confident_placement(placement)
        primary = placement.get("primary")
        return {
            "predicted_pathway": primary["pathway_name"] if primary else None,
            "confident": confident,
            "status": placement.get("status"),
            "primary": primary,
            "secondary": placement.get("secondary", []),
            "message": (None if confident else
                        "No confident FI-partner placement; reactions can be returned unplaced "
                        "for a curator to place manually."),
            "_raw": placement,   # passed straight to build_instances when confident
        }

    @staticmethod
    def _retrieval_summary(retrieval: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten the extractor's output into the signals the Reviewer cares about."""
        scored = retrieval.get("scored", {}) or {}
        return {
            "name_query": retrieval.get("name_query"),
            "context_query": retrieval.get("context_query"),
            "pool_size": retrieval.get("pool_size"),
            "candidate_pool": retrieval.get("candidate_pool"),
            "papers": retrieval.get("papers", []),
            "mean_score": scored.get("mean_score"),
            "per_paper_scores": scored.get("per_paper", []),
            "dropped_below_threshold": retrieval.get("dropped_below_threshold"),
        }

    def build_instances_args(self, result: CuratorResult):
        """Map an approved CuratorResult's placement into (placement, placement_status) for
        reaction_to_instances.build_instances — mirrors CrewAILiteratureAnnotator's contract
        (raw placement only when confident; status dict always)."""
        pl = result.placement
        confident = pl.get("confident")
        placement_arg = pl.get("_raw") if confident else None
        placement_status = {
            "confident": confident,
            "status": pl.get("status"),
            "primary": pl.get("primary") if confident else None,
            "message": pl.get("message"),
        }
        return placement_arg, placement_status
