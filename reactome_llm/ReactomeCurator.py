"""ReactomeCurator — the deterministic Curator (agent 1 of 3).

Given a gene (and optionally the Reviewer's `adjustment` for a re-run), the Curator calls its
FOUR tools in a clear linear flow and bundles the results into one CuratorResult:

    Tool 1  pathway placement (ReactomePathwayPlacement)        -> predicted pathway + confidence
    Tool 2  literature retrieval (ReactomeLiteratureExtractor)  -> scored top-N papers
    Tool 3  full-text resolver (ReactomeFullTextResolver)       -> manifest (pdf/xml/miss)
    Tool 4  full-text analysis (ReactomeFullTextAnalyzer)       -> reactions + evidence

Placement is computed ONCE here and reused everywhere: it's passed INTO literature extraction
(so retrieval's rerank target doesn't recompute it), stored in CuratorResult for the Reviewer,
and handed to build_instances. The Curator does no LLM reasoning of its own — placement is
deterministic; the only LLM work lives inside the tools (retrieval curator-judge, the partner's
extraction/review).

It's a pure function of (gene, adjustment): the Reviewer proposes an adjustment, the
orchestrator re-invokes run(), and the SAME code produces a different result only because the
arguments changed. The Curator never re-runs itself or talks to the Reviewer — the orchestrator
(run_curator.py) owns that loop and its retry cap.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from GenePathwayAnnotator import GenePathwayAnnotator
from ReactomeLiteratureExtractor import ReactomeLiteratureExtractor
from ReactomePathwayPlacement import ReactomePathwayPlacement
from ReactomeFullTextResolver import ReactomeFullTextResolver
from ReactomeFullTextAnalyzer import ReactomeFullTextAnalyzer
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
        self.placement_tool = ReactomePathwayPlacement()                    # Tool 1
        self.extractor = ReactomeLiteratureExtractor(self.gene_annotator)   # Tool 2
        self.resolver = ReactomeFullTextResolver()                          # Tool 3
        self.analyzer = ReactomeFullTextAnalyzer()                          # Tool 4
        self.enable_full_text = enable_full_text
        self.fulltext_index = fulltext_index or {}

    def run(self, gene: str, adjustment: Optional[Dict[str, Any]] = None,
            max_papers: int = 5, papers_only: bool = False) -> CuratorResult:
        gene = (gene or "").strip().upper()
        logger.info(f"Curator attempt for {gene} (adjustment={adjustment or 'none'}, "
                    f"papers_only={papers_only})")

        # Tool 1 — pathway placement, computed ONCE and reused for retrieval + result + build_instances.
        placement = self.placement_tool.resolve(gene)

        # PAPERS-ONLY: the curator already has the papers they want. Skip Tool 2 (retrieval) entirely
        # and feed the local PDF index straight into Tools 3/4. Placement (Tool 1) still runs so the
        # reactions can be grouped; there is nothing to retry, so run_curator caps this at 1 attempt.
        if papers_only:
            return self._run_papers_only(gene, placement)

        # Tool 2 — literature retrieval; reuse the placement so its rerank target isn't recomputed.
        retrieval = self.extractor.extract(gene, max_papers=max_papers, adjustment=adjustment,
                                           placement=placement.get("_raw"))

        # Tools 3 + 4 — resolve the selected papers to full text, then extract reactions from them.
        fulltext = {"manifest": {}, "counts": {"pdf": 0, "xml": 0, "miss": 0},
                    "per_paper": [], "reactions": []}
        if self.enable_full_text and retrieval.get("papers"):
            resolved = self.resolver.resolve(gene, retrieval["papers"], self.fulltext_index)  # Tool 3
            extracted = self.analyzer.analyze(gene, retrieval["papers"], resolved["manifest"])  # Tool 4
            fulltext = {**resolved, **extracted}   # {manifest, counts} + {per_paper, reactions}

        return CuratorResult(
            gene=gene,
            accession=retrieval.get("accession"),
            retrieval=self._retrieval_summary(retrieval),
            placement=placement,
            fulltext=fulltext,
            adjustment_applied=retrieval.get("adjustment_applied", {}),
        )

    def _run_papers_only(self, gene: str, placement: Dict[str, Any]) -> CuratorResult:
        """Build a CuratorResult from the curator-supplied PDFs, with NO retrieval.

        The full-text index (built by run_curator from --papers-dir) is {pmid: path}, where the
        PMID was recovered from each PDF's page-1 DOI. We treat every indexed PDF as a resolved
        'pdf' source and run Tool 4 (analysis) directly. PDFs with no recoverable DOI/PMID never
        make it into the index, so build_index already reported them as unresolvable.
        """
        index = self.fulltext_index or {}
        papers = [{"pmid": pmid} for pmid in index]
        manifest = {pmid: {"source": "pdf", "path": path} for pmid, path in index.items()}
        counts = {"pdf": len(manifest), "xml": 0, "miss": 0}
        extracted = (self.analyzer.analyze(gene, papers, manifest) if manifest
                     else {"per_paper": [], "reactions": []})
        fulltext = {"manifest": manifest, "counts": counts, **extracted}

        # Retrieval normally resolves the accession; without it, get it from the graph directly.
        accession = None
        try:
            import ReactomeNeo4jUtils as neo4j_utils
            accession = neo4j_utils.query_accession_for_gene(gene)
        except Exception as e:
            logger.warning(f"papers-only: could not resolve accession for {gene}: {e}")

        retrieval = {"name_query": None, "context_query": None, "pool_size": None,
                     "candidate_pool": None, "papers": papers, "mean_score": None,
                     "per_paper_scores": [], "dropped_below_threshold": None, "papers_only": True}
        logger.info(f"papers-only {gene}: {len(manifest)} PDF(s) -> {len(fulltext['reactions'])} reaction(s)")
        return CuratorResult(gene=gene, accession=accession, retrieval=retrieval,
                             placement=placement, fulltext=fulltext, adjustment_applied={})

    # ------------------------------------------------------------------ helpers
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
