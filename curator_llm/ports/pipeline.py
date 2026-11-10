import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Protocol

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import ReactomeDraft
from curator_llm.models.session import ExistingMatch, Issue
from curator_llm.models.usage import UsageEntry


@dataclass
class PaperRef:
    """One paper: a PubMed ID (open access in PubMed Central) or an uploaded PDF."""
    pmid: Optional[str] = None
    pdf_path: Optional[str] = None
    label: Optional[str] = None        # a name to show, e.g. the uploaded file's name
    # a review article keeps the reactions it only cites; a research paper does not. None = decide (services/paper_kind:
    # PubMed for a PMID, research for a PDF)
    review: Optional[bool] = None
    # the PubMed ID of an uploaded PDF (the curator's, or found in it: services/paper_identity). The text still comes
    # from the PDF; the ID is the paper's identity: its reactions cite it, and PubMed says whether it is a review.
    pdf_pmid: Optional[str] = None

    @property
    def source(self) -> str:
        """The tag extraction puts on this paper's reactions: 'PMID:24751536', or the PDF's file name."""
        return f'PMID:{self.pmid}' if self.pmid else os.path.basename(self.pdf_path or '')


@dataclass
class PipelineInput:
    pmid: Optional[str] = None
    pdf_path: Optional[str] = None
    focus: Optional[str] = None        # gene/protein the curator cares about; None = whole paper
    papers: List[PaperRef] = field(default_factory=list)   # several papers, merged into ONE draft (pmid/pdf_path unused)
    review: Optional[bool] = None      # for every paper that does not say itself (PaperRef.review); None = decide
    pdf_pmid: Optional[str] = None     # the PubMed ID of the one PDF (PaperRef.pdf_pmid)

    def refs(self) -> List[PaperRef]:
        """The papers to annotate: `papers` if given, else the single pmid or pdf_path. `review` here fills in the papers
        that leave it open; a paper's own setting wins."""
        if self.papers:
            return [replace(r, review=self.review) if r.review is None and self.review is not None else r
                    for r in self.papers]
        return ([PaperRef(self.pmid, self.pdf_path, review=self.review, pdf_pmid=self.pdf_pmid if self.pdf_path else None)]
                if (self.pmid or self.pdf_path) else [])


@dataclass
class PipelineResult:
    draft: ReactomeDraft
    evidence: List[Evidence]
    issues: List[Issue] = field(default_factory=list)
    paper: Optional[dict] = None       # PaperText.to_dict()
    existing: List[ExistingMatch] = field(default_factory=list)
    usage: List[UsageEntry] = field(default_factory=list)
    replayed: bool = False             # served from a saved result: the usage is the original run's
    papers: Dict[str, dict] = field(default_factory=dict)   # every paper's PaperText.to_dict(), by source tag
    reviews: Dict[str, bool] = field(default_factory=dict)  # review (True) or research, by source tag (paper_kind)
    # how each paper's extraction ended (ExtractionResult.paper_status), and whether the whole run is complete: every
    # paper extracted (or without usable text) and merged. An incomplete run is not saved as a snapshot.
    paper_status: Dict[str, dict] = field(default_factory=dict)
    complete: bool = True


class PipelineFailed(RuntimeError):
    """No reactions came out. Carries how each paper's extraction ended, so a caller can tell papers worth another
    try (incomplete) from those that will not change (no usable text)."""

    def __init__(self, message: str, paper_status: Optional[Dict[str, dict]] = None):
        super().__init__(message)
        self.paper_status = dict(paper_status or {})


class Pipeline(Protocol):
    def run(self, spec: PipelineInput, report: Callable[[str], None]) -> PipelineResult:
        """Paper -> verified evidence -> typed, resolved draft. Raises on a failure that yields nothing."""
        ...
