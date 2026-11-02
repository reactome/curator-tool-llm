import os
from dataclasses import dataclass, field
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

    def refs(self) -> List[PaperRef]:
        """The papers to annotate: `papers` if given, else the single pmid or pdf_path."""
        if self.papers:
            return list(self.papers)
        return [PaperRef(self.pmid, self.pdf_path)] if (self.pmid or self.pdf_path) else []


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


class Pipeline(Protocol):
    def run(self, spec: PipelineInput, report: Callable[[str], None]) -> PipelineResult:
        """Paper -> verified evidence -> typed, resolved draft. Raises on a failure that yields nothing."""
        ...
