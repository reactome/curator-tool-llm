from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import ReactomeDraft
from curator_llm.models.session import ExistingMatch, Issue
from curator_llm.models.usage import UsageEntry


@dataclass
class PipelineInput:
    pmid: Optional[str] = None
    pdf_path: Optional[str] = None
    focus: Optional[str] = None        # gene/protein the curator cares about; None = whole paper


@dataclass
class PipelineResult:
    draft: ReactomeDraft
    evidence: List[Evidence]
    issues: List[Issue] = field(default_factory=list)
    paper: Optional[dict] = None       # PaperText.to_dict()
    existing: List[ExistingMatch] = field(default_factory=list)
    usage: List[UsageEntry] = field(default_factory=list)
    replayed: bool = False             # served from a saved result: the usage is the original run's


class Pipeline(Protocol):
    def run(self, spec: PipelineInput, report: Callable[[str], None]) -> PipelineResult:
        """Paper -> verified evidence -> typed, resolved draft. Raises on a failure that yields nothing."""
        ...
