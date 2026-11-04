from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence


@dataclass
class ExtractionResult:
    ok: bool = False
    reactions: List[dict] = field(default_factory=list)     # merged reactions (unmerged if merge failed)
    n_extracted: int = 0
    n_merged: int = 0
    extraction_path: Optional[str] = None
    merged_path: Optional[str] = None
    review_path: Optional[str] = None
    review_score: Optional[float] = None
    usage: Dict[str, int] = field(default_factory=dict)     # LLM tokens spent by the steps, summed
    step_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)   # the same, per step: extraction / merge / review
    error: Optional[str] = None                             # why it failed, with the step's stderr tail
    warnings: List[str] = field(default_factory=list)       # degraded but usable (merge failed, review failed)


class Extractor(Protocol):
    def extract(self, pmid: Optional[str], pdf_path: Optional[str], gene: str,
                report: Optional[Callable[[str], None]] = None, review: bool = False) -> ExtractionResult:
        """One paper -> reactions (extract, merge, optional review). Never raises: failures are in `error`.
        `report` (optional) is called with a short progress message as the work advances. `review` (optional) says the
        paper is a review article, so reactions it only cites are kept; from a research paper they are left out."""
        ...

    def extract_many(self, refs: Sequence, gene: str,
                     report: Optional[Callable[[str], None]] = None) -> ExtractionResult:
        """Several papers (PaperRef, each saying whether it is a review) -> ONE set of reactions, merged within each paper
        and then across papers. Optional:
        an extractor that handles a single paper only may omit it."""
        ...
