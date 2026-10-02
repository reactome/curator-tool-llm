from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


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
    def extract(self, pmid: Optional[str], pdf_path: Optional[str], gene: str) -> ExtractionResult:
        """One paper -> reactions (extract, merge, optional review). Never raises: failures are in `error`."""
        ...
