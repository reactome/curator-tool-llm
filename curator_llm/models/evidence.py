"""Evidence as a first-class, code-verified object.

The LLM supplies `quote`, `supports`, `system`, `experimental_species` and `claim_origin`.
Code fills `char_span`, `page`, `section`, `figure` and `verified` (see PaperText.verify), so
the model never has to carry locations and a quote that is not in the paper is detectable.
"""
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Verification(str, Enum):
    UNVERIFIED = 'unverified'   # not yet checked against the paper
    EXACT = 'exact'             # found verbatim (after whitespace/hyphenation normalisation)
    FUZZY = 'fuzzy'             # found with similarity >= the fuzzy threshold
    FAILED = 'failed'           # not found in the paper: must be rejected or relabelled


class ClaimOrigin(str, Enum):
    THIS_PAPER = 'this_paper'
    CITED = 'cited'                       # shown elsewhere; this paper only cites it
    CURATOR_ASSERTION = 'curator_assertion'  # no paper text behind it


class Evidence(BaseModel):
    id: str = ''
    quote: str
    pmid: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    figure: Optional[str] = None
    char_span: Optional[Tuple[int, int]] = None
    verified: Verification = Verification.UNVERIFIED
    match_score: Optional[float] = None
    # Which reaction fields this quote supports, e.g. "catalystActivity", "regulatedBy[0]".
    supports: List[str] = Field(default_factory=list)
    system: Optional[str] = None                  # in_vitro_recombinant, cellular, cell_free ...
    experimental_species: Optional[str] = None    # keeps ortholog provenance the entity name loses
    claim_origin: ClaimOrigin = ClaimOrigin.THIS_PAPER
    cited_reference: Optional[str] = None
    strength: Optional[str] = None                # direct / cellular / cited
