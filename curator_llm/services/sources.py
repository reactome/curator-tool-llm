"""Source tags: how extraction labels where a reaction came from ('PMID:24751536', '24751536 (abstract)',
'PMC123', or a PDF file name)."""
import re
from typing import Optional

_PMID = re.compile(r'^\s*(?:PMID:?\s*)?(\d{5,9})(?!\d)', re.I)


def pmid_of(source: Optional[str]) -> Optional[str]:
    """The PMID a source tag names, or None. Only a tag that STARTS with it counts: a PDF called
    'Smith_20240101.pdf' contains digits but is not a PMID."""
    m = _PMID.match(str(source or ''))
    return m.group(1) if m else None
