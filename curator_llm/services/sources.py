"""Source tags: how extraction labels where a reaction came from ('PMID:24751536', '24751536 (abstract)',
'PMC123', or a PDF file name)."""
import re
from typing import Any, Dict, List, Optional

_PMID = re.compile(r'^\s*(?:PMID:?\s*)?(\d{5,9})(?!\d)', re.I)


def pmid_of(source: Optional[str]) -> Optional[str]:
    """The PMID a source tag names, or None. Only a tag that STARTS with it counts: a PDF called
    'Smith_20240101.pdf' contains digits but is not a PMID."""
    m = _PMID.match(str(source or ''))
    return m.group(1) if m else None


def record_sources(record: Dict[str, Any]) -> List[str]:
    """Every paper a reaction record draws on, first source first. A reaction from one paper has just its `source`; a
    reaction merged across papers also lists them in `sources`, and each quote may carry its own `source` too."""
    a = record.get('annotation_result', record)
    out = [record.get('source')] + list(a.get('sources') or [])
    out += [m.get('source') for m in (a.get('evidence_details') or {}).values() if isinstance(m, dict)]
    return list(dict.fromkeys(s for s in out if s))


def record_pmids(record: Dict[str, Any]) -> List[str]:
    """The PMIDs of those papers (a PDF has none), in order, without repeats."""
    return list(dict.fromkeys(p for p in (pmid_of(s) for s in record_sources(record)) if p))
