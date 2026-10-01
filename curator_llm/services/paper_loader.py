"""Load PaperText for an extraction `source` tag ('PMID:123', 'PMC123', or a PDF filename)."""
import logging
import os
import sys
from typing import Optional

from curator_llm.services.paper_text import PaperText

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_paper_text(source: str) -> Optional[PaperText]:
    """Best effort: returns None (quotes stay unverified and are reported) if the text is unreachable."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
    try:
        import PubMedFetcher as f
        src = (source or '').strip()
        if src.upper().startswith('PMID:'):
            pmid = src.split(':', 1)[1].strip()
            xml = f.read_fulltext_cache(pmid)
            if xml is None:
                pmcid = f.resolve_pmcids([pmid]).get(pmid)
                xml = f.fetch_jats(pmcid) if pmcid else None
            return PaperText.from_jats(pmid, xml) if xml else None
        if f.is_pmcid(src):
            return PaperText.from_jats(None, f.fetch_jats(src))
        path = src if os.path.isabs(src) else os.path.join(PROJECT_ROOT, 'data', 'papers', src)
        if os.path.isfile(path):
            return PaperText.from_pdf(None, path)
    except Exception as e:
        logger.warning('paper text unavailable for %s: %s', source, e)
    return None
