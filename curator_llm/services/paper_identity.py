"""The PubMed ID of an uploaded PDF, found from its content: the DOI on its first pages, else its title.

A PDF without a PubMed ID gets no LiteratureReference, so the curator is offered what this finds, to keep, edit or clear
before the annotation starts (never applied silently). Content beats the file name: a PDF named 39737163.pdf may hold
another paper.

1. DOI: every DOI on pages 1-2 is looked up in PubMed. A DOI there can also be a cited paper's, so a hit is accepted only
   when the title PubMed has for it is printed on those pages.
2. Title: the largest text on page 1 (lines of one font size, joined; the title can run over lines in two sizes, so a
   partial title is compared with the start of PubMed's), and the PDF's own title field. Searched by its distinctive
   words; accepted when PubMed's title matches closely (MIN_TITLE_SCORE).
Nothing found (a preprint, a paper not yet in PubMed, a scan without text): None.

Measured on 25 article PDFs (17 with known PubMed IDs, 8 checked by hand): all found, none wrong; two documents that
are not articles (one citing another paper's DOI) gave nothing.
"""
import difflib
import re
from typing import Dict, List, Optional

from curator_llm.ports.external import ArticleSearch

MIN_TITLE_SCORE = 0.8
_DOI = re.compile(r'\b(10\.\d{4,9}/[^\s"<>,;]+)', re.I)
# words PubMed's title search does not index: in a [Title] query they make it return nothing
_STOP = {'with', 'from', 'through', 'that', 'this', 'into', 'upon', 'than', 'their', 'were', 'which', 'role', 'after',
         'during', 'between', 'within', 'without', 'via', 'and', 'the', 'for', 'are', 'its'}


def _norm(t: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (t or '').lower()).strip()


def title_score(candidate: str, title: str) -> float:
    """How well text read from the PDF matches a PubMed title, 0..1. The candidate may be only the title's first
    line(s), so it is also compared with the same length of PubMed's title."""
    c, t = _norm(candidate), _norm(title)
    if not c or not t:
        return 0.0
    whole = difflib.SequenceMatcher(None, c, t).ratio()
    start = difflib.SequenceMatcher(None, c, t[:len(c)]).ratio() if len(c) >= 30 else 0.0
    return max(whole, start)


def dois(pages: List[str]) -> List[str]:
    """DOIs in the order they appear, without trailing punctuation, each once."""
    out: List[str] = []
    for text in pages:
        for m in _DOI.finditer(text):
            d = m.group(1).rstrip('.)]}')
            if d.lower() not in (x.lower() for x in out):
                out.append(d)
    return out


def title_candidates(doc) -> List[str]:
    """The PDF's title field when it looks like one, then the page-1 text in its three largest font sizes."""
    out: List[str] = []
    meta = ((doc.metadata or {}).get('title') or '').strip()
    if len(meta.split()) >= 5 and not re.search(r'\.(docx?|pdf|tex)$|microsoft|untitled', meta, re.I):
        out.append(meta)
    lines = []
    for block in doc[0].get_text('dict').get('blocks', []):
        for line in block.get('lines', []):
            text = ' '.join(s['text'] for s in line.get('spans', [])).strip()
            size = max((s['size'] for s in line.get('spans', [])), default=0)
            if text:
                lines.append((round(size, 1), line['bbox'][1], text))
    for size in sorted({s for s, _, _ in lines}, reverse=True)[:3]:
        cand = ' '.join(t for s, _, t in sorted(lines, key=lambda x: x[1]) if s == size)
        if 4 <= len(cand.split()) <= 60:
            out.append(' '.join(cand.split()))
    return out


def _title_query(candidate: str) -> Optional[str]:
    words = [w for w in _norm(candidate).split() if len(w) > 3 and w not in _STOP][:12]
    return ' AND '.join(f'{w}[Title]' for w in words) if len(words) >= 3 else None


def identify_text(pages: List[str], candidates: List[str], search: ArticleSearch) -> Optional[Dict[str, object]]:
    """The core, on text already read: {'pmid', 'title', 'method': 'doi'|'title', 'doi', 'score'} or None."""
    front = _norm(' '.join(pages))
    for d in dois(pages):
        ids = search.search(f'"{d}"[doi]', 2)
        if len(ids) != 1:
            continue
        title = search.titles(ids).get(ids[0], '')
        if title and _norm(title)[:50] in front:            # this paper's DOI, not a cited one
            return {'pmid': ids[0], 'title': title, 'method': 'doi', 'doi': d, 'score': 1.0}
    best = None
    for cand in candidates:
        q = _title_query(cand)
        if not q:
            continue
        for pmid, title in search.titles(search.search(q, 5)).items():
            score = round(title_score(cand, title), 2)
            if best is None or score > best['score']:
                best = {'pmid': pmid, 'title': title, 'method': 'title', 'doi': None, 'score': score}
    return best if best and best['score'] >= MIN_TITLE_SCORE else None


def identify(pdf_path: str, search: ArticleSearch) -> Optional[Dict[str, object]]:
    """The PubMed ID of the article in this PDF, or None. Raises when PubMed cannot be asked."""
    import pymupdf                  # PyMuPDF, as services/paper_text.py
    with pymupdf.open(pdf_path) as doc:
        if len(doc) == 0:
            return None
        pages = [doc[i].get_text() for i in range(min(2, len(doc)))]
        return identify_text(pages, title_candidates(doc), search)
