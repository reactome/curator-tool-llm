"""Whole-paper text with section / page / figure tracking, so evidence can be located and checked.

Built per page from a PDF (PyMuPDF) or per <sec>/<fig> from PMC JATS XML. Text is kept as ONE
string; `locate(offset)` maps an offset back to its page, section and figure, and `char_span`
values in Evidence index into `text`.
"""
import bisect
import re
import xml.etree.ElementTree as et
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from curator_llm.models.evidence import Evidence
from curator_llm.services.quote_verifier import normalize_with_map, verify_quote

_FIG_REF = re.compile(r'\bFig(?:ure)?s?\.?\s*(S?\d+(?:[A-Za-z]|\s[A-Z](?![A-Za-z]))?)', re.I)
_FIG_LOOKAHEAD = 80   # a citation like "(Fig. 3 B)" follows the sentence it supports

_HEADING = re.compile(
    r'^[ \t]*(?:\d+[.)]?\s*)?(abstract|introduction|results(?:\s+and\s+discussion)?|discussion|'
    r'materials\s+and\s+methods|methods|experimental\s+procedures|conclusions?|'
    r'acknowledg(?:e?ments)?|references)[ \t]*$', re.I | re.M)


@dataclass
class Passage:
    text: str
    section: Optional[str]
    page: Optional[int]
    figure: Optional[str]
    char_span: Tuple[int, int]
    score: float = 0.0


@dataclass
class PaperText:
    pmid: Optional[str]
    text: str = ''
    page_starts: List[Tuple[int, int]] = field(default_factory=list)      # (offset, page number)
    section_starts: List[Tuple[int, str]] = field(default_factory=list)   # (offset, title)
    figure_spans: List[Tuple[int, int, str]] = field(default_factory=list)  # legends: (start, end, label)
    _norm: Optional[Tuple[str, List[int]]] = field(default=None, repr=False)

    # ── construction ───────────────────────────────────────────────────────
    @classmethod
    def from_pages(cls, pmid: Optional[str], pages: List[str], first_page: int = 1) -> 'PaperText':
        parts, starts, pos = [], [], 0
        for i, p in enumerate(pages):
            starts.append((pos, first_page + i))
            parts.append(p)
            pos += len(p) + 1                 # '\n' joiner
        text = '\n'.join(parts)
        pt = cls(pmid=pmid, text=text, page_starts=starts)
        pt.section_starts = [(m.start(), m.group(1).strip().title()) for m in _HEADING.finditer(text)]
        return pt

    @classmethod
    def from_pdf(cls, pmid: Optional[str], path: str, first_page: int = 1) -> 'PaperText':
        import pymupdf
        doc = pymupdf.open(path)
        try:
            pages = [p.get_text() for p in doc]
        finally:
            doc.close()
        return cls.from_pages(pmid, pages, first_page)

    @classmethod
    def from_jats(cls, pmid: Optional[str], xml: str) -> 'PaperText':
        """Abstract, body sections (by <title>) and figure legends (by <label>); page is unknown."""
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'reactome_llm'))
        from PubMedFetcher import _tag, _text_of          # reuse the project's JATS flattening
        root = et.fromstring(xml)
        parts: List[str] = []
        pt = cls(pmid=pmid)
        pos = 0

        def add(chunk: str, section: Optional[str] = None, figure: Optional[str] = None):
            nonlocal pos
            if not chunk:
                return
            if section is not None:
                pt.section_starts.append((pos, section))
            if figure is not None:
                pt.figure_spans.append((pos, pos + len(chunk), figure))
            parts.append(chunk)
            pos += len(chunk) + 2             # '\n\n' joiner

        for e in root.iter():
            if _tag(e) == 'abstract':
                add(_text_of(e), section='Abstract')
        body = next((e for e in root.iter() if _tag(e) == 'body'), None)
        if body is not None:
            def walk(sec, title):
                t = next((''.join(c.itertext()).strip() for c in sec if _tag(c) == 'title'), title)
                first = True
                for c in sec:
                    ct = _tag(c)
                    if ct == 'sec':
                        walk(c, t)
                    elif ct == 'fig':
                        label = next((''.join(x.itertext()).strip() for x in c if _tag(x) == 'label'), '')
                        m = _FIG_REF.search(label)
                        add(_text_of(c), section=t if first else None, figure=m.group(1).replace(' ', '') if m else None)
                        first = False
                    elif ct not in ('title', 'label'):
                        add(_text_of(c), section=t if first else None)
                        first = False
            for sec in body:
                if _tag(sec) == 'sec':
                    walk(sec, None)
                elif _tag(sec) == 'p':
                    add(_text_of(sec))
        pt.text = '\n\n'.join(parts)
        return pt

    # ── persistence ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {'pmid': self.pmid, 'text': self.text, 'page_starts': self.page_starts,
                'section_starts': self.section_starts, 'figure_spans': self.figure_spans}

    @classmethod
    def from_dict(cls, d: dict) -> 'PaperText':
        return cls(pmid=d.get('pmid'), text=d.get('text', ''),
                   page_starts=[tuple(x) for x in d.get('page_starts', [])],
                   section_starts=[tuple(x) for x in d.get('section_starts', [])],
                   figure_spans=[tuple(x) for x in d.get('figure_spans', [])])

    # ── lookup ─────────────────────────────────────────────────────────────
    def _at(self, starts, offset):
        i = bisect.bisect_right([s for s, _ in starts], offset) - 1
        return starts[i][1] if i >= 0 else None

    def locate(self, span: Tuple[int, int]) -> dict:
        start, end = span
        figure = next((lab for a, b, lab in self.figure_spans if a <= start < b), None)
        if figure is None:
            tail = self.text[start:min(len(self.text), end + _FIG_LOOKAHEAD)]
            m = _FIG_REF.search(tail)
            figure = m.group(1).replace(' ', '') if m else None
        return {'page': self._at(self.page_starts, start),
                'section': self._at(self.section_starts, start),
                'figure': figure}

    # ── verification ───────────────────────────────────────────────────────
    def _normalized(self):
        if self._norm is None:
            self._norm = normalize_with_map(self.text)
        return self._norm

    def verify(self, ev: Evidence) -> Evidence:
        """Return a copy of `ev` with code-filled location and verification. Never raises on a miss."""
        m = verify_quote(ev.quote, self.text, self._normalized())
        out = ev.model_copy(update={'verified': m.status, 'match_score': m.score or None,
                                    'pmid': ev.pmid or self.pmid})
        if m.span is not None:
            out = out.model_copy(update={'char_span': m.span, **self.locate(m.span)})
        return out

    # ── retrieval (chat's search_paper tool) ───────────────────────────────
    def _passages(self, window: int) -> List[Tuple[int, int]]:
        """Passage spans: whole lines grouped up to ~window chars, never crossing a section start."""
        bounds = {s for s, _ in self.section_starts}
        spans, start, pos = [], 0, 0
        for line in self.text.split('\n'):
            end = pos + len(line)
            if pos in bounds and pos > start:
                spans.append((start, pos))
                start = pos
            elif end - start > window and pos > start:
                spans.append((start, pos))
                start = pos
            pos = end + 1
        if start < len(self.text):
            spans.append((start, len(self.text)))
        return spans

    def search(self, query: str, section: Optional[str] = None, top_k: int = 5,
               window: int = 600) -> List[Passage]:
        """Keyword-overlap passage search, optionally limited to one section (e.g. 'Discussion')."""
        terms = {t for t in re.findall(r'[a-z0-9\-]+', query.lower()) if len(t) > 1}
        if not terms:
            return []
        hits: List[Passage] = []
        for a, b in self._passages(window):
            chunk = self.text[a:b]
            sec = self._at(self.section_starts, a)
            if section and (sec or '').lower() != section.lower():
                continue
            words = set(re.findall(r'[a-z0-9\-]+', chunk.lower()))
            score = len(terms & words) / len(terms)
            if score > 0:
                loc = self.locate((a, b))
                hits.append(Passage(chunk.strip(), sec, loc['page'], loc['figure'], (a, b), score))
        hits.sort(key=lambda p: -p.score)
        return hits[:top_k]
