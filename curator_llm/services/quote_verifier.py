"""Deterministic quote verification against the paper text (no LLM).

PDF text carries hyphenated line breaks, ligatures, curly quotes and irregular whitespace, so
both sides are normalised with an offset map back into the ORIGINAL text; that is how a match
yields a `char_span` that indexes the source text rather than the normalised copy.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from curator_llm.models.evidence import Verification

FUZZY_THRESHOLD = 90.0   # similarity (0-100) required for a FUZZY match
MIN_FUZZY_CHARS = 25     # shorter quotes must match exactly: fuzzy matching on a few words is too lenient

_QUOTE_MAP = {'‘': "'", '’': "'", '“': '"', '”': '"',
              '‐': '-', '‑': '-', '‒': '-', '–': '-', '−': '-'}
_ELLIPSIS = re.compile(r'\s*(?:\.\.\.|…|\[\s*(?:\.\.\.|…)\s*\])\s*')


@dataclass
class Match:
    status: Verification
    score: float = 0.0
    span: Optional[Tuple[int, int]] = None   # [start, end) in the original text


def normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Lower-case, collapse whitespace, de-hyphenate line breaks. Returns (norm, norm_idx -> orig_idx)."""
    out: List[str] = []
    idx: List[int] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in '-­' and i + 1 < n and text[i + 1] in '\r\n' and out and out[-1].isalpha():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j].isalpha():    # "phospho-\nrylation" -> "phosphorylation"
                i = j
                continue
        if ch in '-­':    # hyphens never decide a match: PDFs keep column-break hyphens
            i += 1                 # ("mitochon-dria") that a quote may or may not have copied
            continue
        for c in unicodedata.normalize('NFKC', _QUOTE_MAP.get(ch, ch)):
            if c.isspace():
                if out and out[-1] != ' ':
                    out.append(' ')
                    idx.append(i)
            else:
                out.append(c.lower())
                idx.append(i)
        i += 1
    while out and out[-1] == ' ':
        out.pop()
        idx.pop()
    return ''.join(out), idx


def _find_fragment(frag: str, norm: str, start_at: int = 0) -> Optional[Tuple[float, int, int, Verification]]:
    """Locate one normalised fragment in `norm` at or after start_at. Returns (score, s, e, status)."""
    if not frag:
        return None
    pos = norm.find(frag, start_at)
    if pos != -1:
        return 100.0, pos, pos + len(frag), Verification.EXACT
    if len(frag) < MIN_FUZZY_CHARS:
        return None
    window = norm[start_at:]
    al = fuzz.partial_ratio_alignment(frag, window)
    if al is not None and al.score >= FUZZY_THRESHOLD:
        return al.score, start_at + al.dest_start, start_at + al.dest_end, Verification.FUZZY
    return None


def verify_quote(quote: str, text: str, norm: Optional[Tuple[str, List[int]]] = None) -> Match:
    """Check that `quote` occurs in `text`. A quote with ellipses must match each piece in order."""
    norm_text, idx = norm if norm is not None else normalize_with_map(text)
    pieces = [normalize_with_map(p)[0] for p in _ELLIPSIS.split(quote) if p.strip()]
    if not pieces or not norm_text:
        return Match(Verification.FAILED)
    cursor, first_s, last_e = 0, None, 0
    worst, status = 100.0, Verification.EXACT
    for piece in pieces:
        hit = _find_fragment(piece, norm_text, cursor)
        if hit is None:
            return Match(Verification.FAILED)
        score, s, e, st = hit
        first_s = s if first_s is None else first_s
        last_e, cursor = e, e
        worst = min(worst, score)
        if st == Verification.FUZZY:
            status = Verification.FUZZY
    return Match(status, worst, (idx[first_s], idx[last_e - 1] + 1))
