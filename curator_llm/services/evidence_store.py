"""Per-session evidence registry: stable ids, dedup by quote, verification, and no truncation."""
from typing import Dict, List, Optional, Tuple

from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.services.paper_text import PaperText
from curator_llm.services.quote_verifier import normalize_with_map


class EvidenceStore:
    def __init__(self, paper: Optional[PaperText] = None):
        self.paper = paper
        self._by_id: Dict[str, Evidence] = {}
        self._by_quote: Dict[str, str] = {}

    def __len__(self):
        return len(self._by_id)

    def get(self, ev_id: str) -> Evidence:
        return self._by_id[ev_id]

    def all(self) -> List[Evidence]:
        return list(self._by_id.values())

    def add(self, draft: Evidence, paper: Optional[PaperText] = None) -> Tuple[Optional[Evidence], bool]:
        """Verify and register a draft. Returns (evidence, kept).

        A quote that fails verification is NOT stored unless it is an explicit curator_assertion;
        callers get (failed_copy, False) so they can drop the field it was meant to support.
        The same quote added twice returns the existing id with `supports` merged.
        """
        ev = draft
        paper = paper or self.paper
        if paper is not None and draft.claim_origin != ClaimOrigin.CURATOR_ASSERTION:
            ev = paper.verify(draft)
            if ev.verified == Verification.FAILED:
                return ev, False
        key = (ev.pmid or '') + '|' + normalize_with_map(ev.quote)[0]
        if key in self._by_quote:
            existing = self._by_id[self._by_quote[key]]
            merged = list(dict.fromkeys(existing.supports + ev.supports))
            existing = existing.model_copy(update={'supports': merged})
            self._by_id[existing.id] = existing
            return existing, True
        n = max([int(i.split('-')[1]) for i in self._by_id if i.startswith('ev-') and i.split('-')[1].isdigit()] or [0]) + 1
        ev = ev.model_copy(update={'id': f'ev-{n:03d}'})
        self._by_id[ev.id] = ev
        self._by_quote[key] = ev.id
        return ev, True

    @classmethod
    def load(cls, evidence: List[Evidence], paper: Optional[PaperText] = None) -> 'EvidenceStore':
        """Rebuild a store from saved evidence so new ids continue the numbering."""
        st = cls(paper)
        for e in evidence:
            st._by_id[e.id] = e
            st._by_quote[(e.pmid or '') + '|' + normalize_with_map(e.quote)[0]] = e.id
        return st

    def to_json(self) -> List[dict]:
        return [e.model_dump(mode='json') for e in self._by_id.values()]
