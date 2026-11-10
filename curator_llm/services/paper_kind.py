"""Is each paper a review article or a research paper? That decides whether reactions it only cites are kept
(services/direct_evidence.py).

  - set explicitly (a ticked PDF, `review` in the request, --review): as set;
  - a PubMed ID: PubMed's publication types decide, which matched every paper of the ground-truth set (PMC's own
    article-type did not: it calls review 23256519 a research article);
  - a PDF: research. A PDF carries no reliable metadata, so the curator ticks the ones that are reviews.
When PubMed cannot be asked, the paper is treated as research and a note says so.

`KindPipeline` does this BEFORE the pipeline it wraps (the saved-result layer keys on it), and hands the decision back
in `PipelineResult.reviews` so the session shows it.
"""
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from curator_llm.ports.external import PublicationTypes
from curator_llm.ports.pipeline import PaperRef, PipelineInput, PipelineResult
from curator_llm.services import issues as iss

# PubMed publication types that make a paper a review: it reports others' results
REVIEW_TYPES = frozenset({'Review', 'Systematic Review', 'Scoping Review', 'Meta-Analysis'})


def decide(spec: PipelineInput, pubmed: Optional[PublicationTypes]) -> Tuple[PipelineInput, List[str]]:
    """(the input with every paper's `review` set to True or False, notes for the curator)."""
    refs = spec.refs()
    ask = [r.pmid or r.pdf_pmid for r in refs if r.review is None and (r.pmid or r.pdf_pmid)]   # a PDF's PubMed ID counts
    types, failed = {}, None
    if ask:
        if pubmed is None:
            failed = 'no PubMed client configured'
        else:
            try:
                types = pubmed.publication_types(ask)
            except Exception as e:                       # network, HTTP error, bad JSON: never fail the annotation
                failed = f'{type(e).__name__}: {e}'
    notes: List[str] = []
    decided: List[PaperRef] = []
    for r in refs:
        review = r.review
        pmid = r.pmid or r.pdf_pmid
        if review is None and pmid:
            if failed or pmid not in types:
                review = False
                notes.append(f'PMID {pmid}: could not read its publication type from PubMed '
                             f'({failed or "PubMed has no record of it"}); treated as a research paper. If it is a '
                             'review, annotate it again marked as a review.')
            else:
                review = bool(REVIEW_TYPES.intersection(types[pmid]))
                if review:
                    found = ', '.join(t for t in types[pmid] if t in REVIEW_TYPES)
                    notes.append(f'PMID {pmid} is a review article (PubMed publication type: {found}): the reactions '
                                 'it only cites from earlier work are kept.')
        decided.append(replace(r, review=bool(review)))
    if spec.papers:
        out = replace(spec, papers=decided, review=None)
    else:                                                  # one paper: keep the single-paper shape
        out = replace(spec, review=decided[0].review if decided else spec.review)
    return out, notes


class KindPipeline:
    """Decides review or research for every paper, then runs the wrapped pipeline (the saved-result layer, which keys
    on the decision)."""

    def __init__(self, inner, pubmed: Optional[PublicationTypes]):
        self.inner, self.pubmed = inner, pubmed

    def run(self, spec: PipelineInput, report: Callable[[str], None]) -> PipelineResult:
        spec, notes = decide(spec, self.pubmed)
        result = self.inner.run(spec, report)
        result.reviews = {r.source: bool(r.review) for r in spec.refs()}
        if notes:
            result.issues = list(result.issues) + iss.issues_from_notes('pipeline', notes)
        return result
