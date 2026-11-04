"""Keep the reactions a research paper itself shows; leave out the ones it only cites.

Reactome annotates from direct evidence: a reaction drawn from a research paper must rest on that paper's own results,
not on a sentence that summarises earlier work ("WEE1 phosphorylates CDK1 on Tyr15 (refs. 1-3)"). The extractor tags
every quote with `claim_origin` ("this_paper" or "cited"), so the test is done here in code, not left to the prompt.
A review article is different: what it reports IS cited work, so nothing is left out for a review.

A reaction is cited-only when it has quotes and EVERY one is "cited". A reaction with at least one quote showing the
paper's own result stays, whatever else it cites. A reaction with no quote metadata at all stays too (fail open): it
is the quote verifier's job to deal with missing quotes, not this filter's.
"""
from typing import Iterable, List, Tuple


def evidence_origins(annotation: dict) -> List[str]:
    """The claim_origin of each of a reaction's quotes ('this_paper' when a quote does not say)."""
    details = annotation.get('evidence_details') or {}
    if details:
        return [(meta or {}).get('claim_origin') or 'this_paper' for meta in details.values()]
    out = []
    for ev in annotation.get('evidence') or []:          # the extractor's raw shape, before split_evidence
        if isinstance(ev, dict):
            out.append(ev.get('claim_origin') or 'this_paper')
    return out


def is_cited_only(annotation: dict) -> bool:
    origins = evidence_origins(annotation)
    return bool(origins) and all(o == 'cited' for o in origins)


def split_cited_only(records: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    """(kept, left_out) for extraction records ({'annotation_result': ...} or bare reactions), order kept."""
    kept, left_out = [], []
    for r in records:
        (left_out if is_cited_only(r.get('annotation_result', r)) else kept).append(r)
    return kept, left_out
