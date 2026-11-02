"""Suggested fixes for problems the checks find. A suggestion is only ever held as a pending proposal: the curator reads
it and accepts or rejects it, and nothing in the draft changes before that. Only fixes with one clear answer are made;
a problem that needs judgement (an invented state, two related reactions that might be one, evidence from another
species) is left to the curator.
"""
from dataclasses import dataclass
from typing import Dict, List

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import ReactomeDraft


@dataclass
class Suggestion:
    reaction_key: str
    code: str               # the kind of fix: 'enzyme_as_input' | 'missing_pmids'
    reason: str
    ops: List[dict]         # RFC 6902 patch against the draft


def _enzyme_as_input(draft: ReactomeDraft) -> List[Suggestion]:
    """A reaction whose catalyst is also an input and not an output consumes its own enzyme. Take it out of the inputs
    (the catalyst stays). Not suggested when that would leave the reaction with no input."""
    out = []
    for r in draft.reactions:
        if not r.catalyst or r.catalyst.entity not in r.inputs or r.catalyst.entity in r.outputs:
            continue
        rest = [k for k in r.inputs if k != r.catalyst.entity]
        if not rest:
            continue
        name = draft.participants[r.catalyst.entity].name
        ops = [{'op': 'remove', 'path': f'/reactions/{r.key}/inputs/{i}'}
               for i in sorted((i for i, k in enumerate(r.inputs) if k == r.catalyst.entity), reverse=True)]
        out.append(Suggestion(r.key, 'enzyme_as_input', f'"{name}" is the catalyst of "{r.name}", so it is not consumed: '
                              f'remove it from the inputs (it stays as the catalyst)', ops))
    return out


def _missing_pmids(draft: ReactomeDraft, evidence: Dict[str, Evidence]) -> List[Suggestion]:
    """A reaction with no literature reference whose quotes come from PubMed papers: cite those papers."""
    out = []
    for r in draft.reactions:
        if r.pmids:
            continue
        pmids = list(dict.fromkeys(e.source.split(':', 1)[1] for i in r.evidence_ids if (e := evidence.get(i))
                                   and e.source and e.source.startswith('PMID:')))
        if pmids:
            out.append(Suggestion(r.key, 'missing_pmids', f'"{r.name}" has no literature reference, and its quotes come '
                                  f'from {", ".join("PMID " + p for p in pmids)}: cite {"it" if len(pmids) == 1 else "them"}',
                                  [{'op': 'replace', 'path': f'/reactions/{r.key}/pmids', 'value': pmids}]))
    return out


def suggest_fixes(draft: ReactomeDraft, evidence: List[Evidence]) -> List[Suggestion]:
    by_id = {e.id: e for e in evidence}
    return _enzyme_as_input(draft) + _missing_pmids(draft, by_id)
