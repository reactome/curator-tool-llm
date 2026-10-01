"""Turn the pipeline's notes and flags into structured Issues for the frontend."""
from typing import Dict, Iterable, List, Optional

from curator_llm.models.reactome import ReactomeDraft
from curator_llm.models.session import Issue

_ACTION_FLAGS = {'uniprot', 'uniprot-mismatch', 'compartment', 'chebi'}


def _classify(source: str, msg: str):
    """(code, severity) from a note's wording. Notes are produced by our own code, so this is a
    closed set of patterns; anything unrecognised stays a generic warning."""
    m = msg.lower()
    rules = [
        ('unavailable', 'service_unavailable', 'warning'),
        ('does not match', 'uniprot_mismatch', 'action'),
        ('does not exist or is obsolete', 'uniprot_invalid', 'action'),
        ('found by gene-name search', 'uniprot_suggested', 'info'),
        ('suggested by label search', 'chebi_suggested', 'info'),
        ('not found in gk_central', 'compartment_unresolved', 'action'),
        ('not resolved to a compartment', 'compartment_unresolved', 'action'),
        ('go function', 'go_unresolved', 'action'),
        ('new goterm', 'go_new', 'action'),
        ('new go_molecularfunction', 'go_new', 'action'),
        ('psi-mod', 'psimod_unresolved', 'action'),
        ('no uniprot accession', 'uniprot_missing', 'action'),
        ('no chebi id', 'chebi_missing', 'action'),
        ('new referencegeneproduct', 'new_reference_entity', 'action'),
        ('new referencemolecule', 'new_reference_entity', 'action'),
        ('already in reactome', 'existing_reaction', 'action'),
        ('add the new reactions to the existing pathway', 'existing_pathway', 'action'),
        ('defined identically', 'duplicate_entities', 'action'),
        ('not in the draft', 'reaction_missing', 'action'),
        ('dropped', 'dropped_by_builder', 'warning'),
        ('removed', 'dropped_by_builder', 'warning'),
        ('not found in the paper', 'quote_rejected', 'warning'),
    ]
    for needle, code, sev in rules:
        if needle in m:
            return code, sev
    return 'note', 'warning'


def issues_from_notes(source: str, notes: Iterable[str]) -> List[Issue]:
    out = []
    for n in notes:
        code, sev = _classify(source, n)
        out.append(Issue(source=source, severity=sev, code=code, message=n))
    return out


def issues_from_draft(draft: ReactomeDraft) -> List[Issue]:
    """One issue per unresolved flag on a participant (the frontend can jump to that entity)."""
    out = []
    for key, p in draft.participants.items():
        for flag in dict.fromkeys(p.needs_resolution):
            if flag in _ACTION_FLAGS or flag == 'uniprot-unchecked':
                out.append(Issue(source='resolver', severity='action' if flag in _ACTION_FLAGS else 'warning',
                                 code=f'needs_resolution:{flag}', participant_key=key,
                                 message=f'{p.name}: {flag.replace("-", " ")} needs a curator'))
    return out


def rejected_quote_issues(rejected_by_reaction: Dict[str, List[str]]) -> List[Issue]:
    return [Issue(source='evidence', severity='warning', code='quote_rejected', reaction_key=rk,
                  message=f'quote not found in the paper and dropped: "{q[:120]}"')
            for rk, qs in rejected_by_reaction.items() for q in qs]


def assign_ids(issues: List[Issue]) -> List[Issue]:
    """Give ids to issues that lack one, continuing after the highest existing id. Ids already
    assigned never change, so a curator's reference to an issue survives later edits."""
    top = max([int(i.id.split('-')[1]) for i in issues if i.id.startswith('iss-') and i.id.split('-')[1].isdigit()] or [0])
    for i in issues:
        if not i.id:
            top += 1
            i.id = f'iss-{top:03d}'
    return issues


def link_to_instances(issues: List[Issue], key_to_db_id: Dict[str, int],
                      participant_db_ids: Optional[Dict[str, int]] = None) -> List[Issue]:
    """Attach the emitted dbId of the reaction/participant an issue is about."""
    for i in issues:
        if i.reaction_key and i.reaction_key in key_to_db_id:
            i.instance_db_id = key_to_db_id[i.reaction_key]
        elif i.participant_key and participant_db_ids and i.participant_key in participant_db_ids:
            i.instance_db_id = participant_db_ids[i.participant_key]
    return issues
