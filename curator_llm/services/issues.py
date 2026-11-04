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
        ('is a review article (pubmed', 'paper_is_review', 'info'),          # services/paper_kind.py
        ('could not read its publication type', 'paper_kind_unknown', 'info'),
        ('only cites from earlier work', 'cited_only_left_out', 'info'),   # services/extraction.py, _cited_note
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


# ── organizing the list for a curator ──────────────────────────────────────
# A draft gets hundreds of issues (291 on one 30-reaction paper), most of them on reactions that should not exist at
# all. So: decide first whether each such reaction stays ("keep"), list the rest grouped by kind, and hold back the
# issues of a reaction that waits on that decision (they are moot if it goes).
# "should this reaction stay": not a molecular reaction, or no quote shows it as a result of this paper
KEEP_CODES = frozenset({'qa_not_a_reaction', 'qa_only_cited', 'qa_negative_only', 'qa_hedged_only', 'qa_methods_only'})
_IDENTIFIER_PREFIXES = ('needs_resolution', 'uniprot', 'chebi', 'new_reference_entity', 'go_', 'compartment_',
                        'psimod_', 'duplicate_entities')
_EVIDENCE_CODES = frozenset({'qa_no_evidence', 'qa_only_assertion',
                             'qa_fuzzy_evidence', 'qa_non_human_evidence', 'qa_no_summation',
                             'qa_no_literature_reference', 'quote_rejected', 'qa_unsupported_catalyst',
                             'qa_unsupported_regulation', 'paper_no_pmid'})
_EVIDENCE_FIELDS = ('evidence', 'summation', 'pmids', 'literature')
CATEGORY_ORDER = ('keep', 'existing', 'content', 'identifiers', 'evidence', 'paper', 'other')


def categorize(i: Issue) -> str:
    c = i.code
    if c in KEEP_CODES:
        return 'keep'
    if c.startswith('existing') or i.source == 'existing':
        return 'existing'
    if c.startswith(_IDENTIFIER_PREFIXES) or (i.source == 'resolver' and c == 'note'):
        return 'identifiers'
    if c in _EVIDENCE_CODES or (c == 'qa_llm' and (i.field or '').split('[')[0] in _EVIDENCE_FIELDS):
        return 'evidence'
    if i.source == 'pipeline' or c.startswith('paper_') or c == 'cited_only_left_out':
        return 'paper'
    if c.startswith('qa_') or i.source in ('builder', 'emitter', 'checks'):
        return 'content'
    return 'other'


def _participant_of(message: str, names: Dict[str, str]) -> Optional[str]:
    """The participant a "NAME: ..." note is about (the resolver's and emitter's notes start that way): the longest
    participant name the message starts with, followed by a colon."""
    best = None
    for name, key in names.items():
        if name and message.startswith(name + ':') and (best is None or len(name) > len(best[0])):
            best = (name, key)
    return best[1] if best else None


def organize(issues: List[Issue], draft: Optional[ReactomeDraft], proposals: Iterable = ()) -> None:
    """Fill category, blocked_by and proposal_id on every issue (in place); link an entity note to its participant."""
    keep: Dict[str, str] = {}
    for i in issues:
        i.category = categorize(i)
        if i.category == 'keep' and i.status == 'open' and i.reaction_key and i.reaction_key not in keep:
            keep[i.reaction_key] = i.id
    used_by: Dict[str, set] = {}
    names: Dict[str, str] = {}
    if draft is not None:
        names = {p.name: k for k, p in draft.participants.items()}

        def within(k: str, seen: set):
            if k in seen or k not in draft.participants:
                return
            seen.add(k)
            p = draft.participants[k]
            for x in list(getattr(p, 'components', []) or []) + list(getattr(p, 'members', []) or []):
                within(x, seen)
        for r in draft.reactions:
            seen: set = set()
            for k in r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else []):
                within(k, seen)
            for k in seen:
                used_by.setdefault(k, set()).add(r.key)
    for i in issues:
        i.blocked_by = None
        if i.participant_key is None and i.reaction_key is None and names:
            i.participant_key = _participant_of(i.message, names)
        if i.category == 'keep':
            continue
        if i.code == 'qa_possible_duplicate':      # helps decide which of two look-alikes to keep: never held back
            continue
        if i.reaction_key:
            i.blocked_by = keep.get(i.reaction_key)
        elif i.participant_key and used_by.get(i.participant_key):
            rs = used_by[i.participant_key]
            if all(r in keep for r in rs):                  # every reaction using the entity waits on a decision
                i.blocked_by = keep[sorted(rs)[0]]
    # a pending suggested fix that resolves the issue: a reaction's removal resolves its keep issues
    fix: Dict[tuple, str] = {}
    for p in proposals:
        if getattr(p, 'actor', None) != 'suggestions' or getattr(p, 'status', None) != 'pending':
            continue
        for op in p.ops:
            path = op.get('path', '')
            parts = path.strip('/').split('/')
            if op.get('op') == 'remove' and len(parts) == 2 and parts[0] == 'reactions':
                fix[(parts[1], 'keep')] = p.id
            elif op.get('op') == 'remove' and len(parts) == 4 and parts[0] == 'reactions' and parts[2] == 'inputs':
                fix[(parts[1], 'qa_catalyst_is_consumed')] = p.id
            elif op.get('op') == 'replace' and len(parts) == 3 and parts[0] == 'reactions' and parts[2] == 'pmids':
                fix[(parts[1], 'qa_no_literature_reference')] = p.id
    for i in issues:
        i.proposal_id = (fix.get((i.reaction_key, 'keep')) if i.category == 'keep'
                         else fix.get((i.reaction_key, i.code))) if i.reaction_key else None
