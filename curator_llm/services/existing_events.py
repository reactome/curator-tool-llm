"""Does Reactome already have this reaction? (filter, then rank)

Filter: candidate events are the ones catalysed by the draft reaction's catalyst (or, with no catalyst,
sharing its participants), plus every event that already cites this paper's PMID.
Rank: because draft participants carry UniProt accessions, matching is on accessions, not names.

  similarity     Jaccard of the draft's substrate accessions against the event's inputs+outputs. Both directions
                 count, so a big generic event ("Polyubiquitination of substrate") cannot match a small draft
                 reaction just by containing its accessions. The catalyst is excluded from the draft side so that
                 sharing it alone cannot look like a match; a reaction whose only accession is its catalyst
                 (autophosphorylation) is measured on that accession.
  overlap        share of the draft's substrate accessions the event contains (shown for context)
  catalyst_match the catalysts share an accession (or neither side has one)
  cites_pmid     the existing event cites this paper

  same     catalyst_match and similarity >= 0.6
  similar  similarity >= 0.25, or the catalyst matches and the event cites this paper
"""
import logging
from typing import List, Optional, Set, Tuple

from curator_llm.models.reactome import ComplexSpec, DefinedSetSpec, EwasSpec, ReactomeDraft
from curator_llm.models.session import ExistingMatch, Issue
from curator_llm.ports.events import EventLookup, EventRow

logger = logging.getLogger(__name__)
SAME_SIMILARITY = 0.6
SIMILAR_SIMILARITY = 0.25


def _accessions(draft: ReactomeDraft, key: str, seen: Optional[Set[str]] = None) -> Set[str]:
    seen = seen if seen is not None else set()
    if key in seen or key not in draft.participants:
        return set()
    seen.add(key)
    p = draft.participants[key]
    if isinstance(p, EwasSpec):
        return {p.uniprot} if p.uniprot else set()
    refs = p.components if isinstance(p, ComplexSpec) else p.members if isinstance(p, DefinedSetSpec) else []
    out: Set[str] = set()
    for k in refs:
        out |= _accessions(draft, k, seen)
    return out


def reaction_accessions(draft: ReactomeDraft, reaction) -> Tuple[Set[str], Set[str]]:
    """(catalyst accessions, all participant accessions incl. catalyst; regulators are not part of identity)."""
    cats = _accessions(draft, reaction.catalyst.entity) if reaction.catalyst else set()
    allp = set(cats)
    for k in reaction.inputs + reaction.outputs:
        allp |= _accessions(draft, k)
    return cats, allp


def _score(cats: Set[str], allp: Set[str], ev: EventRow, cited: bool) -> Optional[ExistingMatch]:
    ev_cats = set(ev.catalysts)
    ev_core = (set(ev.inputs) | set(ev.outputs)) or ev_cats
    core = (allp - cats) or allp
    inter = core & ev_core
    union = core | ev_core
    similarity = len(inter) / len(union) if union else 0.0
    overlap = len(inter) / len(core) if core else 0.0
    cat_match = bool(cats & ev_cats) if (cats or ev_cats) else True
    if cat_match and similarity >= SAME_SIMILARITY:
        level = 'same'
    elif similarity >= SIMILAR_SIMILARITY or (cats & ev_cats and cited):
        level = 'similar'
    else:
        return None
    score = 0.6 * similarity + 0.3 * (1 if cat_match else 0) + 0.1 * (1 if cited else 0)
    reasons = [f'{len(inter)} of {len(core)} participant accessions shared (similarity {similarity:.2f})']
    if cats & ev_cats:
        reasons.append('same catalyst')
    if cited:
        reasons.append('cites this paper')
    return ExistingMatch(reaction_key='', db_id=ev.db_id, display_name=ev.display_name, st_id=ev.st_id,
                         level=level, score=round(score, 3), similarity=round(similarity, 3), cites_pmid=cited,
                         catalyst_match=cat_match, overlap=round(overlap, 3), reasons=reasons)


def find_existing(draft: ReactomeDraft, pmid: Optional[str], events: EventLookup,
                  only: Optional[List[str]] = None) -> Tuple[List[ExistingMatch], List[Issue]]:
    """Matches (best first per reaction, at most 3 each) and the issues to show the curator."""
    issues: List[Issue] = []
    try:
        citing = events.events_citing(pmid) if pmid else []
    except Exception as e:
        logger.warning('existing-event lookup failed: %s', e)
        return [], [Issue(source='existing', severity='warning', code='existing_check_unavailable',
                          message=f'could not check Reactome for existing events ({type(e).__name__})')]
    cited_ids = {e.db_id for e in citing}
    if citing and only is None:                       # paper-level, so only a full check reports it
        names = '; '.join(f'"{e.display_name}" ({e.st_id})' for e in citing[:5])
        issues.append(Issue(source='existing', severity='info', code='paper_already_curated',
                            message=f'Reactome already has {len(citing)} event(s) citing PMID {pmid}: {names}'))
    matches: List[ExistingMatch] = []
    for r in draft.reactions:
        if only is not None and r.key not in only:
            continue
        cats, allp = reaction_accessions(draft, r)
        if not allp:
            issues.append(Issue(source='existing', severity='info', code='existing_check_skipped', reaction_key=r.key,
                                message=f'"{r.name}": no participant has a UniProt accession, so it cannot be '
                                        f'compared with existing reactions'))
            continue
        try:
            cands = {e.db_id: e for e in events.candidate_events(sorted(cats), sorted(allp))}
        except Exception as e:
            logger.warning('candidate lookup failed for %s: %s', r.key, e)
            issues.append(Issue(source='existing', severity='warning', code='existing_check_unavailable', reaction_key=r.key,
                                message=f'could not look up existing events for "{r.name}"'))
            continue
        for e in citing:
            cands.setdefault(e.db_id, e)
        scored = []
        for e in cands.values():
            m = _score(cats, allp, e, e.db_id in cited_ids)
            if m:
                m.reaction_key = r.key
                scored.append(m)
        scored.sort(key=lambda m: (m.level != 'same', -m.score))
        matches += scored[:3]
        for m in scored[:3]:
            tag = ' (cites this paper)' if m.cites_pmid else ''
            if m.level == 'same':
                issues.append(Issue(source='existing', severity='action', code='existing_reaction_match', reaction_key=r.key,
                                    message=f'"{r.name}" appears to already exist in Reactome as "{m.display_name}" '
                                            f'({m.st_id}){tag}; consider reusing it instead of adding a new reaction'))
            else:
                issues.append(Issue(source='existing', severity='info', code='existing_reaction_similar', reaction_key=r.key,
                                    message=f'"{r.name}" is similar to the existing "{m.display_name}" ({m.st_id}){tag}'))
    return matches, issues
