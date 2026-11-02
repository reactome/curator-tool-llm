"""Suggested fixes are pending proposals: made by the service, applied only when a curator accepts them."""
import pytest

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import CatalystSpec, EwasSpec, ReactionSpec, ReactomeDraft, SimpleEntitySpec
from curator_llm.models.session import Session
from curator_llm.services.session_service import ProposalError, SessionService
from curator_llm.services.suggestions import suggest_fixes
from tests.fakes.ports import SyncJobRunner


def draft(inputs=('ub', 'pink1'), outputs=('pub',), pmids=()):
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'ub': EwasSpec(key='ub', name='UB', uniprot='P0CG47'),
                      'pub': EwasSpec(key='pub', name='pUB', uniprot='P0CG47'), 'atp': SimpleEntitySpec(key='atp', name='ATP')}
    d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates UB', inputs=list(inputs), outputs=list(outputs),
                                catalyst=CatalystSpec(entity='pink1'), pmids=list(pmids), evidence_ids=['e1'])]
    return d


EV = [Evidence(id='e1', quote='q', source='PMID:24751536')]


def service(d=None):
    store = InMemorySessionStore()
    s = Session(owner='o', draft=d or draft(), evidence=list(EV), status='ready')
    SessionService.refresh(s)
    store.save(s)
    return SessionService(store, SyncJobRunner(), None), s.id


def test_an_enzyme_listed_as_an_input_is_a_suggestion_to_remove_it_and_the_draft_is_untouched():
    svc, sid = service()
    pending = svc.suggest_fixes('o', sid)
    enzyme = [p for p in pending if 'consumed' in p.reason]
    assert len(enzyme) == 1 and enzyme[0].actor == 'suggestions' and enzyme[0].ops == [{'op': 'remove', 'path': '/reactions/r0/inputs/1'}]
    assert svc.store.get(sid, 'o').draft.reactions[0].inputs == ['ub', 'pink1']        # nothing applied


def test_accepting_it_removes_the_enzyme_keeps_the_catalyst_and_clears_the_issue():
    svc, sid = service()
    p = next(x for x in svc.suggest_fixes('o', sid) if 'consumed' in x.reason)
    s = svc.accept('o', sid, p.id)
    r = s.draft.reactions[0]
    assert r.inputs == ['ub'] and r.catalyst.entity == 'pink1'
    assert not [i for i in s.issues if i.code == 'qa_catalyst_is_consumed']
    assert not [x for x in s.proposals if x.status == 'pending' and 'consumed' in x.reason]


def test_a_rejected_suggestion_is_not_made_again():
    svc, sid = service()
    p = next(x for x in svc.suggest_fixes('o', sid) if 'consumed' in x.reason)
    svc.reject('o', sid, p.id)
    assert not [x for x in svc.suggest_fixes('o', sid) if 'consumed' in x.reason]
    assert len([x for x in svc.store.get(sid, 'o').proposals if 'consumed' in x.reason]) == 1


def test_a_pending_suggestion_that_no_longer_applies_is_dropped():
    svc, sid = service()
    svc.suggest_fixes('o', sid)
    s = svc.store.get(sid, 'o')
    s.draft.reactions[0].inputs = ['ub']                # the curator fixed it by hand
    svc.store.save(s)
    assert not [x for x in svc.suggest_fixes('o', sid) if 'consumed' in x.reason]
    assert not [x for x in svc.store.get(sid, 'o').proposals if x.status == 'pending' and 'consumed' in x.reason]


def test_no_enzyme_suggestion_when_it_is_also_an_output_or_when_it_would_leave_no_input():
    assert not [s for s in suggest_fixes(draft(outputs=('pub', 'pink1')), EV) if s.code == 'enzyme_as_input']
    assert not [s for s in suggest_fixes(draft(inputs=('pink1',)), EV) if s.code == 'enzyme_as_input']


def test_a_reaction_with_no_pmid_is_offered_the_papers_its_quotes_come_from():
    got = [s for s in suggest_fixes(draft(inputs=('ub',)), EV) if s.code == 'missing_pmids']
    assert len(got) == 1 and got[0].ops == [{'op': 'replace', 'path': '/reactions/r0/pmids', 'value': ['24751536']}]
    assert not suggest_fixes(draft(inputs=('ub',), pmids=['1']), EV)                   # already cited: nothing to suggest
    assert not suggest_fixes(draft(inputs=('ub',)), [Evidence(id='e1', quote='q', source='uploaded.pdf')])   # a PDF has no PMID


def test_accepting_the_pmid_suggestion_cites_the_paper():
    svc, sid = service(draft(inputs=('ub',)))
    p = next(x for x in svc.suggest_fixes('o', sid) if x.ops[0]['path'].endswith('/pmids'))
    assert svc.accept('o', sid, p.id).draft.reactions[0].pmids == ['24751536']


def test_suggestions_are_not_made_for_a_session_that_is_not_ready():
    svc = SessionService(InMemorySessionStore(), SyncJobRunner(), None)
    s = Session(owner='o')
    svc.store.save(s)
    with pytest.raises(ProposalError):
        svc.suggest_fixes('o', s.id)
