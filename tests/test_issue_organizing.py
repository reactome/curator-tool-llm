"""Organizing the issue list: decide first whether a reaction stays, then the rest by kind, with the issues of a
reaction awaiting that decision held back, and one issue for a fact that holds for the whole paper."""
import json

from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import ClaimOrigin, Evidence
from curator_llm.models.reactome import CatalystSpec, EwasSpec, PathwaySpec, ReactionSpec, ReactomeDraft
from curator_llm.models.session import Issue, PaperInfo, Session
from curator_llm.ports.chat_model import ModelTurn
from curator_llm.ports.pipeline import PipelineResult
from curator_llm.services.issues import categorize, organize
from curator_llm.services.qa import parse_llm_review, qa_reaction
from curator_llm.services.session_service import SessionService
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner
from tests.test_session_api import AUTH, TOKENS

OWN, CITED = ClaimOrigin.THIS_PAPER, ClaimOrigin.CITED


def draft():
    d = ReactomeDraft()
    d.participants = {k: EwasSpec(key=k, name=n) for k, n in
                      (('wee1', 'WEE1'), ('cdh1', 'CDH1'), ('pcdh1', 'CDH1 (phosphorylated)'), ('cdk1', 'CDK1'),
                       ('pcdk1', 'CDK1 (phospho-Tyr15)'), ('h', 'APC1'))}
    d.reactions = [
        ReactionSpec(key='r0', name='WEE1 phosphorylates CDH1', inputs=['cdh1'], outputs=['pcdh1'],
                     catalyst=CatalystSpec(entity='wee1'), summation='s', evidence_ids=['e0']),
        ReactionSpec(key='r1', name='WEE1 phosphorylates CDK1', inputs=['cdk1'], outputs=['pcdk1'],
                     catalyst=CatalystSpec(entity='wee1'), summation='s', evidence_ids=['e1']),
        ReactionSpec(key='r2', name='CDH1 Tyr91 hydrogen bonds APC1', inputs=['h'], outputs=['h'], summation='s',
                     evidence_ids=['e2'])]
    d.pathway = PathwaySpec(name='p', reactions=['r0', 'r1', 'r2'])
    return d


EV = [Evidence(id='e0', quote='WEE1 phosphorylated tyrosines in CDH1 (Fig. 4f).', claim_origin=OWN, source='p.pdf'),
      Evidence(id='e1', quote='WEE1 phosphorylates CDK1 on Tyr15 (refs. 1-3).', claim_origin=CITED, source='p.pdf'),
      Evidence(id='e2', quote='Tyr91 formed a hydrogen bond with His1231 in APC1 (Fig. 4g).', claim_origin=OWN, source='p.pdf')]


def session(**kw):
    return Session(owner='o', draft=draft(), evidence=list(EV), status='ready', papers=[PaperInfo(source='p.pdf', kind='pdf')], **kw)


# ── the model says whether it is a reaction ────────────────────────────────
class Model:
    model = 'fake'

    def __init__(self, replies):
        self.replies = replies

    def turn(self, system, messages, tools, on_text):
        key = json.loads(messages[0]['content'])['reaction']['key']
        return ModelTurn(json.dumps(self.replies.get(key, {'is_reaction': True, 'verdict': 'ok', 'score': 0.9,
                                                          'summary': 'No problems found', 'findings': []})))


NOT_A_REACTION = {'r2': {'is_reaction': False, 'not_reaction_reason': 'a structural contact inside the APC/C complex',
                         'verdict': 'needs_work', 'score': 0.2, 'summary': None,
                         'findings': [{'severity': 'action', 'message': 'the catalyst is missing', 'field': 'catalyst'}]}}


def test_the_review_answers_is_this_a_reaction_and_then_says_nothing_else():
    assert parse_llm_review(json.dumps({'is_reaction': False, 'not_reaction_reason': 'a phenotype'}))['not_reaction_reason'] == 'a phenotype'
    assert parse_llm_review(json.dumps({'is_reaction': 'no'}))['is_reaction'] is None           # not a boolean: unknown
    res = qa_reaction(draft(), 'r2', EV, Model(NOT_A_REACTION))
    assert res.is_reaction is False and res.priority == 'fix_first' and res.summary.startswith('a structural contact')
    assert not [f for f in res.findings if f.source == 'llm']                                  # the catalyst finding is moot


def test_qa_results_are_kept_and_not_a_reaction_becomes_a_decision_with_a_remove_fix():
    s = session()
    SessionService.refresh(s)
    SessionService.qa_into(s, Model(NOT_A_REACTION))
    SessionService(None, None, None).refresh_suggestions(s)
    assert s.qa['r2'].is_reaction is False and s.qa['r0'].is_reaction is True and s.qa['r0'].priority != 'fix_first'
    keep = {i.reaction_key: i for i in s.issues if i.category == 'keep'}
    assert set(keep) == {'r1', 'r2'}                                    # r1: cited only; r2: not a reaction
    assert keep['r2'].code == 'qa_not_a_reaction' and 'structural contact' in keep['r2'].message
    assert all(i.proposal_id for i in keep.values())                     # each with a pending "remove it"


def test_issues_of_a_reaction_awaiting_the_decision_are_held_back_until_it_is_kept():
    s = session()
    s.issues = [Issue(id='k', source='checks', code='qa_only_cited', message='cited', reaction_key='r1'),
                Issue(id='a', source='qa', code='qa_llm', message='summation overstates', reaction_key='r1', field='summation'),
                Issue(id='b', source='qa', code='qa_llm', message='wrong catalyst', reaction_key='r0', field='catalyst'),
                Issue(id='c', source='emitter', code='uniprot_missing', message='CDK1 (phospho-Tyr15): no UniProt accession'),
                Issue(id='d', source='emitter', code='uniprot_missing', message='WEE1: no UniProt accession')]
    organize(s.issues, s.draft)
    by = {i.id: i for i in s.issues}
    assert [by[x].category for x in 'kabcd'] == ['keep', 'evidence', 'content', 'identifiers', 'identifiers']
    assert by['a'].blocked_by == 'k' and by['b'].blocked_by is None
    assert by['c'].participant_key == 'pcdk1' and by['c'].blocked_by == 'k'   # only r1 uses it
    assert by['d'].participant_key == 'wee1' and by['d'].blocked_by is None    # r0 uses WEE1 too
    by['k'].status = 'dismissed'                                              # the curator keeps r1
    organize(s.issues, s.draft)
    assert by['a'].blocked_by is None and by['c'].blocked_by is None


def test_removing_a_reaction_takes_its_issues_and_qa_with_it():
    store = InMemorySessionStore()
    svc = SessionService(store, SyncJobRunner(), None)
    s = session()
    SessionService.refresh(s)
    svc.refresh_suggestions(s)
    store.save(s)
    fix = next(i.proposal_id for i in s.issues if i.reaction_key == 'r1' and i.category == 'keep')
    after = svc.accept('o', s.id, fix)
    assert [r.key for r in after.draft.reactions] == ['r0', 'r2']
    assert not [i for i in after.issues if i.reaction_key == 'r1']


def test_a_paper_with_no_pmid_is_one_issue_not_one_per_reaction():
    s = session()
    SessionService.refresh(s)
    SessionService.qa_into(s, Model({}))
    assert [i.code for i in s.issues].count('paper_no_pmid') == 1
    assert not [i for i in s.issues if i.code == 'qa_no_literature_reference']
    assert categorize(next(i for i in s.issues if i.code == 'paper_no_pmid')) == 'evidence'


def test_the_api_returns_the_qa_and_keeping_a_reaction_lists_its_issues_again():
    class Pipe:
        def run(self, spec, report):
            return PipelineResult(draft(), list(EV), [])
    store = InMemorySessionStore()
    c = TestClient(create_app(FakeAuthProvider(TOKENS), store, SyncJobRunner(), Pipe(), qa_model=Model(NOT_A_REACTION),
                              auto_qa=True))
    sid = c.post('/api/llm/sessions', headers=AUTH('alice'), json={'pmid': '1', 'review': False}).json()['sessionId']
    s = c.get(f'/api/llm/sessions/{sid}', headers=AUTH('alice')).json()
    assert s['qa']['r2']['is_reaction'] is False
    issues = c.get(f'/api/llm/sessions/{sid}/issues', headers=AUTH('alice')).json()
    keep = next(i for i in issues if i['reaction_key'] == 'r2' and i['category'] == 'keep')
    c.patch(f'/api/llm/sessions/{sid}/issues/{keep["id"]}', headers=AUTH('alice'), json={'status': 'dismissed'})
    after = c.get(f'/api/llm/sessions/{sid}/issues', headers=AUTH('alice')).json()
    assert not [i for i in after if i['reaction_key'] == 'r2' and i['blocked_by']]


def test_a_look_alike_note_is_never_held_back_and_reactions_carry_their_equation():
    s = session()
    s.issues = [Issue(id='k', source='checks', code='qa_methods_only', message='methods only', reaction_key='r1'),
                Issue(id='dup', source='checks', code='qa_possible_duplicate', message='may describe the same event as r0',
                      reaction_key='r1')]
    organize(s.issues, s.draft)
    assert s.issues[1].blocked_by is None and s.issues[1].category == 'content'
    from curator_llm.api.app import equation
    assert equation(s.draft, s.draft.reactions[1]) == 'CDK1 -> CDK1 (phospho-Tyr15) (catalyst WEE1)'
    assert equation(s.draft, s.draft.reactions[2]) == 'APC1 -> APC1'


def test_the_curator_removes_a_reaction_directly_and_everything_follows():
    class Pipe:
        def run(self, spec, report):
            return PipelineResult(draft(), list(EV), [])
    c = TestClient(create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(),
                              qa_model=Model(NOT_A_REACTION), auto_qa=True))
    sid = c.post('/api/llm/sessions', headers=AUTH('alice'), json={'pmid': '1', 'review': False}).json()['sessionId']
    # a pending chat edit of r2, which removing r2 makes moot
    edit = c.post(f'/api/llm/sessions/{sid}/proposals', headers=AUTH('alice'),
                  json={'ops': [{'op': 'replace', 'path': '/reactions/r2/summation', 'value': 'new text'}], 'reason': 'x'}).json()['id']
    before = c.get(f'/api/llm/sessions/{sid}', headers=AUTH('alice')).json()
    assert 'r2' in before['qa'] and any(r['key'] == 'r2' for r in before['reactions'])
    r = c.delete(f'/api/llm/sessions/{sid}/reactions/r2', headers=AUTH('alice'))
    assert r.status_code == 200 and r.json()['proposal']['status'] == 'accepted'
    assert [x['key'] for x in r.json()['reactions']] == ['r0', 'r1']
    after = c.get(f'/api/llm/sessions/{sid}', headers=AUTH('alice')).json()
    assert 'r2' not in after['qa'] and [x['key'] for x in after['reactions']] == ['r0', 'r1']
    assert not [i for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=AUTH('alice')).json() if i['reaction_key'] == 'r2']
    props = c.get(f'/api/llm/sessions/{sid}/proposals', headers=AUTH('alice')).json()
    assert next(p for p in props if p['id'] == edit)['status'] == 'stale'
    assert not [p for p in props if p['status'] == 'pending' and 'r2' in json.dumps(p['ops'])]   # its removal suggestion is gone
    staged = c.get(f'/api/llm/sessions/{sid}/export', headers=AUTH('alice')).json()
    events = [i for i in staged['newInstances'] if i['schemaClassName'] in ('Reaction', 'BlackBoxEvent')]
    assert len(events) == 2                                              # the staged instances follow too
    assert c.delete(f'/api/llm/sessions/{sid}/reactions/r2', headers=AUTH('alice')).status_code == 404
    assert c.delete(f'/api/llm/sessions/{sid}/reactions/r0', headers=AUTH('bob')).status_code == 404     # not her session
