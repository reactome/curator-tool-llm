import json

from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (CatalystSpec, EwasSpec, ReactionSpec, ReactomeDraft, SimpleEntitySpec)
from curator_llm.ports.pipeline import PipelineResult
from curator_llm.services.paper_text import PaperText
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner
from tests.test_chat import ScriptedModel, chat, final, tool
from tests.test_edit_api import H, PAGES, Q2, TOKENS
from tests.test_existing_events import EXISTING, FakeEvents
from curator_llm.services.existing_events import find_existing


class Pipe:
    def __init__(self, events):
        self.events = events

    def run(self, spec, report):
        d = ReactomeDraft()
        d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'ub': EwasSpec(key='ub', name='UB', uniprot='P0CG47'),
                          'atp': SimpleEntitySpec(key='atp', name='ATP'), 'cccp': SimpleEntitySpec(key='cccp', name='CCCP')}
        d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates ubiquitin', inputs=['ub', 'atp'], outputs=['ub'],
                                    catalyst=CatalystSpec(entity='pink1'), evidence_ids=['ev-001'], pmids=[spec.pmid])]
        paper = PaperText.from_pages('1', PAGES)
        matches, issues = find_existing(d, spec.pmid, self.events)
        return PipelineResult(d, [paper.verify(Evidence(id='ev-001', quote=Q2))], issues, paper.to_dict(), matches)


def env(events, model=None):
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(events), events=events, chat_model=model)
    c = TestClient(app)
    sid = c.post('/api/llm/sessions', json={'pmid': '24751536'}, headers=H()).json()['sessionId']
    return c, sid


def test_pipeline_result_puts_matches_and_their_issues_on_the_session():
    c, sid = env(FakeEvents(citing=[EXISTING], candidates=[EXISTING]))
    ex = c.get(f'/api/llm/sessions/{sid}/existing', headers=H()).json()
    assert ex[0]['level'] == 'same' and ex[0]['display_name'].startswith('PINK1 phosphorylates Ub') and ex[0]['cites_pmid']
    codes = {i['code'] for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json()}
    assert {'existing_reaction_match', 'paper_already_curated'} <= codes


def test_recheck_replaces_existing_issues_keeps_ids_and_statuses_and_links_instances():
    events = FakeEvents(citing=[EXISTING], candidates=[EXISTING])
    c, sid = env(events)
    issues = c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json()
    match_issue = next(i for i in issues if i['code'] == 'existing_reaction_match')
    c.patch(f'/api/llm/sessions/{sid}/issues/{match_issue["id"]}', json={'status': 'dismissed'}, headers=H())
    r = c.post(f'/api/llm/sessions/{sid}/existing/check', headers=H())
    assert r.status_code == 200 and r.json()[0]['level'] == 'same'
    again = c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json()
    same = [i for i in again if i['code'] == 'existing_reaction_match']
    assert len(same) == 1 and same[0]['id'] == match_issue['id'] and same[0]['status'] == 'dismissed'
    assert same[0]['instance_db_id'] < 0                                            # linked to the emitted reaction
    events.candidates = []                                                           # Reactome no longer has it
    c.post(f'/api/llm/sessions/{sid}/existing/check', headers=H())
    assert not [i for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json() if i['code'] == 'existing_reaction_match' and i['status'] == 'open']


def test_issue_ids_stay_stable_across_edits():
    c, sid = env(FakeEvents(citing=[EXISTING], candidates=[EXISTING]))
    before = {(i['code'], i['message']): i['id'] for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json()}
    pid = c.post(f'/api/llm/sessions/{sid}/proposals', json={'ops': [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'x'}]}, headers=H()).json()['id']
    c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/accept', headers=H())
    after = {(i['code'], i['message']): i['id'] for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json()}
    assert all(after[k] == v for k, v in before.items() if k in after) and len(set(after.values())) == len(after)


def test_accepting_an_edit_rechecks_the_touched_reactions():
    events = FakeEvents(candidates=[])
    c, sid = env(events)
    assert c.get(f'/api/llm/sessions/{sid}/existing', headers=H()).json() == []
    events.candidates = [EXISTING]
    pid = c.post(f'/api/llm/sessions/{sid}/proposals', json={'ops': [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'y'}]}, headers=H()).json()['id']
    c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/accept', headers=H())
    assert c.get(f'/api/llm/sessions/{sid}/existing', headers=H()).json()[0]['level'] == 'same'


def test_existing_check_unconfigured_is_503_and_private():
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(FakeEvents()))
    c = TestClient(app)
    sid = c.post('/api/llm/sessions', json={'pmid': '1'}, headers=H()).json()['sessionId']
    assert c.post(f'/api/llm/sessions/{sid}/existing/check', headers=H()).status_code == 503
    c2, sid2 = env(FakeEvents())
    assert c2.get(f'/api/llm/sessions/{sid2}/existing', headers=H('bob')).status_code == 404


def test_chat_tool_reports_matches_and_the_curator_can_record_reuse_by_patch():
    model = ScriptedModel([tool('find_existing_reactome', key='r0'), tool('find_existing_reactome', key='nope'), final('found one')])
    c, sid = env(FakeEvents(candidates=[EXISTING]), model)
    chat(c, sid, 'does Reactome have r0?')
    res = lambda i: json.loads(model.calls[i]['messages'][-1]['content'][0]['content'])
    assert res(1)[0]['level'] == 'same' and res(1)[0]['st_id'] == 'R-HSA-9834945'
    assert 'no reaction with key' in res(2)['error']
    # reuse is recorded as an ordinary proposal that sets `existing`; the emitter then skips the new reaction
    ops = [{'op': 'add', 'path': '/reactions/r0/existing', 'value': {'db_id': 9834945, 'display_name': EXISTING.display_name, 'schema_class': 'Reaction'}}]
    p = c.post(f'/api/llm/sessions/{sid}/proposals', json={'ops': ops, 'curator_assertion': True, 'reason': 'reuse'}, headers=H())
    assert p.status_code == 201
    c.post(f'/api/llm/sessions/{sid}/proposals/{p.json()["id"]}/accept', headers=H())
    ui = c.get(f'/api/llm/sessions/{sid}/export', headers=H()).json()
    assert not any(i['schemaClassName'] == 'Reaction' for i in ui['newInstances'])


def test_a_scoped_recheck_never_duplicates_the_paper_level_issue():
    c, sid = env(FakeEvents(citing=[EXISTING], candidates=[EXISTING]))
    for _ in range(3):
        pid = c.post(f'/api/llm/sessions/{sid}/proposals', json={'ops': [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': str(_)}]}, headers=H()).json()['id']
        c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/accept', headers=H())
    issues = c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json()
    assert sum(i['code'] == 'paper_already_curated' for i in issues) == 1
    assert len({i['id'] for i in issues}) == len(issues)                        # ids are unique
