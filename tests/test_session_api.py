import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (CatalystSpec, EwasSpec, ExistingRef, ReactionSpec, ReactomeDraft,
                                         SimpleEntitySpec)
from curator_llm.models.session import Issue
from curator_llm.ports.pipeline import PipelineResult
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner

TOKENS = {'alice': ('alice', 'curator'), 'bob': ('bob', 'curator'), 'viewer': ('v', 'viewer')}
AUTH = lambda t: {'Authorization': f'Bearer {t}'}


class FakePipeline:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def run(self, spec, report):
        self.calls.append(spec)
        report('extracting')
        if self.fail:
            raise RuntimeError('PMC has no full text')
        d = ReactomeDraft()
        d.participants = {
            'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7',
                              reference_entity=ExistingRef(db_id=5, display_name='UniProt:Q9BXM7 PINK1', schema_class='ReferenceGeneProduct')),
            'ub': EwasSpec(key='ub', name='UB', needs_resolution=['uniprot']),
            'atp': SimpleEntitySpec(key='atp', name='ATP', chebi='CHEBI:30616')}
        d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates ubiquitin', inputs=['ub', 'atp'],
                                    catalyst=CatalystSpec(entity='pink1'), pmids=[spec.pmid],
                                    evidence_ids=['ev-001'])]
        ev = [Evidence(id='ev-001', quote='q', supports=['catalystActivity'], page=3)]
        return PipelineResult(d, ev, [Issue(source='builder', code='duplicate_entities', message='x', severity='action')])


@pytest.fixture
def env():
    pipe = FakePipeline()
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), pipe)
    return TestClient(app), pipe


def start(c, token='alice', pmid='24751536'):
    r = c.post('/api/llm/sessions', json={'pmid': pmid, 'focus': 'PINK1'}, headers=AUTH(token))
    assert r.status_code == 202
    return r.json()


def test_start_runs_pipeline_and_session_becomes_ready(env):
    c, pipe = env
    ids = start(c)
    assert pipe.calls[0].pmid == '24751536' and pipe.calls[0].focus == 'PINK1'
    j = c.get(f'/api/llm/jobs/{ids["jobId"]}', headers=AUTH('alice')).json()
    assert j['status'] == 'done'
    s = c.get(f'/api/llm/sessions/{ids["sessionId"]}', headers=AUTH('alice')).json()
    assert s['status'] == 'ready' and s['n_reactions'] == 1 and s['reactions'][0]['dbId'] < 0
    assert s['n_open_issues'] >= 1


def test_export_returns_user_instances(env):
    c, _ = env
    sid = start(c)['sessionId']
    ui = c.get(f'/api/llm/sessions/{sid}/export', headers=AUTH('alice')).json()
    assert set(ui) == {'newInstances', 'updatedInstances', 'deletedInstances', 'bookmarks'}
    assert any(i['schemaClassName'] == 'Reaction' for i in ui['newInstances'])


def test_all_issues_are_listed_with_links_to_the_instances_they_concern(env):
    c, _ = env
    sid = start(c)['sessionId']
    issues = c.get(f'/api/llm/sessions/{sid}/issues', headers=AUTH('alice')).json()
    codes = {i['code'] for i in issues}
    assert 'duplicate_entities' in codes and 'needs_resolution:uniprot' in codes      # builder + resolver flag
    ub = next(i for i in issues if i['code'] == 'needs_resolution:uniprot')
    assert ub['participant_key'] == 'ub' and ub['instance_db_id'] < 0 and ub['id'].startswith('iss-')
    assert any(i['source'] == 'emitter' for i in issues)                                # e.g. no UniProt on UB


def test_issue_status_can_be_changed_and_survives_a_refresh(env):
    c, _ = env
    sid = start(c)['sessionId']
    issues = c.get(f'/api/llm/sessions/{sid}/issues', headers=AUTH('alice')).json()
    target = issues[0]['id']
    assert c.patch(f'/api/llm/sessions/{sid}/issues/{target}', json={'status': 'dismissed'}, headers=AUTH('alice')).status_code == 200
    assert c.patch(f'/api/llm/sessions/{sid}/issues/{target}', json={'status': 'bogus'}, headers=AUTH('alice')).status_code == 422
    assert c.get(f'/api/llm/sessions/{sid}/issues?status=open', headers=AUTH('alice')).json() != issues
    assert target not in [i['id'] for i in c.get(f'/api/llm/sessions/{sid}/issues?status=open', headers=AUTH('alice')).json()]


def test_evidence_is_served_by_instance_dbid(env):
    c, _ = env
    sid = start(c)['sessionId']
    ui = c.get(f'/api/llm/sessions/{sid}/export', headers=AUTH('alice')).json()
    cat = next(i for i in ui['newInstances'] if i['schemaClassName'] == 'CatalystActivity')
    ev = c.get(f'/api/llm/sessions/{sid}/instances/{cat["dbId"]}/evidence', headers=AUTH('alice')).json()
    assert ev[0]['field'] == 'catalystActivity' and ev[0]['evidence']['id'] == 'ev-001' and ev[0]['evidence']['page'] == 3
    assert c.get(f'/api/llm/sessions/{sid}/instances/-9999/evidence', headers=AUTH('alice')).json() == []


def test_sessions_and_jobs_are_private_to_their_owner(env):
    c, _ = env
    ids = start(c, 'alice')
    for url in (f'/api/llm/sessions/{ids["sessionId"]}', f'/api/llm/sessions/{ids["sessionId"]}/export',
                f'/api/llm/sessions/{ids["sessionId"]}/issues', f'/api/llm/jobs/{ids["jobId"]}'):
        assert c.get(url, headers=AUTH('bob')).status_code == 404
    assert c.get('/api/llm/sessions', headers=AUTH('bob')).json() == []
    assert len(c.get('/api/llm/sessions', headers=AUTH('alice')).json()) == 1


def test_failed_pipeline_marks_session_failed_with_the_reason():
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), FakePipeline(fail=True))
    c = TestClient(app)
    ids = start(c)
    s = c.get(f'/api/llm/sessions/{ids["sessionId"]}', headers=AUTH('alice')).json()
    assert s['status'] == 'failed' and 'no full text' in s['error']
    assert c.get(f'/api/llm/jobs/{ids["jobId"]}', headers=AUTH('alice')).json()['status'] == 'failed'
    assert c.get(f'/api/llm/sessions/{ids["sessionId"]}/export', headers=AUTH('alice')).status_code == 409


def test_bad_input_and_unconfigured_service():
    c = TestClient(create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), FakePipeline()))
    assert c.post('/api/llm/sessions', json={'pmid': 'abc'}, headers=AUTH('alice')).status_code == 422
    bare = TestClient(create_app(FakeAuthProvider(TOKENS)))
    assert bare.get('/api/llm/sessions', headers=AUTH('alice')).status_code == 503


def test_viewer_role_is_forbidden(env):
    c, _ = env
    assert c.post('/api/llm/sessions', json={'pmid': '1234567'}, headers=AUTH('viewer')).status_code == 403
