import os

import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (CatalystSpec, EwasSpec, ReactionSpec, ReactomeDraft, SimpleEntitySpec)
from curator_llm.ports.pipeline import PipelineResult
from curator_llm.services.paper_text import PaperText
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner

TOKENS = {'alice': ('alice', 'curator'), 'bob': ('bob', 'curator')}
H = lambda t='alice': {'Authorization': f'Bearer {t}'}
Q1 = 'TcPINK1 WT, but not KD, incorporates 32P from radiolabeled ATP onto Ub'
Q2 = 'CCCP treatment stabilised PINK1 on the outer mitochondrial membrane'
PAGES = [f'Results\n{Q1} (Fig. 3 B).\n{Q2}.\n', 'Discussion\nWe propose a feed-forward model of Parkin activation.\n']


class Pipe:
    calls = []

    def run(self, spec, report):
        self.calls.append(spec)
        d = ReactomeDraft()
        d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'ub': EwasSpec(key='ub', name='UB'),
                          'atp': SimpleEntitySpec(key='atp', name='ATP', chebi='CHEBI:30616'),
                          'cccp': SimpleEntitySpec(key='cccp', name='CCCP')}
        d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates ubiquitin', inputs=['ub', 'atp'], outputs=['ub'],
                                    catalyst=CatalystSpec(entity='pink1'), evidence_ids=['ev-001'], pmids=['24751536'])]
        paper = PaperText.from_pages('24751536', PAGES)
        ev = paper.verify(Evidence(id='ev-001', quote=Q1, supports=['catalystActivity']))
        return PipelineResult(d, [ev], [], paper.to_dict())


class Resolving:
    """Stands in for Resolver: records that it ran, adds one note."""
    calls = 0

    def resolve(self, draft):
        Resolving.calls += 1
        return []


@pytest.fixture
def env(tmp_path):
    Pipe.calls = []
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(),
                     resolver_factory=Resolving, upload_dir=str(tmp_path), cors_origins=['http://localhost:4200'])
    c = TestClient(app)
    sid = c.post('/api/llm/sessions', json={'pmid': '24751536'}, headers=H()).json()['sessionId']
    return c, sid, tmp_path


def propose(c, sid, ops, **kw):
    return c.post(f'/api/llm/sessions/{sid}/proposals', json={'ops': ops, **kw}, headers=H())


CCCP_OP = [{'op': 'add', 'path': '/reactions/r0/regulations/-',
            'value': {'kind': 'positive', 'regulator': 'cccp', 'note': 'seen after CCCP treatment'}}]


def test_claim_edit_needs_verified_evidence_and_nothing_changes_until_accepted(env):
    c, sid, _ = env
    r = propose(c, sid, CCCP_OP, reason='add the damage condition')
    assert r.status_code == 422 and 'needs at least one verified quote' in r.json()['detail']
    r = propose(c, sid, CCCP_OP, reason='add the damage condition',
                evidence=[{'quote': Q2, 'supports': ['regulatedBy[0]']}])
    assert r.status_code == 201
    p = r.json()
    assert p['status'] == 'pending' and p['id'] == 'p-001' and p['reaction_keys'] == ['r0']
    assert p['evidence'][0]['verified'] == 'exact' and p['evidence'][0]['page'] == 1
    assert any('regulations' in l for l in p['summary'])
    ui = c.get(f'/api/llm/sessions/{sid}/export', headers=H()).json()
    assert not any(i['schemaClassName'] == 'PositiveRegulation' for i in ui['newInstances'])      # still untouched


def test_accept_applies_edit_registers_evidence_and_re_emits(env):
    c, sid, _ = env
    pid = propose(c, sid, CCCP_OP, reason='r', evidence=[{'quote': Q2, 'supports': ['regulatedBy[0]']}]).json()['id']
    Resolving.calls = 0
    r = c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/accept', headers=H())
    assert r.status_code == 200 and r.json()['proposal']['status'] == 'accepted' and Resolving.calls == 1
    ui = c.get(f'/api/llm/sessions/{sid}/export', headers=H()).json()
    reg = next(i for i in ui['newInstances'] if i['schemaClassName'] == 'PositiveRegulation')
    ev = c.get(f'/api/llm/sessions/{sid}/instances/{reg["dbId"]}/evidence', headers=H()).json()
    assert ev[0]['field'] == 'regulatedBy[0]' and ev[0]['evidence']['quote'] == Q2
    detail = c.get(f'/api/llm/sessions/{sid}/reactions/r0', headers=H()).json()
    assert [e['id'] for e in detail['evidence']] == ['ev-001', 'ev-002'] and 'cccp' in detail['participants']
    assert c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/accept', headers=H()).status_code == 409   # only once


def test_reject_changes_nothing(env):
    c, sid, _ = env
    pid = propose(c, sid, [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'x'}]).json()['id']
    assert c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/reject', headers=H()).json()['status'] == 'rejected'
    assert c.get(f'/api/llm/sessions/{sid}/reactions/r0', headers=H()).json()['reaction']['summation'] == ''
    assert c.post(f'/api/llm/sessions/{sid}/proposals/{pid}/accept', headers=H()).status_code == 409


def test_summation_edit_needs_no_evidence_but_a_noop_is_refused(env):
    c, sid, _ = env
    assert propose(c, sid, [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'New text.'}]).status_code == 201
    r = propose(c, sid, [{'op': 'replace', 'path': '/reactions/r0/name', 'value': 'PINK1 phosphorylates ubiquitin'}])
    assert r.status_code == 422 and 'changes nothing' in r.json()['detail']


def test_fabricated_quote_is_refused_but_curator_assertion_is_allowed(env):
    c, sid, _ = env
    r = propose(c, sid, CCCP_OP, evidence=[{'quote': 'PINK1 is degraded by CCCP in zebrafish neurons overnight'}])
    assert r.status_code == 422 and 'not found in the paper' in r.json()['detail']
    r = propose(c, sid, CCCP_OP, reason='curator knows this from the lab', curator_assertion=True)
    assert r.status_code == 201 and r.json()['evidence'][0]['claim_origin'] == 'curator_assertion'


def test_bad_patch_reports_why(env):
    c, sid, _ = env
    r = propose(c, sid, [{'op': 'replace', 'path': '/reactions/r0/inputs', 'value': ['ghost']}],
                evidence=[{'quote': Q1}])
    assert r.status_code == 422 and 'unknown participant' in r.json()['detail']


def test_stale_proposal_is_detected(env):
    c, sid, _ = env
    a = propose(c, sid, [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'A'}]).json()['id']
    propose(c, sid, [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'B'}])
    c.post(f'/api/llm/sessions/{sid}/proposals/{a}/accept', headers=H())
    # b is still a valid replace; make it stale by removing what it needs
    d = propose(c, sid, [{'op': 'remove', 'path': '/participants/cccp'}]).json()['id']
    e = propose(c, sid, [{'op': 'add', 'path': '/reactions/r0/regulations/-', 'value': {'kind': 'positive', 'regulator': 'cccp'}}],
                evidence=[{'quote': Q2}]).json()['id']
    assert c.post(f'/api/llm/sessions/{sid}/proposals/{d}/accept', headers=H()).status_code == 200
    r = c.post(f'/api/llm/sessions/{sid}/proposals/{e}/accept', headers=H())
    assert r.status_code == 409 and 'no longer applies' in r.json()['detail']
    assert c.get(f'/api/llm/sessions/{sid}/proposals?status=stale', headers=H()).json()[0]['id'] == e


def test_paper_search_and_privacy(env):
    c, sid, _ = env
    hits = c.get(f'/api/llm/sessions/{sid}/paper/search', params={'q': 'feed-forward Parkin', 'section': 'Discussion'}, headers=H()).json()
    assert hits[0]['section'] == 'Discussion' and hits[0]['page'] == 2
    for url in (f'/api/llm/sessions/{sid}/paper/search?q=x', f'/api/llm/sessions/{sid}/proposals', f'/api/llm/sessions/{sid}/reactions/r0'):
        assert c.get(url, headers=H('bob')).status_code == 404
    assert c.post(f'/api/llm/sessions/{sid}/proposals', json={'ops': CCCP_OP, 'curator_assertion': True}, headers=H('bob')).status_code == 404


# ── PDF upload ───────────────────────────────────────────────────────────
def test_pdf_upload_starts_a_session_and_stores_the_file_per_user(env):
    c, _, tmp = env
    r = c.post('/api/llm/sessions/upload', files={'file': ('My Paper (v2).pdf', b'%PDF-1.7 fake', 'application/pdf')},
               data={'focus': 'PINK1'}, headers=H())
    assert r.status_code == 202
    spec = Pipe.calls[-1]
    assert spec.pmid is None and spec.focus == 'PINK1' and spec.pdf_path.startswith(str(tmp / 'alice'))
    assert os.path.isfile(spec.pdf_path) and spec.pdf_path.endswith('My_Paper_v2_.pdf') and '..' not in spec.pdf_path
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=H()).json()
    assert s['source'] == 'My_Paper_v2_.pdf' and s['pmid'] is None and s['status'] == 'ready'


def test_pdf_upload_rejects_non_pdf_oversize_and_path_tricks(env, monkeypatch):
    c, _, tmp = env
    assert c.post('/api/llm/sessions/upload', files={'file': ('x.pdf', b'<html>not a pdf', 'application/pdf')}, headers=H()).status_code == 422
    monkeypatch.setattr('curator_llm.api.app.MAX_UPLOAD_BYTES', 10)
    assert c.post('/api/llm/sessions/upload', files={'file': ('x.pdf', b'%PDF-' + b'0' * 50, 'application/pdf')}, headers=H()).status_code == 413
    monkeypatch.setattr('curator_llm.api.app.MAX_UPLOAD_BYTES', 10_000)
    c.post('/api/llm/sessions/upload', files={'file': ('../../etc/passwd.pdf', b'%PDF-1', 'application/pdf')}, headers=H())
    assert not os.path.exists(tmp.parent / 'etc') and Pipe.calls[-1].pdf_path.startswith(str(tmp / 'alice'))
    assert c.post('/api/llm/sessions/upload', files={'file': ('x.pdf', b'%PDF-1', 'application/pdf')}).status_code == 401


def test_cors_allows_the_frontend_origin_only(env):
    c, _, _ = env
    ok = c.options('/api/llm/sessions', headers={'Origin': 'http://localhost:4200', 'Access-Control-Request-Method': 'GET',
                                                 'Access-Control-Request-Headers': 'Authorization'})
    bad = c.options('/api/llm/sessions', headers={'Origin': 'http://evil.example', 'Access-Control-Request-Method': 'GET'})
    assert ok.headers.get('access-control-allow-origin') == 'http://localhost:4200'
    assert 'access-control-allow-origin' not in bad.headers
