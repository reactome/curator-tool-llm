import json

import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.models.reactome import (CatalystSpec, EwasSpec, ModifiedResidueSpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec, SimpleEntitySpec)
from curator_llm.ports.chat_model import ModelTurn
from curator_llm.ports.pipeline import PipelineResult
from curator_llm.services.paper_text import PaperText
from curator_llm.services.qa import parse_llm_review, qa_reaction, rule_findings
from tests.fakes.ports import FakeAuthProvider, SyncJobRunner
from tests.test_chat import ScriptedModel, chat, final, tool
from tests.test_edit_api import H, PAGES, Q1, TOKENS

PHOS = ModifiedResidueSpec(psi_mod='MOD:00046', residue='S', coordinate=65)


def draft():
    d = ReactomeDraft()
    d.participants = {
        'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'ub': EwasSpec(key='ub', name='UB'),
        'pub': EwasSpec(key='pub', name='UB', modifications=[PHOS]), 'atp': SimpleEntitySpec(key='atp', name='ATP'),
        'cccp': SimpleEntitySpec(key='cccp', name='CCCP')}
    d.reactions = [ReactionSpec(key='good', name='PINK1 phosphorylates Ub', inputs=['ub', 'atp'], outputs=['pub'],
                                catalyst=CatalystSpec(entity='pink1'), summation='s', pmids=['1'],
                                regulations=[RegulationSpec(kind='positive', regulator='cccp', note='after CCCP')],
                                evidence_ids=['e1', 'e2'])]
    return d


EV = [Evidence(id='e1', quote='q', supports=['catalystActivity'], verified=Verification.EXACT, experimental_species='Homo sapiens'),
      Evidence(id='e2', quote='q2', supports=['regulatedBy[0]'], verified=Verification.EXACT)]
codes = lambda fs: {f.code for f in fs}


def test_a_sound_reaction_has_no_action_findings():
    d = draft()
    fs = rule_findings(d, d.reactions[0], {e.id: e for e in EV})
    assert fs == []


def test_each_rule_fires():
    d = draft()
    r = d.reactions[0]
    ev = {e.id: e for e in EV}
    r.inputs, r.outputs = [], []
    assert {'qa_no_inputs', 'qa_no_outputs'} <= codes(rule_findings(d, r, ev))
    r.inputs, r.outputs = ['ub'], ['ub']
    assert 'qa_no_change' in codes(rule_findings(d, r, ev))
    r.outputs = ['pub']
    r.catalyst = CatalystSpec(entity='atp')
    assert 'qa_catalyst_not_protein' in codes(rule_findings(d, r, ev))
    r.catalyst = CatalystSpec(entity='pink1')
    r.regulations = [RegulationSpec(kind='positive', regulator='ub'), RegulationSpec(kind='positive', regulator='cccp'),
                     RegulationSpec(kind='negative', regulator='cccp')]
    fs = codes(rule_findings(d, r, ev))
    assert {'qa_regulator_is_input', 'qa_conflicting_regulation', 'qa_unsupported_regulation'} <= fs
    r.regulations = []
    r.evidence_ids = []
    assert 'qa_no_evidence' in codes(rule_findings(d, r, ev))
    r.evidence_ids, r.summation, r.pmids = ['e2'], '', []
    fs = codes(rule_findings(d, r, ev))
    assert {'qa_unsupported_catalyst', 'qa_no_summation', 'qa_no_literature_reference'} <= fs


def test_weak_evidence_kinds():
    d = draft()
    r = d.reactions[0]
    r.regulations = []
    mk = lambda **k: {'e1': Evidence(id='e1', quote='q', supports=['catalystActivity'], **k)}
    r.evidence_ids = ['e1']
    assert 'qa_only_assertion' in codes(rule_findings(d, r, mk(claim_origin=ClaimOrigin.CURATOR_ASSERTION)))
    assert 'qa_only_cited' in codes(rule_findings(d, r, mk(claim_origin=ClaimOrigin.CITED)))
    assert 'qa_fuzzy_evidence' in codes(rule_findings(d, r, mk(verified=Verification.FUZZY)))
    fs = rule_findings(d, r, mk(experimental_species='Tribolium castaneum'))
    assert 'qa_non_human_evidence' in codes(fs) and next(f for f in fs if f.code == 'qa_non_human_evidence').severity == 'info'


class Reviewer:
    def __init__(self, text=None, boom=False):
        self.text, self.boom, self.calls = text, boom, []

    def turn(self, system, messages, tools, on_text):
        self.calls.append((system, messages, tools))
        if self.boom:
            raise RuntimeError('overloaded')
        return ModelTurn(self.text)


def test_llm_review_adds_findings_score_and_sees_the_quotes():
    d = draft()
    rv = Reviewer('Here you go: {"verdict": "needs_work", "score": 0.4, "findings": [{"severity": "action", '
                  '"message": "summation overstates the result", "field": "summation"}, {"severity": "weird", "message": "m2"}, {"message": ""}]}')
    res = qa_reaction(d, 'good', [Evidence(id='e1', quote='UNIQUE-QUOTE-TEXT', supports=['catalystActivity']), EV[1]], rv)
    assert res.llm_used and res.score == 0.4 and res.verdict == 'needs_work'
    assert [f.message for f in res.findings if f.source == 'llm'] == ['summation overstates the result', 'm2']
    assert [f.severity for f in res.findings if f.source == 'llm'] == ['action', 'warning']
    assert 'UNIQUE-QUOTE-TEXT' in rv.calls[0][1][0]['content'] and rv.calls[0][2] == []


def test_llm_failure_or_garbage_keeps_the_rule_findings():
    d = draft()
    d.reactions[0].evidence_ids = []
    for rv in (Reviewer(boom=True), Reviewer('not json at all')):
        res = qa_reaction(d, 'good', [], rv)
        assert not res.llm_used and 'qa_llm_unavailable' in codes(res.findings) and 'qa_no_evidence' in codes(res.findings)
        assert res.verdict == 'needs_work'


def test_parse_llm_review_is_tolerant_but_bounded():
    r = parse_llm_review('{"verdict": "maybe", "score": 7, "findings": []}')
    assert r['verdict'] is None and r['score'] is None and r['findings'] == []
    with pytest.raises(ValueError):
        parse_llm_review('nothing')
    with pytest.raises(KeyError):
        qa_reaction(draft(), 'nope', [])


# ── through the API and chat ───────────────────────────────────────────
class Pipe:
    def run(self, spec, report):
        d = draft()
        d.reactions[0].evidence_ids = ['ev-001']
        paper = PaperText.from_pages('1', PAGES)
        return PipelineResult(d, [paper.verify(Evidence(id='ev-001', quote=Q1, supports=['catalystActivity']))], [], paper.to_dict())


def env(model=None):
    c = TestClient(create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(), chat_model=model))
    sid = c.post('/api/llm/sessions', json={'pmid': '1'}, headers=H()).json()['sessionId']
    return c, sid


def test_qa_endpoint_records_issues_and_replaces_them_on_rerun():
    c, sid = env(Reviewer('{"verdict": "ok", "score": 0.9, "findings": []}'))
    r = c.post(f'/api/llm/sessions/{sid}/qa/good', headers=H()).json()
    assert r['llm_used'] and r['score'] == 0.9 and 'qa_unsupported_regulation' in {f['code'] for f in r['findings']}
    qa_issues = [i for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json() if i['source'] == 'qa']
    assert qa_issues and all(i['reaction_key'] == 'good' and i['instance_db_id'] < 0 for i in qa_issues)
    target = qa_issues[0]
    c.patch(f'/api/llm/sessions/{sid}/issues/{target["id"]}', json={'status': 'dismissed'}, headers=H())
    c.post(f'/api/llm/sessions/{sid}/qa/good?llm=false', headers=H())
    again = [i for i in c.get(f'/api/llm/sessions/{sid}/issues', headers=H()).json() if i['source'] == 'qa']
    assert len(again) == len(qa_issues)                                        # replaced, not duplicated
    same = next(i for i in again if i['message'] == target['message'])
    assert same['id'] == target['id'] and same['status'] == 'dismissed'        # id and status survive a re-run


def test_qa_without_llm_and_errors_and_privacy():
    c, sid = env()
    r = c.post(f'/api/llm/sessions/{sid}/qa/good', headers=H()).json()
    assert not r['llm_used'] and r['verdict'] in ('ok', 'needs_work')
    assert c.post(f'/api/llm/sessions/{sid}/qa/nope', headers=H()).status_code == 404
    assert c.post(f'/api/llm/sessions/{sid}/qa/good', headers=H('bob')).status_code == 404
    assert c.post(f'/api/llm/sessions/{sid}/qa/good').status_code == 401


def test_chat_tool_runs_qa():
    model = ScriptedModel([tool('run_qa', key='good'), final('looks mostly fine')])
    c, sid = env(model)
    # the QA call inside the tool goes to the same model: give it a reviewer turn first
    model.turns.insert(1, ModelTurn('{"verdict": "ok", "score": 0.8, "findings": []}'))
    chat(c, sid, 'is r good sound?')
    out = json.loads(model.calls[2]['messages'][-1]['content'][0]['content'])
    assert out['reaction_key'] == 'good' and out['llm_used'] and out['score'] == 0.8
