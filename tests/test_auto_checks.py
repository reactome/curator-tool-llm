"""The automatic checks a curator gets on every draft: what the quotes say about themselves, reactions the merge left
apart that look alike, a suggestion to remove a reaction with no direct evidence, and the model QA run after the
pipeline."""
import json

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.models.evidence import ClaimOrigin, Evidence
from curator_llm.models.reactome import (CatalystSpec, EwasSpec, PathwaySpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec)
from curator_llm.models.session import PaperInfo, Session
from curator_llm.ports.chat_model import ModelTurn
from curator_llm.ports.pipeline import PipelineInput
from curator_llm.services.patching import apply_patch
from curator_llm.services.qa import auto_findings, duplicate_findings, rule_findings
from curator_llm.services.session_service import SessionService
from curator_llm.services.suggestions import suggest_fixes
from tests.fakes.ports import SyncJobRunner
from tests.test_session_api import FakePipeline

P = lambda k, name: EwasSpec(key=k, name=name)


def one(quotes, **kw):
    """A draft with one reaction whose quotes are `quotes` [(text, origin, section)]."""
    d = ReactomeDraft()
    d.participants = {'wee1': P('wee1', 'WEE1'), 'cdh1': P('cdh1', 'CDH1'), 'pcdh1': P('pcdh1', 'phospho-CDH1')}
    ev = [Evidence(id=f'e{i}', quote=q, claim_origin=o, section=s, source=kw.get('source', 'paper.pdf'))
          for i, (q, o, s) in enumerate(quotes)]
    d.reactions = [ReactionSpec(key='r0', name='WEE1 phosphorylates CDH1', inputs=['cdh1'], outputs=['pcdh1'],
                                catalyst=CatalystSpec(entity='wee1'), summation='s', pmids=['1'],
                                evidence_ids=[e.id for e in ev])]
    return d, ev


def codes(quotes, reviews=(), **kw):
    d, ev = one(quotes, **kw)
    return {f.code for f in rule_findings(d, d.reactions[0], {e.id: e for e in ev}, reviews)}


OWN, CITED = ClaimOrigin.THIS_PAPER, ClaimOrigin.CITED


# ── what the quotes say ────────────────────────────────────────────────────
def test_a_result_shown_here_raises_nothing():
    assert not codes([('WEE1 phosphorylated tyrosines in CDH1 in vitro (Fig. 4f).', OWN, 'Results')]) & {
        'qa_only_cited', 'qa_negative_only', 'qa_hedged_only', 'qa_methods_only'}


def test_a_quote_carrying_a_reference_counts_as_cited_unless_the_paper_is_a_review():
    q = [('mTOR inhibition decreased Thr129 phosphorylation; however, mTOR only phosphorylates Thr129 (ref. 43).', OWN, None),
         ('mTOR-mediated phosphorylation of CDH1 Thr129 inactivates APC/C (ref. 43).', CITED, None)]
    assert 'qa_only_cited' in codes(q)
    assert 'qa_only_cited' not in codes(q, reviews=['paper.pdf'])
    assert 'qa_only_cited' not in codes(q + [('We show WEE1 phosphorylates CDH1.', OWN, None)])   # one own result is enough


def test_negative_and_hedged_results_are_judged_on_the_quotes_that_state_something():
    neg = [('whereas phosphorylation of related coactivator CDC20 was negligible (Extended Data Fig. 7g).', OWN, None),
           ('1.5 ug of GST-FZY N159 protein was incubated with 160 ng WEE1 kinase.', OWN, 'Methods'),
           ('Western blot showing in vitro kinase reaction of GST-CDC20 N159 with WEE1.', OWN, None)]
    assert 'qa_negative_only' in codes(neg) and 'qa_methods_only' not in codes(neg)
    assert 'qa_hedged_only' in codes([('WEE1 may be required for inhibition of APC/C before CDK2-mediated degradation.', OWN, 'Discussion')])


def test_only_methods_or_figure_panel_descriptions():
    q = [('Western blot showing stability of CDH1 protein in cells treated with WEE1i.', OWN, None),
         ('Cells were lysed every hour for 4 h.', OWN, 'Methods')]
    assert 'qa_methods_only' in codes(q)
    assert 'qa_methods_only' not in codes(q + [('CDH1 levels did fall after WEE1 inhibition.', OWN, 'Results')])


# ── look-alike reactions ───────────────────────────────────────────────────
def pair(a, b):
    d = ReactomeDraft()
    names = {n for spec in (a, b) for side in spec[1:3] for n in side} | {spec[0] for spec in (a, b) if spec[0]}
    names |= {g for spec in (a, b) for g in (spec[3] if len(spec) > 3 else [])}
    d.participants = {n: P(n, n) for n in names}
    d.reactions = [ReactionSpec(key=f'r{i}', name=f'reaction {i}', inputs=list(s[1]), outputs=list(s[2]),
                                catalyst=CatalystSpec(entity=s[0]) if s[0] else None,
                                regulations=[RegulationSpec(kind='positive', regulator=g) for g in (s[3] if len(s) > 3 else [])])
                   for i, s in enumerate((a, b))]
    return [k for k, _ in duplicate_findings(d)]


def test_a_look_alike_with_less_detail_is_a_possible_duplicate():
    assert pair(('WEE1', ['CDK1', 'ATP'], ['CDK1']), ('WEE1', ['CDK1:Cyclin B'], ['CDK1(pTyr15):Cyclin B'])) == ['r1']
    assert pair(('APC/C-CDH1', ['Cyclin B'], ['ubiquitinated Cyclin B']), ('APC/C-CDH1', ['cyclin B'], ['degraded cyclin B'])) == ['r1']


def test_different_catalyst_substrate_or_regulators_are_not():
    assert pair(('WEE1', ['CDK1'], ['pCDK1']), ('CDK2', ['CDK1'], ['pCDK1'])) == []
    assert pair(('WEE1', ['CDK1'], ['pCDK1']), ('WEE1', ['CDH1'], ['pCDH1'])) == []
    # no catalyst and nothing produced: too little to go on
    assert pair((None, ['FZR1'], []), (None, ['FZR1'], [])) == []
    assert pair((None, ['FZR1'], ['FZR1 degraded'], ['EMI1']), (None, ['FZR1'], ['FZR1 degraded'], ['siRNA'])) == []


# ── the suggestion to remove a reaction, and the session ───────────────────
def test_a_cited_only_reaction_gets_a_remove_suggestion_that_applies_cleanly():
    d, ev = one([('WEE1 phosphorylates CDK1 on Tyr15 (refs. 1-3).', CITED, None)])
    d.participants['x'] = P('x', 'X')
    d.reactions.append(ReactionSpec(key='r1', name='next', inputs=['x'], outputs=['x'], preceding=['r0']))
    d.pathway = PathwaySpec(name='p', reactions=['r0', 'r1'])
    [s] = [s for s in suggest_fixes(d, ev) if s.code == 'remove_reaction']
    after = apply_patch(d, s.ops)
    assert [r.key for r in after.reactions] == ['r1'] and after.reactions[0].preceding == [] and after.pathway.reactions == ['r1']
    assert [x for x in suggest_fixes(d, ev, reviews=['paper.pdf']) if x.code == 'remove_reaction'] == []


def test_the_session_lists_the_checks_and_knows_which_papers_are_reviews():
    d, ev = one([('WEE1 phosphorylates CDK1 on Tyr15 (refs. 1-3).', CITED, None)])
    s = Session(owner='o', draft=d, evidence=ev, status='ready', papers=[PaperInfo(source='paper.pdf', kind='pdf')])
    SessionService.refresh(s)
    assert any(i.code == 'qa_only_cited' and i.source == 'checks' for i in s.issues)
    s.papers[0].review = True
    SessionService.refresh(s)
    assert not any(i.code == 'qa_only_cited' for i in s.issues)
    assert {k for k, f in auto_findings(d, ev, ['paper.pdf'])} == set()


class QAModel:
    model = 'fake-qa'

    def __init__(self):
        self.calls = 0

    def turn(self, system, messages, tools, on_text):
        self.calls += 1
        return ModelTurn(json.dumps({'verdict': 'needs_work', 'score': 0.3, 'summary': 'Remove it.', 'findings': [
            {'severity': 'action', 'message': 'not a molecular reaction: remove it', 'field': None}]}))


def run_session(auto_qa=True, replayed=False):
    model, store = QAModel(), InMemorySessionStore()

    class Pipe(FakePipeline):
        def run(self, spec, report):
            r = super().run(spec, report)
            r.replayed = replayed
            return r
    svc = SessionService(store, SyncJobRunner(), Pipe(), qa_model=model, auto_qa=auto_qa)
    sid, _ = svc.start('o', PipelineInput(pmid='1'))
    return store.get(sid, 'o'), model


def test_a_new_draft_gets_the_model_qa_before_it_is_ready_unless_off_or_replayed():
    s, model = run_session()
    assert s.status == 'ready' and model.calls == len(s.draft.reactions) == 1
    assert 'qa_llm' in [i.code for i in s.issues if i.source == 'qa']     # with the rules a QA run adds (no summation)
    assert any(c.action.startswith('qa ') and c.actor == 'pipeline' for c in s.change_log)
    assert run_session(auto_qa=False)[1].calls == 0
    assert run_session(replayed=True)[1].calls == 0
