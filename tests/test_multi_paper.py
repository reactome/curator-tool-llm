"""Several papers in one session: extraction and merge, evidence per paper, the REST surface, saved results."""
import json
import os
import subprocess

import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app, parse_pmids
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import ReactomeDraft
from curator_llm.models.session import PaperInfo, Session
from curator_llm.ports.events import EventRow
from curator_llm.ports.extractor import ExtractionResult
from curator_llm.ports.pipeline import PaperRef, PipelineInput, PipelineResult
from curator_llm.services.existing_events import find_existing
from curator_llm.services.extraction import SubprocessExtractor
from curator_llm.services.paper_text import PaperText
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.session_service import ProposalError, SessionService
from curator_llm.services.snapshots import SnapshotStore, snapshot_key
from tests.fakes.ports import FakeAuthProvider, FakeInstanceLookup, FakeUniProt, SyncJobRunner
from tests.test_pipeline_default import Model
from tests.test_session_api import AUTH, FakePipeline, TOKENS

P1, P2 = '11111111', '22222222'
Q1 = 'Parkin binds phosphorylated ubiquitin in cells.'
Q2 = 'PINK1 phosphorylates ubiquitin at serine 65 in vitro.'


# ── the extractor ──────────────────────────────────────────────────────────
class MultiRunner:
    """Fake subprocess runner: run_extraction.py writes its paper's file, run_merge.py writes the merged file."""

    def __init__(self, tmp, fail=(), empty=()):
        self.tmp, self.fail, self.empty, self.cmds, self.lines = str(tmp), set(fail), set(empty), [], {}

    def stem(self, spec, gene):
        return f'{gene}_{os.path.basename(spec)}'

    def __call__(self, cmd, cwd, timeout, on_line=None):
        self.cmds.append(cmd[1:])
        name = os.path.basename(cmd[1])
        if name == 'run_extraction.py':
            spec = cmd[2]
            gene = cmd[cmd.index('--gene') + 1]
            label = self.stem(spec, gene) + ('_localpdf' if '--tag' in cmd else '')
            if spec in self.fail:
                return subprocess.CompletedProcess(cmd, 3, '', 'Traceback: no full text')
            source = f'PMID:{spec}' if spec.isdigit() else os.path.basename(spec)
            rx = [] if spec in self.empty else [{'source': source, 'annotation_result': {'name': f'r of {source}', 'input': ['A'], 'output': ['B']}}]
            json.dump(rx, open(os.path.join(self.tmp, f'{label}_extraction.json'), 'w'))
            if on_line:
                on_line('[setup] x: Results=1w (via heuristic) -> 2 chunks')
                on_line('  chunk 1/2: 1 reaction(s)')
            return subprocess.CompletedProcess(cmd, 0, '[done] 5 call(s), tokens in 100 / out 20\\n'.replace('\\\\n', '\n'), '')
        out = cmd[cmd.index('--out') + 1]
        inputs = cmd[2:cmd.index('--out')]
        merged = [r for path in inputs for r in json.load(open(path))]
        json.dump(merged[:1] + merged[1:], open(out, 'w'))
        if on_line:
            on_line('[cross] merging 2 reactions from 2 papers')
        return subprocess.CompletedProcess(cmd, 0, '[usage] 7 call(s), tokens in 1,000 / out 200\n', '')


def extractor(tmp, **kw):
    runner = MultiRunner(tmp, **kw)
    return SubprocessExtractor(root=str(tmp), results_dir=str(tmp), run=runner, stem=runner.stem, paper_workers=2), runner


def test_extract_many_extracts_each_paper_then_merges_them_in_one_call(tmp_path):
    ex, runner = extractor(tmp_path)
    got = []
    res = ex.extract_many([PaperRef(pmid=P1), PaperRef(pdf_path='/up/b.pdf')], 'g', report=got.append)
    assert res.ok and res.n_extracted == 2 and len(res.reactions) == 2
    merges = [c for c in runner.cmds if os.path.basename(c[0]) == 'run_merge.py']
    assert len(merges) == 1 and len([a for a in merges[0] if a.endswith('_extraction.json')]) == 2      # ONE merge over both papers
    assert {r['source'] for r in res.reactions} == {f'PMID:{P1}', 'b.pdf'}
    assert res.step_usage['extraction']['calls'] == 10 and res.step_usage['merge']['calls'] == 7        # usage of both papers added up
    assert any('paper 1' in m and 'paper 2' in m for m in got) and any('merging across papers' in m for m in got)


def test_one_paper_that_fails_is_skipped_with_a_warning_and_the_others_still_merge(tmp_path):
    ex, _ = extractor(tmp_path, fail=[P2], empty=['/up/c.pdf'])
    res = ex.extract_many([PaperRef(pmid=P1), PaperRef(pmid=P2), PaperRef(pdf_path='/up/c.pdf', label='c.pdf')], 'g')
    assert res.ok and [r['source'] for r in res.reactions] == [f'PMID:{P1}']
    assert len(res.warnings) == 2 and any(P2 in w for w in res.warnings) and any('c.pdf' in w for w in res.warnings)


def test_extract_many_reports_the_error_when_no_paper_gives_anything(tmp_path):
    ex, _ = extractor(tmp_path, fail=[P1, P2])
    res = ex.extract_many([PaperRef(pmid=P1), PaperRef(pmid=P2)], 'g')
    assert not res.ok and 'no paper gave any reactions' in res.error and len(res.warnings) == 2
    assert not extractor(tmp_path)[0].extract_many([], 'g').ok


# ── the pipeline ───────────────────────────────────────────────────────────
def merged_reactions():
    return [{'source': f'PMID:{P1}', 'annotation_result': {
        'name': 'PINK1 phosphorylates ubiquitin', 'input': ['Ub'], 'output': ['pUb'], 'sources': [f'PMID:{P1}', f'PMID:{P2}'],
        'evidence': [Q1, Q2],
        'evidence_details': {Q1: {'supports': ['reaction'], 'source': f'PMID:{P1}'}, Q2: {'supports': ['reaction'], 'source': f'PMID:{P2}'}},
        'summation': [{'text': 's'}]}}]


class FakeManyExtractor:
    def __init__(self, result):
        self.result, self.calls = result, []

    def extract_many(self, refs, gene, report=None):
        self.calls.append((list(refs), gene))
        return self.result


class FakeEvents:
    def __init__(self):
        self.cited = []

    def events_citing(self, pmid):
        self.cited.append(pmid)
        return []

    def candidate_events(self, cats, accs, limit=100):
        return []


def test_pipeline_runs_several_papers_and_checks_each_quote_against_its_own_paper(monkeypatch):
    texts = {f'PMID:{P1}': PaperText.from_pages(P1, [Q1]), f'PMID:{P2}': PaperText.from_pages(P2, [Q2])}
    monkeypatch.setattr('curator_llm.services.pipeline_default.load_paper_text', lambda src: texts.get(src))
    ext = FakeManyExtractor(ExtractionResult(ok=True, reactions=merged_reactions(), warnings=['b.pdf: skipped, no reactions']))
    events = FakeEvents()
    pipe = DefaultPipeline(FakeInstanceLookup([]), FakeUniProt({}, {}), None, model=Model(), extractor=ext, events=events)
    res = pipe.run(PipelineInput(papers=[PaperRef(pmid=P1), PaperRef(pmid=P2)], focus='PINK1'), lambda m: None)
    assert len(ext.calls[0][0]) == 2 and ext.calls[0][1] == 'PINK1'
    assert {e.pmid for e in res.evidence} == {P1, P2} and all(e.verified.value == 'exact' for e in res.evidence)
    assert {e.source for e in res.evidence} == {f'PMID:{P1}', f'PMID:{P2}'}
    assert res.draft.reactions[0].pmids == [P1, P2]                                  # the reaction cites both papers
    assert set(res.papers) == {f'PMID:{P1}', f'PMID:{P2}'} and res.paper == res.papers[f'PMID:{P1}']
    assert sorted(events.cited) == [P1, P2]                                          # both papers checked against Reactome
    assert any(i.source == 'pipeline' and 'b.pdf' in i.message for i in res.issues)  # a skipped paper is reported, not hidden


def test_pipeline_with_several_papers_needs_an_extractor_that_can_merge_them():
    class OneAtATime:
        def extract(self, pmid, pdf_path, gene):
            return ExtractionResult(ok=True, reactions=merged_reactions())
    pipe = DefaultPipeline(FakeInstanceLookup([]), FakeUniProt({}, {}), None, model=Model(), extractor=OneAtATime())
    with pytest.raises(ValueError, match='one paper at a time'):
        pipe.run(PipelineInput(papers=[PaperRef(pmid=P1), PaperRef(pmid=P2)]), lambda m: None)


def test_pipeline_error_for_several_papers_says_papers():
    ext = FakeManyExtractor(ExtractionResult(ok=False, error='boom'))
    pipe = DefaultPipeline(FakeInstanceLookup([]), FakeUniProt({}, {}), None, model=Model(), extractor=ext)
    with pytest.raises(RuntimeError, match='these papers.*boom'):
        pipe.run(PipelineInput(papers=[PaperRef(pmid=P1), PaperRef(pmid=P2)]), lambda m: None)


def test_input_refs_and_source_tags():
    assert PipelineInput(pmid=P1).refs()[0].source == f'PMID:{P1}'
    assert PipelineInput(pdf_path='/a/b/x.pdf').refs()[0].source == 'x.pdf'
    assert [r.source for r in PipelineInput(papers=[PaperRef(pmid=P1), PaperRef(pdf_path='/q/z.pdf')]).refs()] == [f'PMID:{P1}', 'z.pdf']
    assert PipelineInput().refs() == []


# ── existing Reactome events, several papers ───────────────────────────────
def test_existing_check_unions_the_events_citing_any_of_the_papers():
    class Events(FakeEvents):
        def events_citing(self, pmid):
            return [EventRow(1 if pmid == P1 else 2, f'event for {pmid}', f'R-HSA-{pmid[:1]}', 'Reaction', [], [], [])]
    _, issues = find_existing(ReactomeDraft(), [P1, P2], Events())
    note = next(i for i in issues if i.code == 'paper_already_curated')
    assert '2 event(s)' in note.message and P1 in note.message and P2 in note.message
    _, single = find_existing(ReactomeDraft(), P1, Events())                          # one paper: the wording is unchanged
    assert next(i for i in single if i.code == 'paper_already_curated').message.startswith(f'Reactome already has 1 event(s) citing PMID {P1}:')


# ── saved results ──────────────────────────────────────────────────────────
def test_snapshot_key_for_several_papers_ignores_their_order_and_differs_from_one(tmp_path):
    a, b = PipelineInput(papers=[PaperRef(pmid=P1), PaperRef(pmid=P2)], focus='pink1'), PipelineInput(papers=[PaperRef(pmid=P2), PaperRef(pmid=P1)], focus='pink1')
    assert snapshot_key(a) == snapshot_key(b) and snapshot_key(a).startswith('multi-') and snapshot_key(a).endswith('__PINK1')
    assert snapshot_key(a) != snapshot_key(PipelineInput(pmid=P1, focus='pink1')) and snapshot_key(PipelineInput(pmid=P1)) == f'pmid-{P1}'
    pdf = tmp_path / 'a.pdf'
    pdf.write_bytes(b'%PDF-1 abc')
    assert snapshot_key(PipelineInput(pdf_path=str(pdf))).startswith('pdf-')


def test_snapshot_keeps_every_papers_text(tmp_path):
    store = SnapshotStore(str(tmp_path))
    store.save('k', PipelineResult(ReactomeDraft(), [], paper={'text': 'one'}, papers={'PMID:1': {'text': 'one'}, 'b.pdf': {'text': 'two'}}))
    assert set(store.load('k').papers) == {'PMID:1', 'b.pdf'}


# ── the session: papers, search, edits ─────────────────────────────────────
def session_with_two_papers():
    s = Session(owner='alice', papers=[PaperInfo(source=f'PMID:{P1}', pmid=P1, name=P1), PaperInfo(source='b.pdf', kind='pdf', name='b.pdf')])
    s.paper_texts = {f'PMID:{P1}': PaperText.from_pages(P1, [Q1]).to_dict(), 'b.pdf': PaperText.from_pages(None, [Q2]).to_dict()}
    s.paper = s.paper_texts[f'PMID:{P1}']
    return s


def service():
    return SessionService(InMemorySessionStore(), SyncJobRunner(), FakePipeline())


def test_an_edit_may_quote_any_of_the_sessions_papers_and_the_quote_says_which():
    s = session_with_two_papers()
    ok = service()._verify_evidence(s, [{'quote': Q2, 'supports': ['reaction']}, {'quote': Q1, 'supports': ['reaction']}])
    assert [e.source for e in ok] == ['b.pdf', f'PMID:{P1}'] and ok[1].pmid == P1
    again = service()._verify_evidence(s, [{'quote': Q2, 'source': 'b.pdf'}])
    assert again[0].source == 'b.pdf'
    with pytest.raises(ProposalError, match='not found in the papers'):
        service()._verify_evidence(s, [{'quote': 'not in either paper'}])


def test_search_covers_all_papers_or_one_and_names_the_paper():
    s = session_with_two_papers()
    hits = service().search_papers(s, 'ubiquitin', limit=5)
    assert {h['source'] for h in hits} == {f'PMID:{P1}', 'b.pdf'}
    assert {h['source'] for h in service().search_papers(s, 'ubiquitin', source='b.pdf')} == {'b.pdf'}
    old = Session(owner='x', paper=PaperText.from_pages(P1, [Q1]).to_dict(), source='old.pdf')   # made before several papers existed
    assert list(service().papers_of(old)) == ['old.pdf'] and service().search_papers(old, 'Parkin')[0]['source'] == 'old.pdf'


# ── the REST surface ───────────────────────────────────────────────────────
@pytest.fixture
def env(tmp_path):
    pipe = FakePipeline()
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), pipe, upload_dir=str(tmp_path / 'up'), max_papers=4)
    return TestClient(app), pipe, tmp_path / 'up'


def post(c, **kw):
    return c.post('/api/llm/sessions', headers=AUTH('alice'), **kw)


def pdf(name):
    return (name, b'%PDF-1.4 fake', 'application/pdf')


def test_start_with_several_pmids_is_one_session_with_all_the_papers(env):
    c, pipe, _ = env
    r = post(c, json={'pmids': [P1, P2, P1], 'pmid': '33333333', 'focus': 'PINK1'})      # repeats dropped; `pmid` comes first, then `pmids`
    assert r.status_code == 202
    assert [p.pmid for p in pipe.calls[0].papers] == ['33333333', P1, P2] and pipe.calls[0].focus == 'PINK1'
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()
    assert [p['source'] for p in s['papers']] == ['PMID:33333333', f'PMID:{P1}', f'PMID:{P2}'] and s['pmid'] == '33333333'
    assert s['source'].startswith('3 papers: ')
    assert next(x for x in c.get('/api/llm/sessions', headers=AUTH('alice')).json() if x['id'] == s['id'])['n_papers'] == 3


def test_start_with_one_pmid_is_unchanged(env):
    c, pipe, _ = env
    r = post(c, json={'pmid': P1})
    assert pipe.calls[0].pmid == P1 and not pipe.calls[0].papers
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()
    assert [p['source'] for p in s['papers']] == [f'PMID:{P1}'] and s['source'] == P1


def test_start_validation(env):
    c, pipe, _ = env
    assert post(c, json={}).status_code == 422 and post(c, json={'pmids': []}).status_code == 422
    assert post(c, json={'pmids': [P1, 'abc']}).status_code == 422
    assert post(c, json={'pmids': [str(10000000 + i) for i in range(5)]}).status_code == 422          # over max_papers=4
    assert not pipe.calls
    assert parse_pmids('1234567, 7654321  1234567', ['7654321', '555555']) == ['1234567', '7654321', '555555']


def test_upload_several_pdfs_with_pmids_makes_one_session(env):
    c, pipe, up = env
    r = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'pmids': f'{P1} {P2}', 'focus': 'PINK1'},
               files=[('files', pdf('a.pdf')), ('files', pdf('b one.pdf'))])
    assert r.status_code == 202
    spec = pipe.calls[0]
    assert [(p.label, bool(p.pdf_path), p.pmid) for p in spec.papers] == [('a.pdf', True, None), ('b_one.pdf', True, None), (None, False, P1), (None, False, P2)]
    assert all(os.path.isfile(p.pdf_path) for p in spec.papers if p.pdf_path) and len(os.listdir(up / 'alice')) == 2
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()
    assert [p['kind'] for p in s['papers']] == ['pdf', 'pdf', 'pmid', 'pmid'] and s['papers'][1]['name'] == 'b_one.pdf'


def test_upload_one_pdf_is_unchanged_and_the_old_field_name_still_works(env):
    c, pipe, _ = env
    r = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'focus': 'PINK1'}, files={'file': pdf('only.pdf')})
    assert r.status_code == 202 and not pipe.calls[0].papers and pipe.calls[0].pdf_path.endswith('_only.pdf')
    assert c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()['source'] == 'only.pdf'


def test_a_refused_upload_leaves_no_files_and_starts_nothing(env):
    c, pipe, up = env
    bad = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), files=[('files', pdf('good.pdf')), ('files', ('bad.pdf', b'not a pdf', 'application/pdf'))])
    assert bad.status_code == 422 and 'bad.pdf' in bad.json()['detail'] and not pipe.calls
    assert not os.path.isdir(up / 'alice') or os.listdir(up / 'alice') == []                      # the good one was removed again
    assert c.post('/api/llm/sessions/upload', headers=AUTH('alice')).status_code == 422             # nothing sent
    too_many = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), files=[('files', pdf(f'{i}.pdf')) for i in range(5)])
    assert too_many.status_code == 422 and not pipe.calls


def test_search_endpoint_covers_papers_and_can_pick_one(env):
    c, pipe, _ = env
    sid = post(c, json={'pmids': [P1, P2]}).json()['sessionId']
    svc = c.app.state.service
    s = svc.store.get(sid, 'alice')
    s.paper_texts = {f'PMID:{P1}': PaperText.from_pages(P1, [Q1]).to_dict(), f'PMID:{P2}': PaperText.from_pages(P2, [Q2]).to_dict()}
    svc.store.save(s)
    hits = c.get(f'/api/llm/sessions/{sid}/paper/search', params={'q': 'ubiquitin'}, headers=AUTH('alice')).json()
    assert {h['source'] for h in hits} == {f'PMID:{P1}', f'PMID:{P2}'}
    one = c.get(f'/api/llm/sessions/{sid}/paper/search', params={'q': 'ubiquitin', 'source': f'PMID:{P2}'}, headers=AUTH('alice')).json()
    assert {h['source'] for h in one} == {f'PMID:{P2}'}
    assert c.get(f'/api/llm/sessions/{sid}/paper/search', params={'q': 'x', 'source': 'PMID:999'}, headers=AUTH('alice')).status_code == 404
