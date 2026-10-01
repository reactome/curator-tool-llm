import json
import os

import pytest
from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import EwasSpec, ReactionSpec, ReactomeDraft
from curator_llm.models.session import ExistingMatch, Issue
from curator_llm.ports.pipeline import PipelineInput, PipelineResult
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.paper_text import PaperText
from curator_llm.services.snapshots import SnapshotPipeline, SnapshotStore, snapshot_key
from tests.fakes.ports import FakeAuthProvider, FakeInstanceLookup, SyncJobRunner


def result(name='PINK1 phosphorylates Ub') -> PipelineResult:
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7', needs_resolution=['compartment'])}
    d.reactions = [ReactionSpec(key='r0', name=name, inputs=['pink1'], evidence_ids=['ev-001'], pmids=['24751536'])]
    paper = PaperText.from_pages('24751536', ['Results\nTcPINK1 phosphorylates Ub (Fig. 3 B).\n'])
    return PipelineResult(d, [Evidence(id='ev-001', quote='TcPINK1 phosphorylates Ub', page=1, supports=['catalystActivity'])],
                          [Issue(source='builder', code='note', message='m', severity='info')], paper.to_dict(),
                          [ExistingMatch(reaction_key='r0', db_id=9, display_name='x', st_id='R-HSA-9', level='same', score=0.9)])


class Inner:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def run(self, spec, report):
        self.calls += 1
        report('extracting')
        if self.fail:
            raise RuntimeError('PMC has no full text')
        return result()


def dump(r: PipelineResult):
    # compared in JSON form: PaperText offsets are tuples in memory and lists once saved, which from_dict treats as the same
    return json.loads(json.dumps((r.draft.model_dump(mode='json'), [e.model_dump(mode='json') for e in r.evidence],
                                  [i.model_dump(mode='json') for i in r.issues], r.paper,
                                  [m.model_dump(mode='json') for m in r.existing])))


class TestKey:
    def test_pmid_and_focus(self):
        assert snapshot_key(PipelineInput(pmid='24751536')) == 'pmid-24751536'
        assert snapshot_key(PipelineInput(pmid=' 24751536 ', focus=' pink1 ')) == 'pmid-24751536__PINK1'
        assert snapshot_key(PipelineInput(pmid='1', focus='PINK1')) != snapshot_key(PipelineInput(pmid='1', focus='PRKN'))

    def test_pdf_is_keyed_by_content_not_name_and_the_focus_is_made_safe(self, tmp_path):
        a, b, c = tmp_path / 'a.pdf', tmp_path / 'renamed.pdf', tmp_path / 'c.pdf'
        a.write_bytes(b'%PDF-1 same'); b.write_bytes(b'%PDF-1 same'); c.write_bytes(b'%PDF-1 other')
        assert snapshot_key(PipelineInput(pdf_path=str(a))) == snapshot_key(PipelineInput(pdf_path=str(b)))
        assert snapshot_key(PipelineInput(pdf_path=str(a))) != snapshot_key(PipelineInput(pdf_path=str(c)))
        assert snapshot_key(PipelineInput(pdf_path=str(a), focus='../../etc/passwd')).endswith('__ETCPASSWD')   # no path tricks

    def test_needs_something_to_key_on(self):
        with pytest.raises(ValueError):
            snapshot_key(PipelineInput())


class TestStore:
    def test_round_trip_keeps_everything_and_leaves_no_temp_file(self, tmp_path):
        store = SnapshotStore(str(tmp_path / 'nested' / 'snaps'))
        path = store.save('k', result())
        assert os.path.isfile(path) and os.listdir(os.path.dirname(path)) == ['k.json']
        assert dump(store.load('k')) == dump(result())

    def test_missing_corrupt_old_and_truncated_files_read_as_no_snapshot(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        assert store.load('absent') is None
        (tmp_path / 'bad.json').write_text('{not json')
        assert store.load('bad') is None
        (tmp_path / 'old.json').write_text(json.dumps({'version': 0}))
        assert store.load('old') is None
        store.save('t', result())
        doc = json.loads((tmp_path / 't.json').read_text())
        del doc['draft']['participants']                                   # a snapshot that no longer fits the model
        doc['draft']['reactions'][0]['inputs'] = 'not a list'
        (tmp_path / 't.json').write_text(json.dumps(doc))
        assert store.load('t') is None


class TestPipeline:
    spec = PipelineInput(pmid='24751536', focus='PINK1')

    def make(self, tmp_path, mode, inner=None):
        inner = inner or Inner()
        return inner, SnapshotStore(str(tmp_path)), SnapshotPipeline(inner, SnapshotStore(str(tmp_path)), mode)

    def test_record_saves_and_still_always_runs_the_pipeline(self, tmp_path):
        inner, store, p = self.make(tmp_path, 'record')
        p.run(self.spec, lambda m: None)
        p.run(self.spec, lambda m: None)
        assert inner.calls == 2 and store.load(snapshot_key(self.spec)) is not None

    def test_replay_runs_once_then_serves_the_saved_result_without_the_models(self, tmp_path):
        inner, _, p = self.make(tmp_path, 'replay')
        steps = []
        first = p.run(self.spec, steps.append)
        second = p.run(self.spec, steps.append)
        assert inner.calls == 1
        assert dump(first) == dump(second)
        assert any('replay mode' in s and 'no models called' in s for s in steps)

    def test_replay_does_not_mix_up_papers_or_focus(self, tmp_path):
        inner, _, p = self.make(tmp_path, 'replay')
        p.run(self.spec, lambda m: None)
        p.run(PipelineInput(pmid='24751536', focus='PRKN'), lambda m: None)
        p.run(PipelineInput(pmid='999999'), lambda m: None)
        assert inner.calls == 3

    def test_replay_falls_back_to_the_pipeline_and_repairs_an_unreadable_snapshot(self, tmp_path):
        inner, store, p = self.make(tmp_path, 'replay')
        (tmp_path / f'{snapshot_key(self.spec)}.json').write_text('garbage')
        p.run(self.spec, lambda m: None)
        assert inner.calls == 1 and store.load(snapshot_key(self.spec)) is not None

    def test_off_neither_saves_nor_replays(self, tmp_path):
        inner, _, p = self.make(tmp_path, 'off')
        p.run(self.spec, lambda m: None)
        p.run(self.spec, lambda m: None)
        assert inner.calls == 2 and os.listdir(tmp_path) == []

    def test_a_failed_run_propagates_and_saves_nothing(self, tmp_path):
        inner, _, p = self.make(tmp_path, 'replay', Inner(fail=True))
        with pytest.raises(RuntimeError, match='no full text'):
            p.run(self.spec, lambda m: None)
        assert os.listdir(tmp_path) == []

    def test_failing_to_save_never_fails_the_annotation(self, tmp_path):
        class BadStore(SnapshotStore):
            def save(self, key, result):
                raise OSError('disk full')
        p = SnapshotPipeline(Inner(), BadStore(str(tmp_path)), 'record')
        assert p.run(self.spec, lambda m: None).draft.reactions

    def test_an_unknown_mode_is_refused_at_start_up(self, tmp_path):
        with pytest.raises(ValueError, match='SNAPSHOT_MODE'):
            SnapshotPipeline(Inner(), SnapshotStore(str(tmp_path)), 'replya')

    def test_a_replayed_session_becomes_ready_through_the_api_without_calling_the_pipeline(self, tmp_path):
        SnapshotStore(str(tmp_path)).save(snapshot_key(self.spec), result())
        inner = Inner()
        app = create_app(FakeAuthProvider({'t': ('u', 'curator')}), InMemorySessionStore(), SyncJobRunner(),
                         SnapshotPipeline(inner, SnapshotStore(str(tmp_path)), 'replay'))
        c = TestClient(app)
        H = {'Authorization': 'Bearer t'}
        sid = c.post('/api/llm/sessions', json={'pmid': '24751536', 'focus': 'PINK1'}, headers=H).json()['sessionId']
        s = c.get(f'/api/llm/sessions/{sid}', headers=H).json()
        assert inner.calls == 0 and s['status'] == 'ready' and s['n_reactions'] == 1
        assert c.get(f'/api/llm/sessions/{sid}/export', headers=H).json()['newInstances']
        assert c.get(f'/api/llm/sessions/{sid}/existing', headers=H).json()[0]['level'] == 'same'


class TestAssemble:
    def test_a_supplied_draft_means_no_model_is_called(self):
        class Boom:
            def with_structured_output(self, *a):
                raise AssertionError('the draft builder must not be called')
        pipe = DefaultPipeline(FakeInstanceLookup([]), model=Boom())
        paper = PaperText.from_pages('1', ['Results\nTcPINK1 phosphorylates Ub.\n'])
        reactions = [{'source': 'PMID:1', 'annotation_result': {'name': 'a', 'evidence': ['TcPINK1 phosphorylates Ub']}}]
        steps = []
        out = pipe.assemble(PipelineInput(pmid='1'), reactions, lambda s: paper, {'main': paper}, steps.append, draft=result().draft)
        assert out.draft.reactions[0].key == 'r0' and out.evidence[0].verified.value == 'exact'
        assert 'building the Reactome draft' not in steps and 'resolving identifiers' in steps
