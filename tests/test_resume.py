"""Large runs survive interruptions: each paper's extraction ends with a status (complete / incomplete / no_text), a
resumed run reuses complete ones, a model call that keeps failing marks the paper incomplete instead of silently losing
its reactions, an incomplete run is not saved as a snapshot, and the command line records progress and finishes the
rest when run again."""
import json
import os
import subprocess
import sys
import types

import pytest
import requests

from curator_llm.cli import main as cli_main
from curator_llm.ports.pipeline import PaperRef, PipelineFailed, PipelineInput
from curator_llm.services.extraction import SubprocessExtractor
from curator_llm.services.snapshots import SnapshotPipeline, SnapshotStore
from tests.test_cli import FakePipeline, pdf


# ── run_extraction.py: status, retries, resume ─────────────────────────────
@pytest.fixture
def rx(monkeypatch, tmp_path):
    monkeypatch.setenv('ANTHROPIC_API_KEY', os.getenv('ANTHROPIC_API_KEY') or 'test-key')
    import run_extraction as m
    monkeypatch.setattr(m, 'RESULTS_DIR', str(tmp_path))
    monkeypatch.setattr(m, 'RETRY_WAITS', [0.0, 0.0])
    monkeypatch.setattr(m, 'WORKERS', 4)
    monkeypatch.setattr(m.fetcher, 'output_stem', lambda paper, gene=None: f'{gene}_{paper}')
    text = '\n\n'.join(f'Paragraph {i}. ' + 'WEE1 phosphorylates CDH1 in cells. ' * 120 for i in range(4))
    monkeypatch.setattr(m.fetcher, 'load_source', lambda paper, **kw: (f'PMID:{paper}', text, 'jats'))
    return m


class Client:
    """Fake Anthropic client: each call returns one reaction, or raises `fail` for the chunks listed in `bad`."""

    def __init__(self, fail=None, bad=lambda tail: False):
        self.calls, self.fail, self.bad = 0, fail, bad
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        tail = kw['messages'][0]['content'][1]['text']
        if self.fail is not None and self.bad(tail):
            raise self.fail
        text = json.dumps({'reactions': [{'name': f'r{self.calls}', 'input': ['CDH1'], 'output': ['pCDH1'],
                                          'evidence': [{'quote': 'WEE1 phosphorylates CDH1 in cells.', 'claim_origin': 'this_paper'}]}]})
        return types.SimpleNamespace(content=[types.SimpleNamespace(type='text', text=text)], stop_reason='end_turn',
                                     usage=types.SimpleNamespace(input_tokens=1, output_tokens=1,
                                                                 cache_read_input_tokens=0, cache_creation_input_tokens=0))


def status(m, paper):
    return m.read_status(m.output_path(paper, 'g'))


def test_a_complete_extraction_is_marked_and_a_resumed_run_reuses_it(rx, monkeypatch):
    c = Client()
    monkeypatch.setattr(rx, 'client', c)
    rx.extract_paper('111', gene='g', overwrite=True)
    st = status(rx, '111')
    assert st['state'] == 'complete' and st['chunks'] >= 2 and st['reactions'] >= 2 and st['fingerprint']
    n = c.calls
    rx.extract_paper('111', gene='g', overwrite=True, resume=True)
    assert c.calls == n                                                    # nothing extracted again
    rx.extract_paper('111', gene='g', overwrite=True, resume=True, review=True)
    assert c.calls > n                                                     # review changes the result: extracted again


def test_a_chunk_that_keeps_failing_makes_the_paper_incomplete_and_it_is_extracted_again(rx, monkeypatch):
    err = anthropic_error(rx, 'APIConnectionError')
    c = Client(fail=err, bad=lambda tail: 'Paragraph 0.' in tail.split('CURRENT CHUNK')[-1])
    monkeypatch.setattr(rx, 'client', c)
    rx.extract_paper('222', gene='g', overwrite=True)
    st = status(rx, '222')
    assert st['state'] == 'incomplete' and st['failed_chunks'] == [0] and 'run again' in st['reason']
    first = c.calls
    assert first > st['chunks']                                            # the failing chunk was tried several times
    monkeypatch.setattr(rx, 'client', Client())                            # the network is back
    rx.extract_paper('222', gene='g', overwrite=True, resume=True)
    assert status(rx, '222')['state'] == 'complete'


def test_a_bad_request_is_not_retried(rx, monkeypatch):
    class Bad(Exception):
        status_code = 400
    c = Client(fail=Bad('bad request'), bad=lambda tail: True)
    monkeypatch.setattr(rx, 'client', c)
    rx.extract_paper('333', gene='g', overwrite=True)
    st = status(rx, '333')
    assert st['state'] == 'incomplete' and c.calls == st['chunks']        # once per chunk, no retries


def test_no_text_and_network_failures_are_told_apart(rx, monkeypatch):
    monkeypatch.setattr(rx, 'client', Client())

    def no_body(paper, **kw):
        raise ValueError('PMC served no <body>')
    monkeypatch.setattr(rx.fetcher, 'load_source', no_body)
    with pytest.raises(ValueError):
        rx.extract_paper('444', gene='g', overwrite=True)
    assert status(rx, '444')['state'] == 'no_text'

    def offline(paper, **kw):
        raise requests.ConnectionError('no route to host')
    monkeypatch.setattr(rx.fetcher, 'load_source', offline)
    with pytest.raises(requests.RequestException):
        rx.extract_paper('555', gene='g', overwrite=True)
    assert status(rx, '555')['state'] == 'incomplete'


def test_an_old_status_is_removed_when_a_new_extraction_starts(rx, monkeypatch):
    monkeypatch.setattr(rx, 'client', Client())
    rx.extract_paper('666', gene='g', overwrite=True)
    assert status(rx, '666')['state'] == 'complete'

    def crash(paper, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(rx.fetcher, 'load_source', crash)
    with pytest.raises(KeyboardInterrupt):
        rx.extract_paper('666', gene='g', overwrite=True)                  # not resumed: starts over, then dies
    assert status(rx, '666') is None                                       # so nothing claims it is complete


def anthropic_error(m, name):
    cls = getattr(m.anthropic, name)
    try:
        return cls(request=requests.Request('POST', 'https://x').prepare())   # APIConnectionError(request=...)
    except TypeError:
        return cls('x')


# ── the extractor: incomplete papers stay out, --resume is passed ──────────
class Runner:
    """run_extraction.py writes each paper's file and status; `states` says how each paper ends."""

    def __init__(self, tmp, states, reused=()):
        self.tmp, self.states, self.reused, self.cmds = str(tmp), states, set(reused), []

    def __call__(self, cmd, cwd, timeout, on_line=None):
        self.cmds.append(cmd[1:])
        if os.path.basename(cmd[1]) == 'run_extraction.py':
            spec = cmd[2]
            base = os.path.join(self.tmp, f'g_{spec}')
            json.dump([{'source': f'PMID:{spec}', 'annotation_result': {'name': f'r{spec}', 'input': ['A'], 'output': ['B']}}],
                      open(base + '_extraction.json', 'w'))
            json.dump({'state': self.states[spec], 'reason': f'why {spec}', 'finished_at': 1.0 if spec in self.reused else 9e12},
                      open(base + '_extraction.status.json', 'w'))
            return subprocess.CompletedProcess(cmd, 0, '', '')
        out = cmd[cmd.index('--out') + 1]
        json.dump([r for p in cmd[2:cmd.index('--out')] for r in json.load(open(p))], open(out, 'w'))
        return subprocess.CompletedProcess(cmd, 0, '', '')


def test_an_incomplete_paper_is_kept_out_of_the_merge_and_reported(tmp_path):
    runner = Runner(tmp_path, {'1': 'complete', '2': 'incomplete', '3': 'no_text'}, reused=['1'])
    ex = SubprocessExtractor(root=str(tmp_path), results_dir=str(tmp_path), run=runner, stem=lambda s, g: f'{g}_{s}', resume=True)
    res = ex.extract_many([PaperRef(pmid='1'), PaperRef(pmid='2'), PaperRef(pmid='3')], 'g')
    assert all('--resume' in c for c in runner.cmds if c[0] == 'run_extraction.py')
    merge = next(c for c in runner.cmds if c[0] == 'run_merge.py')
    assert [os.path.basename(p) for p in merge[1:merge.index('--out')]] == ['g_1_extraction.json']
    assert res.paper_status['PMID:1'] == {'state': 'complete', 'reason': 'why 1', 'reused': True}
    assert res.paper_status['PMID:2']['state'] == 'incomplete' and not res.complete
    assert any('PMID:2' in w and 'run again' in w for w in res.warnings)


def test_the_service_extracts_afresh_and_a_single_incomplete_paper_fails(tmp_path):
    runner = Runner(tmp_path, {'7': 'incomplete'})
    ex = SubprocessExtractor(root=str(tmp_path), results_dir=str(tmp_path), run=runner, stem=lambda s, g: f'{g}_{s}')
    res = ex.extract('7', None, 'g')
    assert '--resume' not in runner.cmds[0] and not res.ok and res.error.startswith('incomplete, run again')
    assert res.paper_status['PMID:7']['state'] == 'incomplete'


def test_an_incomplete_run_is_not_saved_as_a_snapshot(tmp_path):
    class Pipe:
        def __init__(self, complete):
            self.complete = complete

        def run(self, spec, report):
            r = FakePipeline().run(spec, report)
            r.complete = self.complete
            return r
    store = SnapshotStore(str(tmp_path))
    SnapshotPipeline(Pipe(False), store).run(PipelineInput(pmid='1'), lambda m: None)
    assert not os.listdir(tmp_path)
    SnapshotPipeline(Pipe(True), store).run(PipelineInput(pmid='1'), lambda m: None)
    assert os.listdir(tmp_path)


# ── the command line: progress file, partial runs finished by running again ─
class StepPipeline(FakePipeline):
    """Ends `states[n]` on the n-th run: 'complete', 'partial', or 'fail'."""

    def __init__(self, states):
        super().__init__()
        self.states = list(states)

    def run(self, spec, report):
        state = self.states.pop(0)
        refs = spec.refs()
        if state == 'fail':
            self.calls.append(spec)
            raise PipelineFailed('nothing extracted', {refs[0].source: {'state': 'incomplete', 'reason': 'offline'}})
        r = super().run(spec, report)
        r.paper_status = {x.source: {'state': 'complete', 'reason': '', 'reused': state == 'complete' and len(self.calls) > 1}
                          for x in refs}
        if state == 'partial':
            r.paper_status[refs[-1].source] = {'state': 'incomplete', 'reason': 'the model could not be reached'}
            r.complete = False
        return r


def run(tmp_path, argv, pipe):
    lines = []
    code = cli_main([*argv, '--out', str(tmp_path / 'out')], pipeline=pipe, say=lines.append)
    return code, lines, (json.load(open(tmp_path / 'out' / 'many.progress.json'))
                         if os.path.exists(tmp_path / 'out' / 'many.progress.json') else None)


def test_a_partial_run_is_recorded_and_finished_by_running_the_same_command_again(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    argv = [pdf(folder, 'a.pdf'), pdf(folder, 'b.pdf'), '--name', 'many']
    pipe = StepPipeline(['partial', 'complete'])
    code, lines, prog = run(tmp_path, argv, pipe)
    assert code == 1 and prog['status'] == 'partial' and prog['counts'] == {'complete': 1, 'incomplete': 1, 'no_text': 0}
    assert any('PARTIAL' in m and 'Run the same command again' in m for m in lines)
    code, lines, prog = run(tmp_path, argv, pipe)                          # runs again: not skipped
    assert code == 0 and prog['status'] == 'complete' and len(pipe.calls) == 2
    assert any('resuming' in m for m in lines)
    code, lines, _ = run(tmp_path, argv, pipe)                             # now complete: skipped
    assert code == 0 and len(pipe.calls) == 2 and any('already done' in m for m in lines)


def test_a_failed_run_records_why_per_paper_and_is_tried_again(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    argv = [pdf(folder, 'a.pdf'), '--name', 'many']
    pipe = StepPipeline(['fail', 'complete'])
    code, _, prog = run(tmp_path, argv, pipe)
    assert code == 1 and prog['status'] == 'failed' and prog['papers'][0]['state'] == 'incomplete'
    assert prog['papers'][0]['reason'] == 'offline'
    code, _, prog = run(tmp_path, argv, pipe)
    assert code == 0 and prog['status'] == 'complete'


def test_force_redoes_a_complete_run(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    argv = [pdf(folder, 'a.pdf'), '--name', 'many']
    pipe = StepPipeline(['complete', 'complete'])
    run(tmp_path, argv, pipe)
    run(tmp_path, argv + ['--force'], pipe)
    assert len(pipe.calls) == 2


def test_a_paper_read_completely_with_no_reactions_is_finished_not_failed(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    argv = [pdf(folder, 'a.pdf'), '--name', 'many']

    class Empty(FakePipeline):
        def run(self, spec, report):
            self.calls.append(spec)
            raise PipelineFailed('no reactions', {spec.refs()[0].source: {'state': 'complete', 'reason': ''}})
    pipe = Empty()
    code, lines, prog = run(tmp_path, argv, pipe)
    assert code == 0 and prog['status'] == 'no_reactions' and any('no reactions' in m for m in lines)
    run(tmp_path, argv, pipe)
    assert len(pipe.calls) == 1                                            # not tried again
