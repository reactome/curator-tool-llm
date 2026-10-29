import json
import os
import subprocess
import sys

import pytest

from curator_llm.services.extraction import StepProgress, SubprocessExtractor, _subprocess_run


def feed_all(step, lines):
    p, out = StepProgress(step), []
    for ln in lines:
        m = p.feed(ln)
        if m:
            out.append(m)
    return out


def test_extraction_progress_counts_finished_chunks_even_when_they_finish_out_of_order():
    msgs = feed_all('extraction', [
        '[setup] PINK1.pdf: Results=3,443w (via heuristic) -> 20 chunks',
        '  chunk 5/20: 2 reaction(s)  |  1 call(s), tokens in 1 / out 1',
        '  chunk 2/20: FAILED  |  2 call(s)',
        '  chunk 9/20: 0 reaction(s)  |  3 call(s)',
        '[done] PINK1.pdf: 20 chunks'])
    assert msgs == ['extracting reactions: 0/20 chunks', 'extracting reactions: 1/20 chunks',
                    'extracting reactions: 2/20 chunks', 'extracting reactions: 3/20 chunks']


def test_merge_progress_follows_the_stages():
    msgs = feed_all('merge', [
        '[setup] 26 entries -> 25 unique signatures',
        '[merge] pre-filter: 137/300 pairs can be duplicates; 163 skipped without a call',
        '[merge] judge over 25 reactions (137 pairs), 12 concurrent...',
        '  judged ~100/137 pairs | 12s | 100 call(s)',
        '[merge] judged 137 pairs in 21s | 23 same, 114 different',
        '[revote] 30 verdict(s) under confidence 0.9; 2 more votes each',
        '[cluster] 25 -> 14 conflict-free cluster(s)',
        '[validate] checking 2 cluster(s)',
        '[consolidate] deduplicating subsections of 6/14'])
    assert msgs == ['merging reactions: 25 distinct reactions to compare',
                    'merging reactions: comparing 137 candidate pairs',
                    'merging reactions: compared 100/137 pairs',
                    'merging reactions: re-checking uncertain pairs',
                    'merging reactions: grouping duplicates', 'merging reactions: checking the groups',
                    'merging reactions: combining duplicate details']


def test_unrelated_lines_say_nothing():
    assert feed_all('extraction', ['warning: fitz deprecated', '', '[run] extracting 1 paper(s)']) == []


def test_subprocess_run_streams_lines_while_the_step_runs():
    seen = []
    code = "import sys,time\nfor i in range(3):\n    print('line', i, flush=True)\n    time.sleep(0.05)\nprint('err', file=sys.stderr)"
    proc = _subprocess_run([sys.executable, '-c', code], '.', 20, on_line=seen.append)
    assert seen[:3] == ['line 0', 'line 1', 'line 2'] and 'err' in seen
    assert proc.returncode == 0 and 'line 2' in proc.stdout and 'err' in proc.stdout     # stderr is merged in


def test_subprocess_run_reports_the_exit_code_and_survives_a_failing_callback():
    def boom(_):
        raise RuntimeError('cosmetic')
    proc = _subprocess_run([sys.executable, '-c', "print('x'); raise SystemExit(3)"], '.', 20, on_line=boom)
    assert proc.returncode == 3 and 'x' in proc.stdout


def test_subprocess_run_times_out():
    with pytest.raises(subprocess.TimeoutExpired):
        _subprocess_run([sys.executable, '-c', 'import time; time.sleep(30)'], '.', 1)


def test_extractor_forwards_progress_to_the_report_callback(tmp_path):
    rx = [{'source': 'PMID:1', 'annotation_result': {'name': 'a'}}]

    def run(cmd, cwd, timeout, on_line=None):
        name = os.path.basename(cmd[1])
        if name == 'run_extraction.py':
            json.dump(rx, open(tmp_path / 'g_pmid1_extraction.json', 'w'))
            for ln in ['[setup] x: Results=1w (via heuristic) -> 2 chunks', '  chunk 1/2: 1 reaction(s)', '  chunk 2/2: 0 reaction(s)']:
                on_line and on_line(ln)
        else:
            json.dump(rx, open(tmp_path / 'g_pmid1_merged.json', 'w'))
            on_line and on_line('[cluster] 1 -> 1 conflict-free cluster(s)')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    got = []
    ex = SubprocessExtractor(root=str(tmp_path), results_dir=str(tmp_path), run=run, stem=lambda s, g: f'{g}_pmid{s}')
    res = ex.extract('1', None, 'g', report=got.append)
    assert res.ok and got == ['extracting reactions: 0/2 chunks', 'extracting reactions: 1/2 chunks',
                              'extracting reactions: 2/2 chunks', 'merging reactions: grouping duplicates']


def test_pipeline_passes_report_only_to_extractors_that_accept_it():
    from curator_llm.ports.pipeline import PipelineInput
    from curator_llm.services.pipeline_default import DefaultPipeline
    from tests.fakes.ports import FakeInstanceLookup

    class Old:
        def extract(self, pmid, pdf_path, gene):
            return 'old'

    class New:
        def extract(self, pmid, pdf_path, gene, report=None):
            return report

    spec = PipelineInput(pmid='1')
    sentinel = print
    assert DefaultPipeline(FakeInstanceLookup([]), extractor=Old())._extract(spec, 'g', sentinel) == 'old'
    assert DefaultPipeline(FakeInstanceLookup([]), extractor=New())._extract(spec, 'g', sentinel) is sentinel
