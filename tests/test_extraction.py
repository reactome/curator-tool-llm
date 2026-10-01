import json
import os
import subprocess

from curator_llm.services.extraction import SubprocessExtractor, parse_usage

RX = [{'source': 'PMID:1', 'annotation_result': {'name': 'a'}}, {'source': 'PMID:1', 'annotation_result': {'name': 'b'}}]


class Runner:
    """Fake subprocess runner that writes the files each script would write."""

    def __init__(self, tmp, fail=None, merged=None, timeout=None):
        self.tmp, self.fail, self.merged, self.timeout, self.cmds = tmp, fail, merged, timeout, []

    def __call__(self, cmd, cwd, timeout):
        self.cmds.append(cmd[1:])
        name = os.path.basename(cmd[1])
        if self.timeout == name:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if self.fail == name:
            return subprocess.CompletedProcess(cmd, 3, '', 'Traceback: kaboom')
        out = ''
        if name == 'run_extraction.py':
            json.dump(RX, open(os.path.join(self.tmp, 'g_pmid1_extraction.json'), 'w'))
            out = '[done] 20 call(s), tokens in 1,000 / out 200, cache r5/w6\n'
        elif name == 'run_merge.py':
            json.dump(self.merged if self.merged is not None else RX[:1], open(os.path.join(os.path.dirname(cmd[2]), os.path.basename(cmd[2]).replace('_extraction', '_merged')), 'w'))
            out = '[usage] 231 call(s), tokens in 390,872 / out 23,737\n'
        elif name == 'run_review.py':
            open(os.path.join(os.path.dirname(cmd[2]), os.path.basename(cmd[2]).replace('_merged.json', '_review.md')), 'w').write('SCORE: 7.5\n')
            out = '[usage] tokens in 10 / out 5\n'
        return subprocess.CompletedProcess(cmd, 0, out, '')


def make(tmp, **kw):
    runner = Runner(str(tmp), **{k: kw.pop(k) for k in ('fail', 'merged', 'timeout') if k in kw})
    return SubprocessExtractor(root=str(tmp), results_dir=str(tmp), run=runner, stem=lambda spec, gene: f'{gene}_pmid{spec}', **kw), runner


def test_usage_parsing_takes_the_last_running_total():
    u = parse_usage('3 call(s), tokens in 10 / out 5\n20 call(s), tokens in 1,000 / out 200, cache r5/w6\n')
    assert u == {'calls': 20, 'input': 1000, 'output': 200, 'cache_read': 5, 'cache_write': 6}


def test_happy_path_returns_merged_reactions_and_summed_usage(tmp_path):
    ex, runner = make(tmp_path, review=True)
    res = ex.extract('1', None, 'g')
    assert res.ok and res.n_extracted == 2 and res.n_merged == 1 and len(res.reactions) == 1
    assert [c[0] for c in runner.cmds] == ['run_extraction.py', 'run_merge.py', 'run_review.py']
    assert runner.cmds[0] == ['run_extraction.py', '1', '--gene', 'g', '--overwrite']
    assert res.review_score == 7.5 and res.review_path.endswith('_review.md')
    assert res.usage['input'] == 1000 + 390872 + 10 and res.error is None and res.warnings == []


def test_pdf_uses_the_localpdf_tag_and_skips_review_by_default(tmp_path):
    ex, runner = make(tmp_path)
    ex._stem = lambda spec, gene: 'g_pmid1'
    res = ex.extract(None, '/x/paper.pdf', 'g')
    assert runner.cmds[0] == ['run_extraction.py', '/x/paper.pdf', '--gene', 'g', '--overwrite', '--tag', 'localpdf']
    assert res.extraction_path.endswith('g_pmid1_localpdf_extraction.json') or res.extraction_path.endswith('_localpdf_extraction.json')
    assert 'run_review.py' not in [c[0] for c in runner.cmds]


def test_merge_failure_degrades_to_unmerged_reactions_with_a_warning(tmp_path):
    ex, _ = make(tmp_path, fail='run_merge.py')
    res = ex.extract('1', None, 'g')
    assert res.ok and len(res.reactions) == 2 and res.n_merged == 0
    assert any('merge failed' in w and 'kaboom' in w for w in res.warnings)


def test_review_failure_is_only_a_warning(tmp_path):
    ex, _ = make(tmp_path, fail='run_review.py', review=True)
    res = ex.extract('1', None, 'g')
    assert res.ok and res.review_score is None and any('review failed' in w for w in res.warnings)


def test_extraction_failure_timeout_and_empty_are_reported_not_raised(tmp_path):
    res = make(tmp_path, fail='run_extraction.py')[0].extract('1', None, 'g')
    assert not res.ok and 'extraction failed (exit 3)' in res.error and 'kaboom' in res.error
    res = make(tmp_path, timeout='run_extraction.py')[0].extract('1', None, 'g')
    assert not res.ok and 'timed out' in res.error
    assert not make(tmp_path)[0].extract(None, None, 'g').ok


def test_empty_merge_output_falls_back_to_unmerged(tmp_path):
    ex, _ = make(tmp_path, merged=[])
    res = ex.extract('1', None, 'g')
    assert res.ok and len(res.reactions) == 2 and any('merge wrote no reactions' in w for w in res.warnings)
