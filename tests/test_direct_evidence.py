"""Direct evidence: a research paper keeps only the reactions its own results show; a review keeps what it cites."""
import json
import os
import subprocess

from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.cli import main
from curator_llm.ports.extractor import ExtractionResult
from curator_llm.ports.pipeline import PaperRef, PipelineInput
from curator_llm.services.direct_evidence import is_cited_only, split_cited_only
from curator_llm.services.extraction import SubprocessExtractor
from curator_llm.services.issues import issues_from_notes
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.snapshots import snapshot_key
from tests.fakes.ports import FakeAuthProvider, FakeInstanceLookup, SyncJobRunner
from tests.test_cli import FakePipeline as CliPipeline
from tests.test_session_api import AUTH, FakePipeline, TOKENS


def rx(name, *origins):
    return {'source': 'PMID:1', 'annotation_result': {
        'name': name, 'evidence': [f'q{i}' for i in range(len(origins))],
        'evidence_details': {f'q{i}': {'claim_origin': o} for i, o in enumerate(origins)}}}


# ── the rule ───────────────────────────────────────────────────────────────
def test_only_a_reaction_whose_every_quote_is_cited_is_cited_only():
    assert is_cited_only(rx('a', 'cited', 'cited')['annotation_result'])
    assert not is_cited_only(rx('b', 'cited', 'this_paper')['annotation_result'])       # one own result is enough
    assert not is_cited_only(rx('c', 'this_paper')['annotation_result'])
    assert not is_cited_only({'name': 'no quotes'})                                     # fails open
    assert not is_cited_only(rx('d', None)['annotation_result'])                        # untagged counts as this paper
    # the extractor's raw shape (before split_evidence) is read too
    assert is_cited_only({'evidence': [{'quote': 'x', 'claim_origin': 'cited'}]})
    kept, left = split_cited_only([rx('a', 'cited'), rx('b', 'this_paper'), rx('c', 'cited', 'cited')])
    assert [r['annotation_result']['name'] for r in kept] == ['b'] and [r['annotation_result']['name'] for r in left] == ['a', 'c']


def test_the_note_about_what_was_left_out_is_an_info_issue():
    [i] = issues_from_notes('pipeline', ['left out 2 reaction(s) the paper only cites from earlier work (...): a; b'])
    assert (i.code, i.severity) == ('cited_only_left_out', 'info')


# ── the extractor ──────────────────────────────────────────────────────────
class Runner:
    """run_extraction.py writes the kept reactions and, as the real script does, the ones it left out alongside."""

    def __init__(self, tmp, kept, left_out):
        self.tmp, self.kept, self.left_out, self.cmds = str(tmp), kept, left_out, []

    def __call__(self, cmd, cwd, timeout, on_line=None):
        self.cmds.append(cmd[1:])
        name = os.path.basename(cmd[1])
        if name == 'run_extraction.py':
            label = f'g_{os.path.basename(cmd[2])}' + ('_localpdf' if '--tag' in cmd else '')
            json.dump(self.kept, open(os.path.join(self.tmp, f'{label}_extraction.json'), 'w'))
            json.dump([] if '--review' in cmd else self.left_out, open(os.path.join(self.tmp, f'{label}_cited_only.json'), 'w'))
        else:
            out = cmd[cmd.index('--out') + 1] if '--out' in cmd else cmd[2].replace('_extraction', '_merged')
            json.dump(self.kept, open(out, 'w'))
        return subprocess.CompletedProcess(cmd, 0, '', '')


def make(tmp, kept, left_out):
    runner = Runner(tmp, kept, left_out)
    return SubprocessExtractor(root=str(tmp), results_dir=str(tmp), run=runner, stem=lambda s, g: f'{g}_{os.path.basename(s)}',
                               paper_workers=2), runner


def test_a_research_paper_is_extracted_without_review_and_what_it_left_out_becomes_a_warning(tmp_path):
    ex, runner = make(tmp_path, [rx('own', 'this_paper')], [rx('WEE1 phosphorylates CDK1', 'cited')])
    res = ex.extract('1', None, 'g')
    assert '--review' not in runner.cmds[0] and res.ok
    assert len(res.warnings) == 1 and 'left out 1 reaction(s)' in res.warnings[0] and 'WEE1 phosphorylates CDK1' in res.warnings[0]


def test_a_review_is_extracted_with_review(tmp_path):
    ex, runner = make(tmp_path, [rx('own', 'this_paper')], [rx('cited', 'cited')])
    res = ex.extract('1', None, 'g', review=True)
    assert runner.cmds[0][-1] == '--review' and res.ok and res.warnings == []


def test_a_paper_whose_every_reaction_is_cited_fails_and_says_why(tmp_path):
    ex, _ = make(tmp_path, [], [rx('cited', 'cited')])
    res = ex.extract('1', None, 'g')
    assert not res.ok and 'only cites' in res.error and 'review article' in res.error


def test_several_papers_each_say_whether_they_are_a_review(tmp_path):
    ex, runner = make(tmp_path, [rx('own', 'this_paper')], [rx('cited', 'cited')])
    res = ex.extract_many([PaperRef(pmid='1'), PaperRef(pmid='2', review=True)], 'g')
    by_paper = {c[1]: c for c in runner.cmds if c[0] == 'run_extraction.py'}
    assert '--review' not in by_paper['1'] and '--review' in by_paper['2']
    assert res.ok and len(res.warnings) == 1 and res.warnings[0].startswith('PMID:1: left out 1 reaction(s)')


# ── the pipeline, the saved results, the API and the command line ──────────
class OnePaperExtractor:
    def __init__(self, takes_review=True):
        self.calls = []
        if not takes_review:
            self.extract = self._extract_plain

    def extract(self, pmid, pdf_path, gene, report=None, review=False):
        self.calls.append(review)
        return ExtractionResult(error='stop here')

    def _extract_plain(self, pmid, pdf_path, gene):
        self.calls.append('plain')
        return ExtractionResult(error='stop here')


def test_the_pipeline_passes_review_to_an_extractor_that_takes_it_and_not_to_one_that_does_not():
    for takes, expect in ((True, [True]), (False, ['plain'])):
        ex = OnePaperExtractor(takes)
        DefaultPipeline(FakeInstanceLookup(), extractor=ex)._extract(PipelineInput(pmid='1', review=True), 'g', lambda m: None)
        assert ex.calls == expect


def test_review_marks_every_paper_and_has_its_own_saved_result():
    refs = PipelineInput(papers=[PaperRef(pmid='1'), PaperRef(pmid='2')], review=True).refs()
    assert all(r.review for r in refs)
    assert PipelineInput(pmid='1', review=True).refs()[0].review and not PipelineInput(pmid='1').refs()[0].review
    assert snapshot_key(PipelineInput(pmid='1', focus='X')) == 'pmid-1__X'                       # research: unchanged
    assert snapshot_key(PipelineInput(pmid='1', focus='X', review=True)) == 'pmid-1+review__X'
    assert snapshot_key(PipelineInput(papers=[PaperRef(pmid='1'), PaperRef(pmid='2')])) != \
        snapshot_key(PipelineInput(papers=[PaperRef(pmid='1'), PaperRef(pmid='2', review=True)]))


def test_the_api_takes_review_for_pmids_and_uploads_and_the_session_shows_it(tmp_path):
    pipe = FakePipeline()
    c = TestClient(create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), pipe,
                              upload_dir=str(tmp_path / 'up')))
    r = c.post('/api/llm/sessions', headers=AUTH('alice'), json={'pmid': '11111111', 'review': True})
    assert r.status_code == 202 and pipe.calls[-1].review
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()
    assert s['papers'][0]['review'] is True
    c.post('/api/llm/sessions', headers=AUTH('alice'), json={'pmid': '11111111'})
    assert not pipe.calls[-1].review                                                          # research by default
    r = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'review': 'true'},
               files={'file': ('a.pdf', b'%PDF-1.4 fake', 'application/pdf')})
    assert r.status_code == 202 and pipe.calls[-1].review


def test_the_command_line_takes_review_and_names_a_single_paper_with_name(tmp_path):
    p = tmp_path / 'paper.pdf'
    p.write_bytes(b'%PDF-1.4 fake')
    pipe = CliPipeline()
    code = main([str(p), '--review', '--name', 'mine', '--out', str(tmp_path / 'out')], pipeline=pipe, say=lambda m: None)
    assert code == 0 and pipe.calls[0].refs()[0].review
    assert os.path.isfile(tmp_path / 'out' / 'mine.staged_instances.json')
    session = json.load(open(tmp_path / 'out' / 'mine.session.json'))
    assert session['papers'][0]['review'] is True
