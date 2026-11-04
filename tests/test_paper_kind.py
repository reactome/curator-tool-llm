"""Review or research: a PMID's kind comes from PubMed, a PDF is research unless ticked, an explicit setting wins."""
from dataclasses import replace

from fastapi.testclient import TestClient

from curator_llm.adapters.pubmed import PubMedClient
from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.cli import main
from curator_llm.ports.pipeline import PaperRef, PipelineInput
from curator_llm.services.paper_kind import KindPipeline, decide
from curator_llm.services.snapshots import snapshot_key
from tests.fakes.ports import FakeAuthProvider, FakePubMed, SyncJobRunner
from tests.test_cli import FakePipeline as CliPipeline
from tests.test_session_api import AUTH, FakePipeline, TOKENS

RESEARCH, REVIEW = '24368734', '25670789'
PUBMED = {RESEARCH: ['Journal Article', 'Research Support, N.I.H., Extramural'],
          REVIEW: ['Journal Article', "Research Support, Non-U.S. Gov't", 'Review'],
          '1': ['Journal Article', 'Systematic Review'], '2': ['Meta-Analysis']}


def kinds(spec):
    return [r.review for r in spec.refs()]


# ── deciding ───────────────────────────────────────────────────────────────
def test_a_pmid_is_a_review_when_pubmed_says_so_and_a_pdf_is_research():
    pm = FakePubMed(PUBMED)
    spec, notes = decide(PipelineInput(papers=[PaperRef(pmid=RESEARCH), PaperRef(pmid=REVIEW), PaperRef(pdf_path='/x/a.pdf'),
                                               PaperRef(pmid='1'), PaperRef(pmid='2')]), pm)
    assert kinds(spec) == [False, True, False, True, True]
    assert pm.calls == [[RESEARCH, REVIEW, '1', '2']]                  # one request, PMIDs only
    assert len(notes) == 3 and notes[0].startswith(f'PMID {REVIEW} is a review article (PubMed publication type: Review)')


def test_an_explicit_setting_is_kept_and_pubmed_is_not_asked_about_it():
    pm = FakePubMed(PUBMED)
    spec, notes = decide(PipelineInput(papers=[PaperRef(pmid=REVIEW, review=False), PaperRef(pdf_path='/x/r.pdf', review=True)]), pm)
    assert kinds(spec) == [False, True] and pm.calls == [] and notes == []
    spec, _ = decide(PipelineInput(pmid=RESEARCH, review=True), pm)      # the request-wide setting fills the open ones
    assert kinds(spec) == [True] and pm.calls == []


def test_one_paper_keeps_its_shape():
    spec, _ = decide(PipelineInput(pmid=REVIEW, focus='X'), FakePubMed(PUBMED))
    assert spec.pmid == REVIEW and not spec.papers and spec.review is True and spec.focus == 'X'


def test_when_pubmed_cannot_answer_the_paper_is_research_and_a_note_says_so():
    for pm in (FakePubMed(fail=True), FakePubMed({}), None):
        spec, notes = decide(PipelineInput(pmid=REVIEW), pm)
        assert kinds(spec) == [False] and len(notes) == 1
        assert 'could not read its publication type' in notes[0] and 'treated as a research paper' in notes[0]


class Recorder:
    def __init__(self):
        self.specs = []

    def run(self, spec, report):
        self.specs.append(spec)
        return FakePipeline().run(replace(spec, pmid=spec.pmid or '1'), report)   # that fake builds one-paper drafts only


def test_the_wrapper_decides_before_the_saved_result_key_and_reports_the_decision():
    inner = Recorder()
    result = KindPipeline(inner, FakePubMed(PUBMED)).run(PipelineInput(pmid=REVIEW), lambda m: None)
    assert snapshot_key(inner.specs[0]) == f'pmid-{REVIEW}+review'
    assert result.reviews == {f'PMID:{REVIEW}': True}
    assert [i.code for i in result.issues if i.source == 'pipeline'] == ['paper_is_review']
    result = KindPipeline(Recorder(), FakePubMed(fail=True)).run(PipelineInput(pmid=REVIEW), lambda m: None)
    assert [(i.code, i.severity) for i in result.issues if i.source == 'pipeline'] == [('paper_kind_unknown', 'info')]


# ── PubMed ─────────────────────────────────────────────────────────────────
class Http:
    def __init__(self, doc, status=200):
        self.doc, self.status, self.params = doc, status, []

    def get(self, url, params=None, timeout=None):
        self.params.append(params)
        http = self

        class R:
            status_code = http.status

            def raise_for_status(self):
                if http.status >= 400:
                    raise RuntimeError(f'HTTP {http.status}')

            def json(self):
                return http.doc
        return R()


def test_the_pubmed_client_reads_esummary_and_skips_unknown_ids():
    http = Http({'result': {'uids': [REVIEW, '9'], REVIEW: {'pubtype': ['Journal Article', 'Review']},
                            '9': {'error': 'cannot get document summary'}}})
    assert PubMedClient(session=http).publication_types([REVIEW, '9']) == {REVIEW: ['Journal Article', 'Review']}
    assert http.params[0]['id'] == f'{REVIEW},9' and http.params[0]['db'] == 'pubmed'
    try:
        PubMedClient(session=Http({}, status=500)).publication_types(['1'])
        raise AssertionError('should raise')
    except RuntimeError:
        pass


# ── REST and the command line ──────────────────────────────────────────────
def client(tmp_path, pubmed):
    pipe = Recorder()
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), KindPipeline(pipe, pubmed),
                     upload_dir=str(tmp_path / 'up'))
    return TestClient(app), pipe


def pdf(name):
    return ('files', (name, b'%PDF-1.4 fake', 'application/pdf'))


def test_upload_marks_only_the_pdfs_named_as_reviews_and_reads_pmids_from_pubmed(tmp_path):
    c, pipe = client(tmp_path, FakePubMed(PUBMED))
    r = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'pmids': f'{RESEARCH} {REVIEW}', 'review_files': ['b one.pdf']},
               files=[pdf('a.pdf'), pdf('b one.pdf')])
    assert r.status_code == 202 and kinds(pipe.specs[0]) == [False, True, False, True]
    s = c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()
    assert [(p['name'], p['review']) for p in s['papers']] == [('a.pdf', False), ('b_one.pdf', True), (RESEARCH, False), (REVIEW, True)]


def test_one_uploaded_pdf_ticked_as_a_review_and_a_request_wide_override(tmp_path):
    c, pipe = client(tmp_path, FakePubMed(PUBMED))
    c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'review_files': 'r.pdf'}, files=[pdf('r.pdf')])
    assert kinds(pipe.specs[-1]) == [True]
    c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'review_files': 'r.pdf', 'review': 'false', 'pmids': REVIEW},
           files=[pdf('r.pdf')])
    assert kinds(pipe.specs[-1]) == [False, False]                    # review=false wins over the ticked file and PubMed


def test_pmids_by_json_are_decided_by_pubmed_unless_review_is_given(tmp_path):
    c, pipe = client(tmp_path, FakePubMed(PUBMED))
    r = c.post('/api/llm/sessions', headers=AUTH('alice'), json={'pmid': REVIEW})
    assert kinds(pipe.specs[-1]) == [True]
    assert c.get(f'/api/llm/sessions/{r.json()["sessionId"]}', headers=AUTH('alice')).json()['papers'][0]['review'] is True
    c.post('/api/llm/sessions', headers=AUTH('alice'), json={'pmid': REVIEW, 'review': False})
    assert kinds(pipe.specs[-1]) == [False]


def test_the_command_line_leaves_the_kind_open_unless_review_is_given(tmp_path):
    pipe = CliPipeline()
    main(['--pmid', REVIEW, '--out', str(tmp_path / 'o')], pipeline=pipe, say=lambda m: None)
    assert kinds(pipe.calls[0]) == [None]                             # KindPipeline (in production wiring) decides
    main(['--pmid', REVIEW, '--review', '--force', '--out', str(tmp_path / 'o')], pipeline=pipe, say=lambda m: None)
    assert kinds(pipe.calls[1]) == [True]
