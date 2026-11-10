"""The PubMed ID of an uploaded PDF: found from its DOI or title (and offered, not applied), given at upload or set
later; the paper, its reactions, quotes and pathways then cite it, and PubMed decides whether it is a review."""
import json

import pymupdf
from fastapi.testclient import TestClient

from curator_llm.adapters.pubmed import PubMedClient
from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import EwasSpec, PathwaySpec, ReactionSpec, ReactomeDraft
from curator_llm.ports.pipeline import PaperRef, PipelineInput, PipelineResult
from curator_llm.services.paper_identity import identify, title_score
from curator_llm.services.paper_kind import KindPipeline, decide
from tests.fakes.ports import FakeAuthProvider, FakePubMed, SyncJobRunner
from tests.test_paper_kind import Http
from tests.test_session_api import AUTH, TOKENS

TP53 = '33824975'
ARTICLES = {TP53: {'title': 'Intact TP-53 function is essential for sustaining durable responses to BH3-mimetic drugs '
                            'in leukemias.', 'doi': '10.1182/blood.2020010167'},
            '22517893': {'title': 'Cited paper on something else entirely.', 'doi': '10.1182/blood-2011-11-382226'}}


def make_pdf(path, lines, meta_title=None):
    """lines: [(text, font size)] down page 1."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for text, size in lines:
        page.insert_text((40, y), text, fontsize=size)
        y += size * 1.6
    if meta_title:
        doc.set_metadata({'title': meta_title})
    doc.save(str(path))
    return str(path)


# ── finding it ────────────────────────────────────────────────────────────
def test_the_doi_on_the_first_page_finds_the_paper(tmp_path):
    pdf = make_pdf(tmp_path / 'a.pdf', [('Intact TP-53 function is essential for sustaining durable', 20),
                                         ('responses to BH3-mimetic drugs in leukemias', 20),
                                         ('Blood 2021. doi: 10.1182/blood.2020010167.', 9)])
    hit = identify(pdf, FakePubMed(articles=ARTICLES))
    assert hit['pmid'] == TP53 and hit['method'] == 'doi' and hit['doi'] == '10.1182/blood.2020010167'


def test_a_cited_papers_doi_is_not_taken_for_this_one(tmp_path):
    pdf = make_pdf(tmp_path / 'notes.pdf', [('Lab notes on MCL1 regulation in our AML lines', 20),
                                             ('Building on earlier work (doi: 10.1182/blood-2011-11-382226).', 9)])
    assert identify(pdf, FakePubMed(articles=ARTICLES)) is None


def test_without_a_doi_the_title_finds_it_even_when_only_its_first_line_is_read(tmp_path):
    # the title runs over two lines in two sizes: only the larger, first line is one candidate
    pdf = make_pdf(tmp_path / 'b.pdf', [('Intact TP-53 function is essential for sustaining durable', 22),
                                         ('responses to BH3-mimetic drugs in leukemias', 18),
                                         ('Rachel Thijssen and others', 10)])
    pm = FakePubMed(articles=ARTICLES)
    hit = identify(pdf, pm)
    assert hit['pmid'] == TP53 and hit['method'] == 'title' and hit['score'] >= 0.8
    # words PubMed's title search does not index ("with", "for", ...) are left out: with them it finds nothing
    assert all('for[Title]' not in q and 'with[Title]' not in q for q in pm.queries)


def test_the_pdfs_own_title_field_is_used_and_nothing_is_guessed_without_a_close_match(tmp_path):
    pdf = make_pdf(tmp_path / 'c.pdf', [('x', 8)], meta_title=ARTICLES[TP53]['title'])
    assert identify(pdf, FakePubMed(articles=ARTICLES))['pmid'] == TP53
    other = make_pdf(tmp_path / 'd.pdf', [('Quarterly progress report for the AML project', 20)])
    assert identify(other, FakePubMed(articles=ARTICLES)) is None
    assert title_score('Intact TP-53 function is essential', ARTICLES[TP53]['title']) >= 0.8
    assert title_score('A different paper about something else', ARTICLES[TP53]['title']) < 0.8


def test_the_pubmed_client_searches_and_reads_titles():
    http = Http({'esearchresult': {'idlist': [TP53]}})
    assert PubMedClient(session=http).search('"10.1182/blood.2020010167"[doi]', 2) == [TP53]
    assert http.params[0]['term'] == '"10.1182/blood.2020010167"[doi]' and http.params[0]['retmax'] == 2
    http = Http({'result': {'uids': [TP53], TP53: {'title': 'Intact TP-53 ...'}}})
    assert PubMedClient(session=http).titles([TP53]) == {TP53: 'Intact TP-53 ...'}


def test_a_pdfs_pubmed_id_decides_whether_it_is_a_review():
    pm = FakePubMed({TP53: ['Journal Article', 'Review']})
    spec, notes = decide(PipelineInput(pdf_path='/x/a.pdf', pdf_pmid=TP53), pm)
    assert spec.review is True and pm.calls == [[TP53]] and 'review' in notes[0]
    spec, _ = decide(PipelineInput(pdf_path='/x/a.pdf'), FakePubMed({}))       # no ID: research, PubMed not asked
    assert spec.review is False


# ── REST: identify, upload with the ID, set it later ──────────────────────
class PdfPipeline:
    """Reactions tagged with each paper's source (a PDF's file name), without PubMed IDs, as extraction makes them."""

    def __init__(self):
        self.specs = []

    def run(self, spec, report):
        self.specs.append(spec)
        d, ev = ReactomeDraft(), []
        d.participants = {'mcl1': EwasSpec(key='mcl1', name='MCL1')}
        for i, ref in enumerate(spec.refs()):
            pm = [ref.pmid] if ref.pmid else []
            d.reactions.append(ReactionSpec(key=f'r{i}', name=f'reaction {i}', inputs=['mcl1'], pmids=pm,
                                            evidence_ids=[f'ev{i}']))
            ev.append(Evidence(id=f'ev{i}', quote=f'q{i}', source=ref.source, pmid=ref.pmid))
        d.pathway = PathwaySpec(name='p', reactions=[r.key for r in d.reactions],
                                pmids=[r.pmid for r in spec.refs() if r.pmid])
        return PipelineResult(d, ev)


def client(tmp_path, pubmed=None, search=True):
    pm = pubmed or FakePubMed({TP53: ['Journal Article']}, articles=ARTICLES)
    pipe = PdfPipeline()
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), KindPipeline(pipe, pm),
                     upload_dir=str(tmp_path / 'up'), article_search=pm if search else None)
    return TestClient(app), pipe


def session(c, sid):
    return c.get(f'/api/llm/sessions/{sid}', headers=AUTH('alice')).json()


def full(c, sid):
    return c.app.state.store.get(sid, 'alice')


def test_identify_offers_the_pubmed_id_of_an_uploaded_pdf(tmp_path):
    c, _ = client(tmp_path)
    pdf = make_pdf(tmp_path / 'a.pdf', [('Intact TP-53 function is essential for sustaining durable', 20),
                                         ('responses to BH3-mimetic drugs in leukemias', 20),
                                         ('doi: 10.1182/blood.2020010167', 9)])
    r = c.post('/api/llm/papers/identify', headers=AUTH('alice'), files={'file': ('a.pdf', open(pdf, 'rb'), 'application/pdf')})
    assert r.status_code == 200 and r.json()['found'] and r.json()['pmid'] == TP53 and r.json()['method'] == 'doi'
    r = c.post('/api/llm/papers/identify', headers=AUTH('alice'), files={'file': ('x.pdf', b'not a pdf', 'application/pdf')})
    assert r.status_code == 422
    down, _ = client(tmp_path, FakePubMed(fail=True))
    r = down.post('/api/llm/papers/identify', headers=AUTH('alice'), files={'file': ('a.pdf', open(pdf, 'rb'), 'application/pdf')})
    assert r.status_code == 200 and not r.json()['found'] and 'could not check' in r.json()['note']
    off, _ = client(tmp_path, search=False)
    assert off.post('/api/llm/papers/identify', headers=AUTH('alice'),
                    files={'file': ('a.pdf', open(pdf, 'rb'), 'application/pdf')}).status_code == 503
    assert c.post('/api/llm/papers/identify', files={'file': ('a.pdf', b'%PDF-', 'application/pdf')}).status_code == 401


def test_an_uploaded_pdf_with_its_pubmed_id_is_cited_by_its_reactions(tmp_path):
    c, pipe = client(tmp_path)
    r = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'pdf_pmids': json.dumps({'a.pdf': TP53})},
               files=[('files', ('a.pdf', b'%PDF-1.4 fake', 'application/pdf'))])
    sid = r.json()['sessionId']
    assert pipe.specs[0].refs()[0].pdf_path and pipe.specs[0].refs()[0].pdf_pmid == TP53   # text still from the PDF
    s = full(c, sid)
    assert s.papers[0].kind == 'pdf' and s.papers[0].pmid == TP53 and s.pmid == TP53
    assert s.draft.reactions[0].pmids == [TP53] and s.draft.pathway.pmids == [TP53] and s.evidence[0].pmid == TP53
    assert not any(i.code == 'paper_no_pmid' for i in s.issues)
    refs = [i for i in s.user_instances['newInstances'] if i['schemaClassName'] == 'LiteratureReference']
    assert [i['attributes']['pubMedIdentifier'] for i in refs] == [int(TP53)]
    for wrong in ('x1', '999'):                                         # not a number; unknown to PubMed
        bad = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'pdf_pmids': json.dumps({'a.pdf': wrong})},
                     files=[('files', ('a.pdf', b'%PDF-1.4 fake', 'application/pdf'))])
        assert bad.status_code == 422


def test_set_change_and_remove_a_pdfs_pubmed_id_afterwards(tmp_path):
    c, _ = client(tmp_path)
    sid = c.post('/api/llm/sessions/upload', headers=AUTH('alice'), data={'pmids': '22517893'},
                 files=[('files', ('a.pdf', b'%PDF-1.4 fake', 'application/pdf'))]).json()['sessionId']
    s = full(c, sid)
    pdf_src = s.papers[0].source
    assert s.papers[0].pmid is None and s.draft.reactions[0].pmids == [] and s.draft.reactions[1].pmids == ['22517893']
    url = f'/api/llm/sessions/{sid}/papers/pmid'
    r = c.put(url, headers=AUTH('alice'), json={'source': pdf_src, 'pmid': TP53})
    assert r.status_code == 200 and r.json()['title'].startswith('Intact TP-53')
    s = full(c, sid)
    assert s.papers[0].pmid == TP53 and s.draft.reactions[0].pmids == [TP53]
    assert s.draft.reactions[1].pmids == ['22517893']                     # the other paper's reaction is untouched
    assert sorted(s.draft.pathway.pmids) == sorted(['22517893', TP53])
    assert any(e.action == 'set pubmed id' for e in s.change_log)
    # unknown to PubMed, not a number, not a PDF, no such paper
    assert c.put(url, headers=AUTH('alice'), json={'source': pdf_src, 'pmid': '999'}).status_code == 422
    assert c.put(url, headers=AUTH('alice'), json={'source': pdf_src, 'pmid': 'abc'}).status_code == 422
    assert c.put(url, headers=AUTH('alice'), json={'source': 'PMID:22517893', 'pmid': TP53}).status_code == 409
    assert c.put(url, headers=AUTH('alice'), json={'source': 'nope.pdf', 'pmid': TP53}).status_code == 404
    assert c.put(url, headers=AUTH('bob'), json={'source': pdf_src, 'pmid': TP53}).status_code == 404
    # removing it takes it off the reactions again
    assert c.put(url, headers=AUTH('alice'), json={'source': pdf_src, 'pmid': ''}).status_code == 200
    s = full(c, sid)
    assert s.papers[0].pmid is None and s.draft.reactions[0].pmids == [] and TP53 not in s.draft.pathway.pmids


def test_a_single_pdf_session_gets_its_id_later_and_the_no_pmid_issue_goes(tmp_path):
    c, _ = client(tmp_path)
    sid = c.post('/api/llm/sessions/upload', headers=AUTH('alice'),
                 files=[('files', ('a.pdf', b'%PDF-1.4 fake', 'application/pdf'))]).json()['sessionId']
    assert any(i.code == 'paper_no_pmid' for i in full(c, sid).issues)
    src = full(c, sid).papers[0].source
    c.put(f'/api/llm/sessions/{sid}/papers/pmid', headers=AUTH('alice'), json={'source': src, 'pmid': TP53})
    s = full(c, sid)
    assert s.pmid == TP53 and s.draft.reactions[0].pmids == [TP53]
    assert not any(i.code == 'paper_no_pmid' for i in s.issues)


def test_a_pdf_already_in_an_annotation_can_be_identified_later(tmp_path):
    c, _ = client(tmp_path)
    data = open(make_pdf(tmp_path / 'src.pdf', [('Intact TP-53 function is essential for sustaining durable', 20),
                                                 ('responses to BH3-mimetic drugs in leukemias', 20),
                                                 ('doi: 10.1182/blood.2020010167', 9)]), 'rb').read()
    sid = c.post('/api/llm/sessions/upload', headers=AUTH('alice'),
                 files=[('files', ('tp53.pdf', data, 'application/pdf'))]).json()['sessionId']
    src = full(c, sid).papers[0].source
    url = f'/api/llm/sessions/{sid}/papers/identify'
    r = c.post(url, headers=AUTH('alice'), json={'source': src})
    assert r.status_code == 200 and r.json()['pmid'] == TP53
    assert full(c, sid).papers[0].pmid is None                              # only a suggestion: nothing changed
    assert c.post(url, headers=AUTH('alice'), json={'source': 'other.pdf'}).status_code == 404
    assert c.post(url, headers=AUTH('bob'), json={'source': src}).status_code == 404
