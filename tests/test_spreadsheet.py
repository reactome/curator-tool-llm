"""Annotating the papers a spreadsheet lists: read the .xlsx without openpyxl, a PDF where one is named else the PMID,
PubMed Central full text kept next to the spreadsheet, one annotation per paper (or per group), and a summary per row."""
import csv
import os
import zipfile

from curator_llm.spreadsheet import Paper, download_pmc, jobs_of, main, plan, read_xlsx, select
from tests.test_cli import FakePipeline

HEADER = ['#', 'Drug(s)', 'PDF File Name', 'PMID', 'PMCID']
ROWS = [['1', 'VEN+AZA', '', '39478230', 'PMC11618090'],
        ['2', 'VEN+AZA', '33355179.pdf', '33355179', ''],
        ['3', 'GIL', '', '30623427', 'PMC6436013'],      # the publisher does not allow XML
        ['4', 'GIL', '', '12345678', ''],                # no PMCID in the sheet: looked up
        ['5', 'GIL', 'missing.pdf', '', '']]


def xlsx(path, header=HEADER, rows=ROWS, numeric_pmid=True):
    """A minimal real .xlsx: shared strings for text, numbers stored as numbers (as Excel does for PMIDs)."""
    strings, cells = [], []

    def s(v):
        if v not in strings:
            strings.append(v)
        return strings.index(v)
    for r, row in enumerate([header] + rows, 1):
        out = []
        for c, v in enumerate(row):
            ref = f'{chr(65 + c)}{r}'
            if v == '':
                continue
            if numeric_pmid and r > 1 and header[c] == 'PMID':
                out.append(f'<c r="{ref}"><v>{v}.0</v></c>' if c else f'<c r="{ref}"><v>{v}</v></c>')
            else:
                out.append(f'<c r="{ref}" t="s"><v>{s(v)}</v></c>')
        cells.append(f'<row r="{r}">{"".join(out)}</row>')
    m = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{m}" xmlns:r="{rel}"><sheets>'
                   f'<sheet name="Papers" sheetId="1" r:id="rId1"/><sheet name="Summary" sheetId="2" r:id="rId2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>')
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{m}">' + ''.join(f'<si><t>{v}</t></si>' for v in strings) + '</sst>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{m}"><sheetData>{"".join(cells)}</sheetData></worksheet>')
        z.writestr('xl/worksheets/sheet2.xml', f'<worksheet xmlns="{m}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData></worksheet>')
    return str(path)


def folder(tmp_path):
    (tmp_path / 'pdf').mkdir()
    (tmp_path / 'pdf' / '33355179.pdf').write_bytes(b'%PDF-1.4 fake')
    return xlsx(tmp_path / 'papers.xlsx')


class Fetcher:
    """PubMedFetcher stand-in: CACHE_DIR, resolve_pmcids and fetch_jats (writing the file, as the real one does)."""

    def __init__(self, cache):
        self.CACHE_DIR, self.fetched, self.resolved = str(cache), [], []

    def resolve_pmcids(self, pmids):
        self.resolved += pmids
        return {p: 'PMC999' for p in pmids}

    def fetch_jats(self, pmcid):
        self.fetched.append(pmcid)
        xml = ('<article><!--The publisher of this article does not allow downloading of the full text in XML form.-->'
               '</article>' if pmcid == 'PMC6436013' else '<article><body><p>text</p></body></article>')
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        open(os.path.join(self.CACHE_DIR, f'{pmcid}.xml'), 'w').write(xml)
        return xml


def test_the_sheet_is_read_without_openpyxl_and_numbers_read_as_ids(tmp_path):
    rows = read_xlsx(folder(tmp_path))
    assert rows[0] == {'#': '1', 'Drug(s)': 'VEN+AZA', 'PDF File Name': '', 'PMID': '39478230', 'PMCID': 'PMC11618090'}
    assert read_xlsx(str(tmp_path / 'papers.xlsx'), 'Summary') == []          # a header only
    try:
        read_xlsx(str(tmp_path / 'papers.xlsx'), 'Nope')
        raise AssertionError('should refuse')
    except ValueError as e:
        assert 'Papers' in str(e)


def test_a_pdf_where_named_else_the_pmid_and_a_missing_pdf_is_reported(tmp_path):
    papers = plan(read_xlsx(folder(tmp_path)), str(tmp_path))
    assert [(p.row, bool(p.pdf), p.pmid, p.pmcid) for p in papers[:4]] == [
        ('1', False, '39478230', 'PMC11618090'), ('2', True, '33355179', None), ('3', False, '30623427', 'PMC6436013'),
        ('4', False, '12345678', None)]
    assert papers[1].pdf.endswith(os.path.join('pdf', '33355179.pdf')) and papers[1].source == '33355179.pdf'
    assert 'missing.pdf' in papers[4].problem
    assert [p.row for p in select(papers, '2-3,5')] == ['2', '3', '5']


def test_one_annotation_per_paper_in_sheet_order_or_one_per_group(tmp_path):
    papers = [p for p in plan(read_xlsx(folder(tmp_path)), str(tmp_path), group_by='Drug(s)') if not p.problem]
    assert [n for n, _ in jobs_of(papers, None)] == ['001_pmid39478230', '002_pmid33355179', '003_pmid30623427', '004_pmid12345678']
    assert [(n, [p.row for p in ps]) for n, ps in jobs_of(papers, 'Drug(s)')] == [('VEN_AZA', ['1', '2']), ('GIL', ['3', '4'])]


def test_pmc_full_text_is_kept_in_its_folder_and_a_restricted_paper_is_explained(tmp_path):
    papers = plan(read_xlsx(folder(tmp_path)), str(tmp_path))
    f = Fetcher(tmp_path / 'PMC')
    download_pmc(papers, lambda m: None, f)
    assert f.resolved == ['12345678'] and papers[3].pmcid == 'PMC999'      # only the one without a PMCID is looked up
    assert sorted(f.fetched) == ['PMC11618090', 'PMC6436013', 'PMC999']    # never the PDF row
    assert papers[0].text == 'downloaded' and 'publisher does not allow' in papers[2].text and 'add the PDF' in papers[2].text
    download_pmc(papers, lambda m: None, f)
    assert papers[0].text == 'cached'


def test_the_whole_run_writes_annotations_and_a_summary_per_row(tmp_path, monkeypatch):
    monkeypatch.setenv('PMC_CACHE_DIR', 'restored-after-the-test')         # main() sets it; the test must not leak it
    sheet = folder(tmp_path)
    lines = []
    code = main([sheet, '--no-qa', '--rows', '1-4'], pipeline=FakePipeline(), fetcher=Fetcher(tmp_path / 'PMC'),
                say=lines.append)
    out = tmp_path / 'annotations'
    assert code == 0 and os.environ['PMC_CACHE_DIR'] == str(tmp_path / 'PMC')
    assert os.path.isfile(out / '001_pmid39478230.annotation.json') and os.path.isfile(out / '002_pmid33355179.progress.json')
    rows = list(csv.DictReader(open(out / 'summary.tsv'), delimiter='\t'))
    assert [r['row'] for r in rows] == ['1', '2', '3', '4', 'TOTAL']
    assert rows[-1]['annotation'] == '3 annotation(s)' and rows[-1]['cost USD'] == '0.00'   # the fake spends nothing
    rows = rows[:-1]
    assert rows[2]['run'] == 'not run' and 'add the PDF' in rows[2]['full text']   # PMC gives no text: listed to fetch
    assert rows[1]['input'] == 'PDF 33355179.pdf' and rows[0]['input'] == 'PMC' and rows[0]['run'] == 'complete'
    assert rows[0]['reactions'] != ''
    lines.clear()
    main([sheet, '--no-qa', '--rows', '1-4'], pipeline=FakePipeline(), fetcher=Fetcher(tmp_path / 'PMC'), say=lines.append)
    assert sum('already done' in m for m in lines) == 3                     # a second run continues: nothing to redo


def test_list_changes_nothing_and_a_bad_row_sets_the_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv('PMC_CACHE_DIR', 'restored-after-the-test')
    sheet = folder(tmp_path)
    lines = []
    assert main([sheet, '--list'], say=lines.append) == 0
    assert not (tmp_path / 'PMC').exists() and not (tmp_path / 'annotations').exists()
    assert any('cannot be annotated' in m for m in lines) and any('missing.pdf' in m for m in lines)
    assert main([sheet, '--no-qa', '--rows', '5'], pipeline=FakePipeline(), fetcher=Fetcher(tmp_path / 'PMC'),
                say=lines.append) == 1


# ── a PDF where PMC has no text ──────────────────────────────────────────────
class Http:
    """Unpaywall and the PDF hosts: one repository copy for 10.1/ok, a publisher that refuses for 10.1/blocked."""

    def __init__(self):
        self.urls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.urls.append(url)
        import types
        if 'api.unpaywall.org' in url:
            doi = url.split('/v2/')[1]
            locs = {'10.1/ok': [{'url_for_pdf': 'https://www.biorxiv.org/x.pdf', 'version': 'acceptedVersion'}],
                    '10.1/blocked': [{'url_for_pdf': 'https://ashpublications.org/x.pdf', 'version': 'publishedVersion'}]}
            return types.SimpleNamespace(json=lambda: {'oa_locations': locs.get(doi, [])})
        if 'biorxiv' in url:
            return types.SimpleNamespace(status_code=200, content=b'%PDF-1.5 ' + b'x' * 20000)
        return types.SimpleNamespace(status_code=403, content=b'<html>Just a moment...</html>')


def test_a_pdf_named_by_pmid_in_the_folder_is_used_without_naming_it_in_the_sheet(tmp_path):
    sheet = folder(tmp_path)
    (tmp_path / 'pdf' / '30623427.pdf').write_bytes(b'%PDF-1.4 by hand')
    p = plan(read_xlsx(sheet), str(tmp_path))[2]
    assert p.pdf.endswith('30623427.pdf') and p.pdf_note == 'found 30623427.pdf'


def test_an_open_access_copy_is_downloaded_where_pmc_has_no_text_and_the_rest_are_listed(tmp_path):
    from curator_llm.spreadsheet import fetch_missing_pdfs, write_to_download
    a = Paper(row='26', pmid='30623427', pmcid='PMC8852851', doi='10.1/ok', text='no full text: ...')
    b = Paper(row='6', pmid='33824975', pmcid='PMC8138548', doi='10.1/blocked', text='no full text: ...', title='T')
    c = Paper(row='1', pmid='39478230', pmcid='PMC11618090', doi='10.1/x', text='downloaded')       # has text: not asked
    http = Http()
    still = fetch_missing_pdfs([a, b, c], str(tmp_path / 'pdf'), lambda m: None, 'me@example.org', http)
    assert a.pdf == str(tmp_path / 'pdf' / '30623427.pdf') and open(a.pdf, 'rb').read(4) == b'%PDF'
    assert a.pdf_note == 'downloaded from biorxiv.org, acceptedVersion' and still == [b]
    assert not any('10.1/x' in u for u in http.urls)
    write_to_download(str(tmp_path / 'todo.tsv'), still)
    import csv as _csv
    [row] = list(_csv.DictReader(open(tmp_path / 'todo.tsv'), delimiter='\t'))
    assert row['save as'] == '33824975.pdf' and row['PMC page'].endswith('/PMC8138548/') and row['DOI link'] == 'https://doi.org/10.1/blocked'


def test_a_paper_with_no_text_anywhere_is_listed_to_fetch_and_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv('PMC_CACHE_DIR', 'restored-after-the-test')
    sheet = folder(tmp_path)
    pipe = FakePipeline()
    lines = []
    main([sheet, '--no-qa', '--rows', '1-3', '--no-pdf-download'], pipeline=pipe, fetcher=Fetcher(tmp_path / 'PMC'),
         say=lines.append)
    assert len(pipe.calls) == 2                                            # rows 1 and 2; row 3 has no text
    assert any('Download them by hand' in m for m in lines)
    rows = list(csv.DictReader(open(tmp_path / 'pdf_to_download.tsv'), delimiter='\t'))
    assert [r['row'] for r in rows] == ['3']
    (tmp_path / 'pdf' / '30623427.pdf').write_bytes(b'%PDF-1.4 by hand')   # fetched by hand, then run again
    main([sheet, '--no-qa', '--rows', '1-3', '--no-pdf-download'], pipeline=pipe, fetcher=Fetcher(tmp_path / 'PMC'),
         say=lines.append)
    assert len(pipe.calls) == 3 and not (tmp_path / 'pdf_to_download.tsv').exists()


def test_known_pmcids_are_answered_locally_and_lookups_are_remembered(tmp_path, monkeypatch):
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reactome_llm'))
    import PubMedFetcher as f
    monkeypatch.setattr(f, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(f, '_throttle', lambda: None)
    asked = []

    class R:
        def __init__(self, ids):
            self.ids = ids

        def raise_for_status(self):
            pass

        def json(self):
            return {'records': [{'pmid': i, 'pmcid': 'PMC7' if i == '7' else None} for i in self.ids]}

    def get(url, timeout=None, params=None):
        ids = params['ids'].split(',')
        asked.append(ids)
        return R(ids)
    monkeypatch.setattr(f.requests, 'get', get)
    f.remember_pmcids({'1': 'PMC1', '2': '2'})                            # from a spreadsheet; '2' gets its prefix
    assert f.resolve_pmcids(['1', '2', '7', '8']) == {'1': 'PMC1', '2': 'PMC2', '7': 'PMC7', '8': None}
    assert asked == [['7', '8']]                                           # only what was not known
    assert f.resolve_pmcids(['7']) == {'7': 'PMC7'} and asked == [['7', '8']]   # remembered
    assert '8' not in f.known_pmcids()                                     # a "none" is never remembered
