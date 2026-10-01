"""fulltext_extractor (gene-first flow) now delegates the per-paper work to SubprocessExtractor.
These pin the result shape and token accounting the rest of the gene flow relies on."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fulltext_extractor as fx  # noqa: E402
from curator_llm.ports.extractor import ExtractionResult  # noqa: E402
from reaction_to_instances import _pmid_of  # noqa: E402


class FakeExtractor:
    def __init__(self, result):
        self.result, self.calls = result, []

    def extract(self, pmid, pdf_path, gene):
        self.calls.append((pmid, pdf_path, gene))
        return self.result


def patch(monkeypatch, result):
    fake = FakeExtractor(result)
    monkeypatch.setattr(fx, '_extractor', lambda timeout, review: fake)
    fx.reset_usage()
    return fake


OK = ExtractionResult(ok=True, reactions=[{'annotation_result': {'name': 'a'}}], n_extracted=3, n_merged=1,
                      extraction_path='/r/x_extraction.json', merged_path='/r/x_merged.json',
                      review_path='/r/x_review.md', review_score=7.5,
                      usage={'calls': 5, 'input': 100, 'output': 20, 'cache_read': 1, 'cache_write': 2},
                      warnings=['review failed: nope'])


def test_xml_paper_maps_to_the_result_dict_and_adds_usage(monkeypatch):
    fake = patch(monkeypatch, OK)
    r = fx._pipeline_one_paper({'pmid': '123456', 'spec': '123456', 'tag': None, 'extraction_path': '/r/x_extraction.json'},
                               'PINK1', 60, True)
    assert fake.calls == [('123456', None, 'PINK1')]
    assert r == {'pmid': '123456', 'source': 'xml', 'spec': '123456', 'n_extracted': 3, 'n_merged': 1,
                 'reactions': OK.reactions, 'review_score': 7.5, 'review_path': '/r/x_review.md',
                 'extraction_path': '/r/x_extraction.json', 'merged_path': '/r/x_merged.json', 'ok': True}
    assert fx.get_usage() == {'calls': 5, 'input': 100, 'output': 20, 'cache_read': 1, 'cache_write': 2}


def test_local_pdf_is_passed_as_a_path_and_failure_keeps_the_dead_shape(monkeypatch, tmp_path):
    pdf = tmp_path / 'p.pdf'
    pdf.write_bytes(b'%PDF-1')
    monkeypatch.setattr(fx, 'PAPERS_DIR', str(tmp_path / 'cache'))
    fake = patch(monkeypatch, ExtractionResult(error='extraction failed (exit 1): boom'))
    r = fx._pipeline_one_paper({'pmid': '9', 'spec': str(pdf), 'tag': 'localpdf', 'extraction_path': 'x'}, 'g', 60, False)
    assert fake.calls == [(None, str(pdf), 'g')] and r['source'] == 'pdf'
    assert r['ok'] is False and r['reactions'] == [] and r['n_extracted'] == 0 and r['review_score'] is None
    assert os.path.isfile(tmp_path / 'cache' / 'p.pdf')                         # cached where run_review looks for it
    assert fx.get_usage()['calls'] == 0


def test_usage_accumulates_across_papers_and_resets(monkeypatch):
    patch(monkeypatch, OK)
    spec = {'pmid': '1', 'spec': '1', 'tag': None, 'extraction_path': 'x'}
    fx._pipeline_one_paper(spec, 'g', 60, False)
    fx._pipeline_one_paper(spec, 'g', 60, False)
    assert fx.get_usage()['input'] == 200
    fx.reset_usage()
    assert fx.get_usage()['input'] == 0


def test_dead_cross_paper_mode_is_gone():
    assert not hasattr(fx, 'extract_and_merge') and not hasattr(fx, '_run_review')


def test_pmid_is_found_in_every_source_form_the_pipeline_uses():
    assert _pmid_of('PMID:24751536') == '24751536'
    assert _pmid_of('24751536 (abstract)') == '24751536' and _pmid_of('24751536') == '24751536'
    assert _pmid_of('PINK1.pdf') == '' and _pmid_of('') == ''
    assert _pmid_of('My_Paper_20240101.pdf') == '' and _pmid_of('2024_Smith_12345.pdf') == ''      # digits in a file name are not a PMID
    assert _pmid_of('PMC1234567') == ''
