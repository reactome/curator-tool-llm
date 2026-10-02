import pytest

from curator_llm.models.reactome import ReactomeDraft
from curator_llm.ports.extractor import ExtractionResult
from curator_llm.ports.pipeline import PipelineInput
from curator_llm.services.draft_builder import DraftExtraction, LlmEntity, LlmReaction
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.paper_text import PaperText
from tests.fakes.ports import FakeInstanceLookup, FakeUniProt

QUOTE = 'TcPINK1 WT, but not KD, incorporates 32P from radiolabeled ATP onto Ub'
PAGES = [f'Results\n{QUOTE} (Fig. 3 B).\n']


class Model:
    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt, config=None):
        return DraftExtraction(
            entities=[LlmEntity(key='pink1', kind='protein', name='PINK1'), LlmEntity(key='ub', kind='protein', name='UB')],
            reactions=[LlmReaction(source_index=0, name='PINK1 phosphorylates ubiquitin', inputs=['ub'], outputs=['ub'])])


class FakeExtractor:
    def __init__(self, result):
        self.result, self.calls = result, []

    def extract(self, pmid, pdf_path, gene):
        self.calls.append((pmid, pdf_path, gene))
        return self.result


def reactions():
    return [{'source': 'PMID:24751536', 'annotation_result': {
        'name': 'PINK1 phosphorylates ubiquitin', 'input': ['Ub'], 'output': ['pUb'],
        'evidence': [{'quote': QUOTE, 'supports': ['reaction']}, 'a quote that is not in the paper at all, honest'],
        'summation': [{'text': 's'}]}}]


def make(extracted):
    return DefaultPipeline(FakeInstanceLookup([]), FakeUniProt({}, {'PINK1': 'Q9BXM7'}), None, model=Model(),
                           extractor=FakeExtractor(extracted))


def test_pipeline_verifies_quotes_builds_draft_and_reports_rejected_quotes(monkeypatch):
    monkeypatch.setattr('curator_llm.services.pipeline_default.load_paper_text',
                        lambda src: PaperText.from_pages('24751536', PAGES))
    steps = []
    res = make(ExtractionResult(ok=True, reactions=reactions())).run(PipelineInput(pmid='24751536', focus='PINK1'), steps.append)
    assert isinstance(res.draft, ReactomeDraft) and len(res.draft.reactions) == 1
    assert res.draft.reactions[0].evidence_ids == ['ev-001'] and res.evidence[0].verified.value == 'exact'
    assert res.evidence[0].page == 1 and res.evidence[0].figure == '3B'
    rej = [i for i in res.issues if i.code == 'quote_rejected']
    assert len(rej) == 1 and rej[0].reaction_key == 'r0'
    assert any('UniProt' in m or 'verifying' in m for m in steps) and len(steps) == 4


def test_pipeline_fails_clearly_when_nothing_was_extracted():
    with pytest.raises(RuntimeError, match='no reactions.*boom'):
        make(ExtractionResult(ok=False, error='boom')).run(PipelineInput(pmid='1'), lambda m: None)
    with pytest.raises(ValueError):
        make(ExtractionResult()).run(PipelineInput(), lambda m: None)
