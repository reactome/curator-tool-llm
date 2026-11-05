"""The command-line runner: several papers, merged or separate, written as files, with no service."""
import json
import os

import pytest

from curator_llm.cli import collect_papers, main, parse_ids, UsageError
from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import CatalystSpec, EwasSpec, ReactionSpec, ReactomeDraft, SimpleEntitySpec
from curator_llm.models.session import Issue
from curator_llm.ports.pipeline import PipelineResult


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, spec, report):
        refs = spec.refs()
        self.calls.append(spec)
        for _ in range(30):                                         # chunk counts arrive far faster than anyone reads them
            report('extracting reactions: 3/20 chunks')
        report('merging reactions: comparing pairs')
        if any('boom' in (r.pdf_path or '') for r in refs):
            raise RuntimeError('no full text')
        d = ReactomeDraft()
        d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'ub': EwasSpec(key='ub', name='UB', uniprot='P0CG47'),
                          'atp': SimpleEntitySpec(key='atp', name='ATP', chebi='CHEBI:30616')}
        d.reactions = [ReactionSpec(key=f'r{i}', name=f'reaction {i}', inputs=['ub', 'atp'], catalyst=CatalystSpec(entity='pink1'),
                                    pmids=[r.pmid for r in refs if r.pmid], evidence_ids=['ev-001']) for i in range(len(refs) + 1)]
        issues = [Issue(source='pipeline', severity='warning', code='note', message='b.pdf: skipped, no reactions')] if any('skipme' in (r.pdf_path or '') for r in refs) else []
        return PipelineResult(d, [Evidence(id='ev-001', quote='a quote', source=refs[0].source, page=2)], issues,
                              papers={r.source: {'text': 'x'} for r in refs})


def pdf(folder, name):
    p = folder / name
    p.write_bytes(b'%PDF-1.4 fake')
    return str(p)


def run(argv, tmp_path, pipe=None):
    pipe = pipe or FakePipeline()
    lines = []
    code = main([*argv, '--out', str(tmp_path / 'out')], pipeline=pipe, say=lines.append)
    return code, pipe, lines


def test_several_papers_become_one_merged_draft_with_all_the_result_files(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    pdf(folder, 'b.pdf')
    pdf(folder, 'a.PDF')
    (folder / 'notes.txt').write_text('not a pdf')
    code, pipe, lines = run([str(folder), '--pmid', '24751536', '--pmids', '29168502, 24751536', '--focus', 'PINK1', '--name', 'together'], tmp_path)
    assert code == 0 and len(pipe.calls) == 1                                  # ONE call: one draft for all four papers
    spec = pipe.calls[0]
    assert [r.source for r in spec.refs()] == ['a.PDF', 'b.pdf', 'PMID:24751536', 'PMID:29168502'] and spec.focus == 'PINK1'
    out = tmp_path / 'out'
    assert sorted(os.listdir(out)) == ['together.annotation.json', 'together.evidence.json', 'together.issues.json', 'together.progress.json', 'together.session.json', 'together.staged_instances.json', 'together.suggestions.json']
    assert json.load(open(out / 'together.staged_instances.json'))['newInstances']
    sess = json.load(open(out / 'together.session.json'))
    assert sess['n_reactions'] == 5 and [p['kind'] for p in sess['papers']] == ['pdf', 'pdf', 'pmid', 'pmid'] and sess['source'].startswith('4 papers: ')
    ev = json.load(open(out / 'together.evidence.json'))
    assert ev['evidence'][0]['source'] == 'a.PDF' and ev['links']
    assert any('done: 5 reactions from 4 paper(s)' in m for m in lines)


def test_progress_is_not_printed_for_every_chunk(tmp_path):
    _, _, lines = run(['--pmid', '24751536'], tmp_path)
    assert sum('extracting reactions' in m for m in lines) == 1 and sum('merging reactions' in m for m in lines) == 1


def test_separate_runs_each_paper_on_its_own(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    pdf(folder, 'a.pdf')
    code, pipe, _ = run([str(folder), '--pmid', '24751536', '--separate'], tmp_path)
    assert code == 0 and len(pipe.calls) == 2
    assert sorted(f for f in os.listdir(tmp_path / 'out') if f.endswith('.staged_instances.json')) == ['a.staged_instances.json', 'pmid-24751536.staged_instances.json']


def test_one_paper_is_a_session_named_after_it(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    code, pipe, _ = run([pdf(folder, 'only one.pdf')], tmp_path)
    assert code == 0 and os.path.isfile(tmp_path / 'out' / 'only one.staged_instances.json')


def test_finished_results_are_skipped_unless_forced(tmp_path):
    first = run(['--pmid', '24751536'], tmp_path)
    again = run(['--pmid', '24751536'], tmp_path, FakePipeline())
    assert again[0] == 0 and not again[1].calls and any('already done' in m for m in again[2])
    forced = run(['--pmid', '24751536', '--force'], tmp_path, FakePipeline())
    assert len(forced[1].calls) == 1


def test_a_failing_session_does_not_stop_the_others_and_the_exit_code_says_so(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    code, pipe, lines = run([pdf(folder, 'boom.pdf'), pdf(folder, 'fine.pdf'), '--separate'], tmp_path)
    assert code == 1 and len(pipe.calls) == 2
    assert any('FAILED: RuntimeError: no full text' in m for m in lines) and any('1 session(s) done, 0 partial, 1 failed; failed: boom' in m for m in lines)
    assert os.path.isfile(tmp_path / 'out' / 'fine.staged_instances.json') and not os.path.exists(tmp_path / 'out' / 'boom.staged_instances.json')


def test_a_skipped_paper_is_reported(tmp_path):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    _, _, lines = run([pdf(folder, 'skipme.pdf'), pdf(folder, 'ok.pdf')], tmp_path)
    assert any('note: b.pdf: skipped, no reactions' in m for m in lines)


def test_there_is_no_cap_on_the_number_of_papers(tmp_path):
    code, pipe, _ = run(['--pmids', ' '.join(str(10000000 + i) for i in range(40))], tmp_path)
    assert code == 0 and len(pipe.calls[0].refs()) == 40


def test_bad_input_is_refused_before_anything_runs(tmp_path, capsys):
    folder = tmp_path / 'pdfs'
    folder.mkdir()
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'text.pdf').write_text('plain text')
    other = tmp_path / 'other'
    other.mkdir()
    cases = [
        ([], 'give at least one'),
        (['--pmid', 'abc'], 'not a PubMed ID: abc'),
        ([str(tmp_path / 'nope.pdf')], 'no such file'),
        ([str(tmp_path / 'sub')], 'no PDFs in folder'),
        ([str(tmp_path / 'text.pdf')], 'not a PDF'),
        ([pdf(folder, 'same.pdf'), pdf(other, 'same.pdf')], 'same file name'),
        (['--pmid-file', str(tmp_path / 'nofile')], 'no such file'),
    ]
    for argv, message in cases:
        pipe = FakePipeline()
        code = main([*argv, '--out', str(tmp_path / 'out')], pipeline=pipe, say=lambda m: None)
        assert code == 2 and not pipe.calls and message in capsys.readouterr().err, argv


def test_pmid_parsing_and_files(tmp_path):
    assert parse_ids('1234567, 7654321;555555  ') == ['1234567', '7654321', '555555']
    with pytest.raises(UsageError):
        parse_ids('12 x')
    f = tmp_path / 'ids.txt'
    f.write_text('# my papers\n24751536\n\n29168502  # second\n24751536\n')
    assert collect_papers([], ['11111111'], [str(f)]) == ([], ['11111111', '24751536', '29168502'])


def test_suggested_fixes_are_written_as_proposals_and_not_applied(tmp_path):
    class EnzymeInInputs(FakePipeline):
        def run(self, spec, report):
            res = super().run(spec, report)
            res.draft.reactions[0].inputs.append('pink1')                      # the enzyme listed as a substrate
            return res
    code, pipe, lines = run(['--pmid', '24751536'], tmp_path, EnzymeInInputs())
    out = tmp_path / 'out'
    sug = json.load(open(out / 'pmid-24751536.suggestions.json'))
    assert code == 0 and len(sug) == 1 and sug[0]['status'] == 'pending' and sug[0]['actor'] == 'suggestions'
    assert sug[0]['ops'][0]['op'] == 'remove'
    assert json.load(open(out / 'pmid-24751536.session.json'))['n_suggested_fixes'] == 1
    assert any('1 suggested fixes' in m for m in lines)
