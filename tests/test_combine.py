"""One annotation from many: a top pathway, a sub-pathway per group, a pathway per paper; entities shared; a reaction
several papers of a group report becomes one, listed under each paper."""
import json
import os

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (CatalystSpec, ComplexSpec, EwasSpec, PathwaySpec, ReactionSpec, ReactomeDraft,
                                         SimpleEntitySpec)
from curator_llm.models.session import Issue, PaperInfo, ReactionQA, Session
from curator_llm.services.combine import Part, combine
from curator_llm.services.emitter import emit_user_instances
from curator_llm.services.patching import apply_patch
from curator_llm.services.session_service import SessionService
from curator_llm.services.suggestions import remove_reaction_ops


def paper(pmid, reactions, extra=()):
    """A finished one-paper annotation: GSK3B, MCL1 and gilteritinib as participants, plus `extra`."""
    d = ReactomeDraft()
    d.participants = {'gsk3b': EwasSpec(key='gsk3b', name='GSK3B', uniprot='P49841'),
                      'mcl1': EwasSpec(key='mcl1', name='MCL1', uniprot='Q07820'),
                      'pmcl1': EwasSpec(key='pmcl1', name='MCL1 (phosphorylated)', uniprot='Q07820'),
                      'gil': SimpleEntitySpec(key='gil', name='gilteritinib', chebi='CHEBI:145372')}
    for p in extra:
        d.participants[p.key] = p
    d.reactions = [ReactionSpec(key=f'r{i}', name=n, inputs=ins, outputs=outs, pmids=[pmid], evidence_ids=[f'ev-{i}'],
                                catalyst=CatalystSpec(entity=cat) if cat else None, summation=f'{n} ({pmid})')
                   for i, (n, ins, outs, cat) in enumerate(reactions)]
    d.pathway = PathwaySpec(name='candidate', reactions=[r.key for r in d.reactions], pmids=[pmid])
    ev = [Evidence(id=f'ev-{i}', quote=f'quote {i} of {pmid}', source=f'PMID:{pmid}') for i in range(len(reactions))]
    s = Session(owner='x', papers=[PaperInfo(source=f'PMID:{pmid}', kind='pmid', pmid=pmid, name=pmid)], draft=d,
                evidence=ev, status='ready', paper_texts={f'PMID:{pmid}': {'text': pmid}},
                qa={'r0': ReactionQA(priority='fix', summary=f'fix {pmid}', is_reaction=True)},
                issues=[Issue(source='qa', code='qa_llm', message=f'issue of {pmid}', reaction_key='r0')])
    return s


PHOS = ('GSK3B phosphorylates MCL1', ['mcl1'], ['pmcl1'], 'gsk3b')


def parts():
    a = paper('111', [PHOS, ('MCL1 is degraded', ['pmcl1'], [], None)])
    b = paper('222', [('GSK3 beta phosphorylates Mcl-1', ['mcl1'], ['pmcl1'], 'gsk3b')])     # the same event, other words
    c = paper('333', [('venetoclax binds BCL2', ['bcl2'], ['bcl2'], None)], extra=[EwasSpec(key='bcl2', name='BCL2')])
    return [Part(a, '001', 'Paper one (row 1)', 'GIL'), Part(b, '002', 'Paper two (row 2)', 'GIL'),
            Part(c, '003', 'Paper three (row 3)', 'VEN')]


def judge(a1, a2):
    phos = lambda a: 'phosphorylates' in a['name']
    return (phos(a1) and phos(a2)), 0.95


def test_a_hierarchy_of_top_group_and_paper_pathways_with_a_shared_reaction():
    s = combine(parts(), 'AML drug response', 'wgm', judge=judge)
    d = s.draft
    top = d.pathway
    assert top.name == 'AML drug response' and [g.name for g in top.subpathways] == ['GIL', 'VEN']
    gil = top.subpathways[0]
    assert [p.name for p in gil.subpathways] == ['Paper one (row 1)', 'Paper two (row 2)']
    # one phosphorylation, citing both papers, listed under both papers' pathways
    assert gil.subpathways[0].reactions == ['001_r0', '001_r1'] and gil.subpathways[1].reactions == ['001_r0']
    shared = next(r for r in d.reactions if r.key == '001_r0')
    assert shared.pmids == ['111', '222'] and shared.evidence_ids == ['001_ev-0', '002_ev-0']
    assert '002_r0' not in [r.key for r in d.reactions] and len(d.reactions) == 3
    assert gil.pmids == ['111', '222'] and top.pmids == ['111', '222', '333']
    # entities are shared across papers: one MCL1, and the phosphorylated form stays a different entity
    assert sorted(k for k in d.participants) == ['bcl2', 'gil', 'gsk3b', 'mcl1', 'pmcl1']
    # quotes, QA and issues come along with the reactions they belong to; the merged-away copy's are dropped
    assert {e.id for e in s.evidence} >= {'001_ev-0', '002_ev-0', '003_ev-0'}
    assert set(s.qa) == {'001_r0', '003_r0'} and {i.reaction_key for i in s.issues} == {'001_r0', '003_r0'}
    assert [p.source for p in s.papers] == ['PMID:111', 'PMID:222', 'PMID:333'] and len(s.paper_texts) == 3


def test_without_a_judge_each_paper_keeps_its_copy_and_without_groups_papers_sit_under_the_top():
    ps = parts()
    s = combine(ps, 'top', 'wgm')
    assert len(s.draft.reactions) == 4
    for p in ps:
        p.group = ''
    s = combine(ps, 'top', 'wgm', judge=judge)
    assert [p.name for p in s.draft.pathway.subpathways] == ['Paper one (row 1)', 'Paper two (row 2)', 'Paper three (row 3)']
    assert len(s.draft.reactions) == 3                                    # one group of all: the copies still fold


def test_complexes_point_at_the_shared_parts_and_states_with_one_accession_stay_apart():
    cx = ComplexSpec(key='cx', name='GSK3B:MCL1', components=['gsk3b', 'mcl1'])
    a = paper('111', [('GSK3B binds MCL1', ['gsk3b', 'mcl1'], ['cx'], None)], extra=[cx])
    b = paper('222', [('GSK3B binds MCL1 again', ['gsk3b', 'mcl1'], ['cx2'], None)],
              extra=[ComplexSpec(key='cx2', name='GSK3B:MCL1', components=['gsk3b', 'mcl1'])])
    s = combine([Part(a, '001', 'one'), Part(b, '002', 'two')], 'top', 'wgm')
    assert [k for k, p in s.draft.participants.items() if isinstance(p, ComplexSpec)] == ['cx']   # same name, same parts
    assert s.draft.participants['cx'].components == ['gsk3b', 'mcl1']
    assert {'mcl1', 'pmcl1'} <= set(s.draft.participants)                 # one accession, two states


def test_the_combined_annotation_exports_nested_pathways_and_a_removed_reaction_leaves_every_pathway():
    s = combine(parts(), 'AML drug response', 'wgm', judge=judge)
    SessionService.refresh(s)
    pathways = [i for i in s.user_instances['newInstances'] if i['schemaClassName'] == 'Pathway']
    by_id = {i['dbId']: i for i in s.user_instances['newInstances']}
    top = next(p for p in pathways if p['displayName'] == 'AML drug response')
    assert [by_id[e['dbId']]['displayName'] for e in top['attributes']['hasEvent']] == ['GIL', 'VEN']
    gil = by_id[top['attributes']['hasEvent'][0]['dbId']]
    paper_two = by_id[gil['attributes']['hasEvent'][1]['dbId']]
    assert [e['schemaClassName'] for e in paper_two['attributes']['hasEvent']] == ['Reaction']
    after = apply_patch(s.draft, remove_reaction_ops(s.draft, '001_r0'))   # listed under two papers
    assert all('001_r0' not in pw.reactions for _, pw in after.pathway.walk())
    assert emit_user_instances(after).user_instances['newInstances']


def test_the_spreadsheet_makes_the_hierarchy_from_the_finished_papers(tmp_path, monkeypatch):
    from curator_llm.spreadsheet import main
    from tests.test_cli import FakePipeline
    from tests.test_spreadsheet import Fetcher, folder
    monkeypatch.setenv('PMC_CACHE_DIR', 'restored-after-the-test')
    sheet = folder(tmp_path)
    lines = []
    code = main([sheet, '--no-qa', '--rows', '1-4', '--hierarchy', '--group-by', 'Drug(s)', '--title', 'AML drug response',
                 '--no-shared'], pipeline=FakePipeline(), fetcher=Fetcher(tmp_path / 'PMC'), say=lines.append)
    assert code == 0, lines
    doc = json.load(open(tmp_path / 'annotations' / 'AML_drug_response.annotation.json'))
    top = doc['draft']['pathway']
    assert top['name'] == 'AML drug response' and [g['name'] for g in top['subpathways']] == ['VEN+AZA', 'GIL']
    assert [len(g['subpathways']) for g in top['subpathways']] == [2, 1]     # row 3 has no text anywhere: not run
    assert os.path.isfile(tmp_path / 'annotations' / 'AML_drug_response.staged_instances.json')
    assert any('[hierarchy] AML_drug_response:' in m for m in lines)
