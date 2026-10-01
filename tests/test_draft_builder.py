import json

from curator_llm.models.reactome import ComplexSpec, EwasSpec, SimpleEntitySpec
from curator_llm.services.draft_builder import (DraftExtraction, LlmCatalyst, LlmEntity, LlmModification,
                                                LlmReaction, LlmRegulation, _items, assemble, build_draft,
                                                build_prompt)
from curator_llm.services.emitter import emit_user_instances

REACTIONS = [
    {'source': 'PMID:24751536', 'annotation_result': {
        'name': 'PINK1 phosphorylates ubiquitin at Ser65', 'input': ['Ub', 'ATP'], 'output': ['phospho-Ub (S65)', 'ADP'],
        'catalystActivity': {'catalyst': 'PINK1'}, 'summation': [{'text': 'PINK1 phosphorylates Ub.'}],
        'evidence': ['quote one'], 'evidence_ids': ['ev-001'], 'evidence_details': {'quote one': {}}}},
    {'source': 'PMID:24751536', 'annotation_result': {
        'name': 'Parkin binds phospho-Ub', 'input': ['Parkin', 'phospho-Ub'], 'output': ['Parkin:pUb'],
        'summation': {'text': 'Parkin binds pUb.'}, 'evidence_ids': ['ev-002', 'ev-003']}}]


def extraction():
    return DraftExtraction(
        entities=[
            LlmEntity(key='pink1', kind='protein', name='PINK1', uniprot='Q9BXM7', compartment='mitochondrial outer membrane'),
            LlmEntity(key='ub', kind='protein', name='UB'),
            LlmEntity(key='pub', kind='protein', name='UB',
                      modifications=[LlmModification(psi_mod='MOD:00046', residue='S', coordinate=65)]),
            LlmEntity(key='atp', kind='small_molecule', name='ATP'), LlmEntity(key='adp', kind='small_molecule', name='ADP'),
            LlmEntity(key='parkin', kind='protein', name='PRKN'),
            LlmEntity(key='cx', kind='complex', name='Parkin:pUb', components=['parkin', 'pub'])],
        reactions=[
            LlmReaction(source_index=0, name='PINK1 phosphorylates ubiquitin at Ser65', inputs=['ub', 'atp'],
                        outputs=['pub', 'adp'], catalyst=LlmCatalyst(entity='pink1', go_function='protein kinase activity')),
            LlmReaction(source_index=1, name='Parkin binds phospho-Ub', reaction_type='binding', inputs=['parkin', 'pub'],
                        outputs=['cx'], regulations=[LlmRegulation(kind='positive', regulator='pink1', note='n')],
                        preceding_source_indexes=[0])])


class FakeModel:
    def __init__(self, out):
        self.out, self.prompt = out, None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return self.out


def test_prompt_has_no_quotes_and_lists_allowed_modifications():
    p = build_prompt('PINK1', REACTIONS, 'Q9BXM7')
    assert 'quote one' not in p and 'MOD:00046' in p and 'Q9BXM7' in p
    assert [i['pmid'] for i in _items(REACTIONS)] == ['24751536', '24751536']       # 'PMID:...' sources parse


def test_assemble_takes_evidence_pmid_and_summation_from_the_source_not_the_model():
    d, notes = assemble('PINK1', REACTIONS, extraction())
    assert notes == []
    r0, r1 = d.reactions
    assert r0.evidence_ids == ['ev-001'] and r1.evidence_ids == ['ev-002', 'ev-003']
    assert r0.pmids == ['24751536'] and r0.summation == 'PINK1 phosphorylates Ub.' and r1.summation == 'Parkin binds pUb.'
    assert r1.preceding == ['r0'] and r0.catalyst.activity.name == 'protein kinase activity'
    assert isinstance(d.participants['pub'], EwasSpec) and d.participants['pub'].modifications[0].coordinate == 65
    assert isinstance(d.participants['atp'], SimpleEntitySpec) and isinstance(d.participants['cx'], ComplexSpec)
    assert d.pathway.reactions == ['r0', 'r1'] and d.pathway.pmids == ['24751536']


def test_assemble_reports_instead_of_hiding_model_mistakes():
    ex = extraction()
    ex.reactions[1].inputs = ['parkin', 'ghost']                       # unknown key
    ex.entities[2].modifications.append(LlmModification(psi_mod='MOD:99999'))   # not allowed
    ex.entities[6].components.append('nope')
    d, notes = assemble('PINK1', REACTIONS, ex)
    assert [r.key for r in d.reactions] == ['r0']
    assert any('ghost' in n for n in notes) and any('MOD:99999' in n for n in notes)
    assert any('nope' in n for n in notes) and any('input reaction 1' in n for n in notes)
    assert len(d.participants['pub'].modifications) == 1


def test_duplicate_or_out_of_range_source_index_is_dropped():
    ex = extraction()
    ex.reactions.append(LlmReaction(source_index=0, name='dup'))
    ex.reactions.append(LlmReaction(source_index=7, name='bad'))
    d, notes = assemble('PINK1', REACTIONS, ex)
    assert len(d.reactions) == 2 and sum('bad or repeated' in n for n in notes) == 2


def test_build_draft_end_to_end_with_fake_model_and_emitter():
    m = FakeModel(extraction())
    d, notes = build_draft('PINK1', REACTIONS, 'Q9BXM7', model=m)
    assert m.schema is DraftExtraction and 'INPUT REACTIONS' in m.prompt and notes == []
    r = emit_user_instances(d)
    names = {i['displayName'] for i in r.user_instances['newInstances']}
    assert {'PINK1 phosphorylates ubiquitin at Ser65', 'Parkin binds phospho-Ub'} <= names
    json.dumps(r.user_instances)


def test_empty_input_makes_no_llm_call():
    d, notes = build_draft('PINK1', [], model=None)
    assert d.reactions == [] and notes == ['no reactions']


def test_identically_defined_entities_are_flagged_for_review_not_merged():
    ex = extraction()
    ex.entities.append(LlmEntity(key='ub_s65a', kind='protein', name='UB'))   # same definition as 'ub' (a mutant, really)
    d, notes = assemble('PINK1', REACTIONS, ex)
    assert any('defined identically as "ub"' in n and 'ub_s65a' in n for n in notes)
    assert 'ub' in d.participants and 'ub_s65a' in d.participants
