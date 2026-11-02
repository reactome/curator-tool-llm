"""Genes, transcripts and the like: not small molecules, and not entities of their own when they are only a promoter."""
from curator_llm.models.reactome import EwasSpec, ReactionSpec, ReactomeDraft, SimpleEntitySpec
from curator_llm.services import schema_index
from curator_llm.services.draft_builder import (DraftExtraction, LlmEntity, LlmReaction, assemble, build_prompt)
from curator_llm.services.emitter import emit_user_instances
from curator_llm.services.resolvers import Resolver
from tests.fakes.ports import FakeInstanceLookup, FakeUniProt

REACTIONS = [{'source': 'PMID:1', 'annotation_result': {'name': 'CLOCK:BMAL1 drives Per2 transcription',
                                                         'input': ['Per2 gene'], 'output': ['Per2 mRNA']}}]


def extraction():
    return DraftExtraction(
        entities=[LlmEntity(key='per2_gene', kind='gene', name='PER2 gene', uniprot='Q9ZZZZ'),
                  LlmEntity(key='per2_mrna', kind='rna', name='PER2 mRNA', chebi='CHEBI:1'),
                  LlmEntity(key='per2', kind='protein', name='PER2'),
                  LlmEntity(key='nad', kind='small_molecule', name='NAD+')],
        reactions=[LlmReaction(source_index=0, name='Per2 transcription', inputs=['per2_gene'], outputs=['per2_mrna'])])


def test_gene_and_rna_become_sequence_entities_not_small_molecules():
    d, notes = assemble('clock', REACTIONS, extraction())
    gene, rna = d.participants['per2_gene'], d.participants['per2_mrna']
    assert isinstance(gene, EwasSpec) and gene.molecule == 'dna' and gene.uniprot is None
    assert isinstance(rna, EwasSpec) and rna.molecule == 'rna'
    assert d.participants['per2'].molecule == 'protein'
    assert isinstance(d.participants['nad'], SimpleEntitySpec)


def test_prompt_tells_the_model_how_to_type_them():
    p = build_prompt('clock', REACTIONS, None)
    assert 'gene | rna' in p and 'E-box' in p and 'small_molecule is ONLY' in p and 'PER2 mRNA' in p


def test_emitter_uses_reference_dna_and_rna_sequences_and_they_fit_the_schema():
    d, _ = assemble('clock', REACTIONS, extraction())
    d.reactions = [ReactionSpec(key='r1', name='Per2 transcription', inputs=['per2_gene'], outputs=['per2_mrna'])]
    r = emit_user_instances(d, {})
    new = r.user_instances['newInstances']
    classes = [i['schemaClassName'] for i in new]
    assert 'ReferenceDNASequence' in classes and 'ReferenceRNASequence' in classes
    assert classes.count('ReferenceGeneProduct') == 0           # nothing here has a UniProt entry
    for inst in new:
        assert schema_index.check_instance(inst) == []
    assert any('ReferenceDNASequence' in w and 'identifier' in w for w in r.warnings)


def test_resolver_does_not_look_up_uniprot_for_a_gene_or_transcript():
    d, _ = assemble('clock', REACTIONS, extraction())
    uniprot = FakeUniProt({}, {'PER2 gene': 'Q9ZZZZ', 'PER2 mRNA': 'Q9ZZZZ'})
    Resolver(FakeInstanceLookup([]), uniprot).resolve(d)
    assert d.participants['per2_gene'].uniprot is None and d.participants['per2_mrna'].uniprot is None
    assert 'uniprot' not in d.participants['per2_gene'].needs_resolution


# ── reusing the gene and transcript references Reactome already has ────────
def gene(name='PER2 gene', molecule='dna'):
    return EwasSpec(key='g', name=name, molecule=molecule)


def dna(db_id, name, species='Homo sapiens', genes=('PER2',)):
    d = {'dbId': db_id, 'displayName': name, 'schemaClassName': 'ReferenceDNASequence', 'geneName': list(genes)}
    if species != 'absent':
        d['species'] = species
    return d


def resolve_one(p, instances):
    d = ReactomeDraft()
    d.participants = {'g': p}
    notes = Resolver(FakeInstanceLookup(instances), FakeUniProt({}, {})).resolve(d)
    return d.participants['g'], notes


def test_the_single_human_gene_reference_is_reused_and_the_emitter_creates_nothing_new():
    p, notes = resolve_one(gene(), [dna(10, 'ENSEMBL:ENSG00000132326 PER2'), dna(11, 'ENSEMBL:ENSMUSG00000055866 Per2', 'Mus musculus')])
    assert p.reference_entity.db_id == 10 and notes == []                     # the mouse gene with the same symbol is not used
    d = ReactomeDraft()
    d.participants = {'g': p}
    d.reactions = [ReactionSpec(key='r1', name='x', inputs=['g'], outputs=['g'])]
    r = emit_user_instances(d, {})
    assert not [i for i in r.user_instances['newInstances'] if i['schemaClassName'].startswith('Reference')]
    assert not [w for w in r.warnings if 'new Reference' in w]
    ewas = next(i for i in r.user_instances['newInstances'] if i['schemaClassName'] == 'EntityWithAccessionedSequence')
    assert ewas['attributes']['referenceEntity']['dbId'] == 10


def test_a_species_that_is_not_known_is_not_taken_for_human():
    p, _ = resolve_one(gene(), [dna(11, 'ENSEMBL:ENSMUSG00000055866 Per2', species=None)])
    assert p.reference_entity is None                                          # a reference whose species is empty is skipped, not trusted


def test_several_human_matches_use_the_one_with_exactly_that_symbol_and_say_so():
    p, notes = resolve_one(gene('G6PC gene'), [dna(20, 'ENSEMBL:ENSG00000131482 G6PC1', genes=('G6PC1', 'G6PC')),
                                               dna(21, 'ENSEMBL:ENSG00000999999 G6PC', genes=('G6PC',))])
    assert p.reference_entity.db_id == 21 and len(notes) == 1 and 'confirm' in notes[0]
    p, notes = resolve_one(gene('T gene'), [dna(30, 'ENSEMBL:ENSG1 TBXT', genes=('TBXT', 'T')), dna(31, 'ENSEMBL:ENSG2 TBXT2', genes=('TBXT2', 'T'))])
    assert p.reference_entity is None and 'choose one' in notes[0]             # no guess when none stands out


def test_no_reference_in_reactome_means_a_new_one_for_the_curator_and_a_service_failure_is_survivable():
    p, notes = resolve_one(gene('NEWGENE gene'), [])
    assert p.reference_entity is None and notes == []

    class Down(FakeInstanceLookup):
        def search(self, *a, **k):
            raise ConnectionError('graph down')
    d = ReactomeDraft()
    d.participants = {'g': gene()}
    notes = Resolver(Down([]), FakeUniProt({}, {})).resolve(d)
    assert d.participants['g'].reference_entity is None and any('unavailable' in n for n in notes)


def test_a_transcript_is_looked_up_as_an_rna_reference():
    inst = [{'dbId': 40, 'displayName': 'ENSEMBL:ENST000001 PER2', 'schemaClassName': 'ReferenceRNASequence', 'geneName': ['PER2'], 'species': 'Homo sapiens'}]
    p, _ = resolve_one(gene('PER2 mRNA', 'rna'), inst)
    assert p.reference_entity.db_id == 40
    p, _ = resolve_one(gene('PER2 gene', 'dna'), inst)                         # a DNA entity does not take an RNA reference
    assert p.reference_entity is None
