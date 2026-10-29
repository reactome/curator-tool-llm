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
