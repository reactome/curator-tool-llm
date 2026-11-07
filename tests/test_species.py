"""Molecules of another organism (a virus in a host-pathogen paper): the draft names their organism, the resolver finds
the Species and searches identifiers in it (KSHV's LANA is LANA1, Q9QR71), and the emitter follows Reactome's
convention: the viral protein's species is the virus, a complex, reaction or pathway with human parts is Homo sapiens
with the virus as relatedSpecies."""
from curator_llm.adapters.uniprot import RestUniProtClient
from curator_llm.models.reactome import (ComplexSpec, EwasSpec, ExistingRef, PathwaySpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec)
from curator_llm.services.draft_builder import DraftExtraction, LlmEntity, LlmReaction, assemble, build_prompt
from curator_llm.services.emitter import emit_user_instances
from curator_llm.services.resolvers import Resolver
from tests.fakes.ports import FakeInstanceLookup, FakeUniProt
from tests.test_resolvers import _Http, _R

HHV8 = {'dbId': 2671781, 'displayName': 'Human herpesvirus 8', 'schemaClassName': 'Species', 'taxId': '37296'}
HHV8_REF = ExistingRef(db_id=2671781, display_name='Human herpesvirus 8', schema_class='Species')
LANA1_RGP = {'dbId': 9973338, 'displayName': 'UniProt:Q9QR71 LANA1', 'schemaClassName': 'ReferenceGeneProduct',
             'identifier': 'Q9QR71'}
LANA1_ENTRY = {'accession': 'Q9QR71', 'genes': ['LANA1', 'ORF73'], 'names': ['Protein LANA1'], 'reviewed': True,
               'organism': 'Human herpesvirus 8 type P (isolate GK18)'}


class GraphLookup(FakeInstanceLookup):
    """The graph adapter's extras: species by name or synonym, and accessions by name within a species."""

    def find_species(self, name):
        return HHV8 if name.lower() in ('kshv', 'hhv8', 'human herpesvirus 8') else None

    def find_accession(self, name, species):
        self.calls.append(('accession', name, species))
        return 'Q9QR71' if species == 'Human herpesvirus 8' and 'LANA1'.startswith(name.upper()) else None

    def find_human_accession(self, name):
        return {'bub1': 'O43683'}.get(name.lower())


def test_a_viral_protein_is_found_in_its_own_species_and_takes_its_gene_name():
    d = ReactomeDraft()
    d.participants = {'lana': EwasSpec(key='lana', name='LANA', species='KSHV'),
                      'bub1': EwasSpec(key='bub1', name='BUB1')}
    lookup = GraphLookup([LANA1_RGP])
    notes = Resolver(lookup, FakeUniProt({'Q9QR71': LANA1_ENTRY, 'O43683': {
        'accession': 'O43683', 'genes': ['BUB1'], 'names': [], 'reviewed': True, 'organism': 'Homo sapiens'}})).resolve(d)
    lana = d.participants['lana']
    assert lana.uniprot == 'Q9QR71' and lana.name == 'LANA1' and lana.species == 'Human herpesvirus 8'
    assert lana.species_ref.db_id == 2671781 and lana.reference_entity.db_id == 9973338
    assert ('accession', 'LANA', 'Human herpesvirus 8') in lookup.calls
    assert any('renamed LANA1' in n for n in notes) and not lana.needs_resolution
    assert d.participants['bub1'].uniprot == 'O43683' and d.participants['bub1'].species_ref is None


def test_uniprot_is_searched_in_the_viral_taxon_when_the_graph_does_not_know_the_protein():
    d = ReactomeDraft()
    d.participants = {'lana': EwasSpec(key='lana', name='LANA1', species='Human herpesvirus 8')}
    uniprot = FakeUniProt({'Q9QR71': LANA1_ENTRY}, {('LANA1', 37296): 'Q9QR71'})
    Resolver(FakeInstanceLookup([dict(HHV8, name=['Human herpesvirus 8'], taxId=None)]), uniprot).resolve(d)
    # a lookup without find_species finds the Species by name, but has no taxon id: no UniProt search
    assert d.participants['lana'].species_ref.db_id == 2671781 and d.participants['lana'].uniprot is None
    assert 'uniprot' in d.participants['lana'].needs_resolution and not getattr(uniprot, 'searches', [])
    d.participants = {'lana': EwasSpec(key='lana', name='LANA1', species='Human herpesvirus 8')}
    class NewToTheGraph(GraphLookup):
        def find_accession(self, name, species):
            return None

    Resolver(NewToTheGraph([]), uniprot).resolve(d)
    assert d.participants['lana'].uniprot == 'Q9QR71' and uniprot.searches == [('LANA1', 37296)]


def test_an_organism_reactome_does_not_have_is_flagged_and_nothing_is_guessed():
    d = ReactomeDraft()
    d.participants = {'x': EwasSpec(key='x', name='VP35', species='Zaire ebolavirus')}
    uniprot = FakeUniProt({}, {'VP35': 'HUMAN-GUESS'})
    notes = Resolver(GraphLookup([]), uniprot).resolve(d)
    x = d.participants['x']
    assert x.uniprot is None and {'species', 'uniprot'} <= set(x.needs_resolution)
    assert any('Zaire ebolavirus' in n for n in notes) and not getattr(uniprot, 'searches', [])


def test_uniprot_search_covers_the_strains_of_a_virus_but_only_humans_proper():
    http = _Http(_R(200, {'results': [{'primaryAccession': 'Q9QR71'}]}))
    c = RestUniProtClient(session=http)
    assert c.search_gene('LANA1', 37296) == 'Q9QR71'
    c.search_gene('BUB1')
    assert 'taxonomy_id:37296' in http.calls[0][1]['query'] and 'organism_id:9606' in http.calls[1][1]['query']


def test_the_draft_step_asks_for_the_organism_and_keeps_it_for_non_human_molecules_only():
    assert 'LANA1' in build_prompt('BUB1', [], None)
    ex = DraftExtraction(entities=[
        LlmEntity(key='lana', kind='protein', name='LANA1', species='Human herpesvirus 8'),
        LlmEntity(key='ep', kind='gene', name='KSHV episome gene', species='KSHV'),
        LlmEntity(key='bub1', kind='protein', name='BUB1', species='Homo sapiens')],
        reactions=[LlmReaction(source_index=0, name='LANA1 binds BUB1', inputs=['lana', 'bub1'], outputs=[])])
    d, _ = assemble('BUB1', [{'source': 'PMID:1', 'annotation_result': {'name': 'x'}}], ex)
    assert d.participants['lana'].species == 'Human herpesvirus 8' and d.participants['ep'].species == 'KSHV'
    assert d.participants['bub1'].species == 'Homo sapiens'


def _virus_draft():
    d = ReactomeDraft()
    d.participants = {
        'lana': EwasSpec(key='lana', name='LANA1', uniprot='Q9QR71', species='Human herpesvirus 8', species_ref=HHV8_REF),
        'bub1': EwasSpec(key='bub1', name='BUB1', uniprot='O43683'),
        'cx': ComplexSpec(key='cx', name='LANA1:BUB1', components=['lana', 'bub1']),
        'dimer': ComplexSpec(key='dimer', name='LANA1 dimer', components=['lana', 'lana']),
        'pcna': EwasSpec(key='pcna', name='PCNA', uniprot='P12004')}
    d.reactions = [ReactionSpec(key='r1', name='LANA1 binds BUB1', inputs=['lana', 'bub1'], outputs=['cx']),
                   ReactionSpec(key='r2', name='LANA1 dimerizes', inputs=['lana'], outputs=['dimer']),
                   ReactionSpec(key='r3', name='PCNA is regulated', inputs=['pcna'], outputs=['pcna'],
                                regulations=[RegulationSpec(kind='positive', regulator='cx')])]
    d.pathway = PathwaySpec(name='KSHV latency', reactions=['r1', 'r2', 'r3'])
    return d


def test_emitted_instances_follow_reactomes_virus_host_convention():
    out = emit_user_instances(_virus_draft()).user_instances['newInstances']
    by_name = {i['displayName']: i for i in out}
    species = lambda i, a='species': sorted(x['displayName'] for x in (
        i['attributes'].get(a) if isinstance(i['attributes'].get(a), list) else
        [i['attributes'][a]] if a in i['attributes'] else []))
    assert species(by_name['LANA1']) == ['Human herpesvirus 8']
    assert species(by_name['UniProt:Q9QR71 LANA1']) == ['Human herpesvirus 8']
    assert species(by_name['BUB1']) == ['Homo sapiens']
    cx = by_name['LANA1:BUB1']
    assert species(cx) == ['Homo sapiens'] and species(cx, 'relatedSpecies') == ['Human herpesvirus 8']
    assert species(by_name['LANA1 dimer']) == ['Human herpesvirus 8']          # only viral parts
    assert 'relatedSpecies' not in by_name['LANA1 dimer']['attributes']
    for r in ('LANA1 binds BUB1', 'LANA1 dimerizes', 'PCNA is regulated'):       # the regulator counts too
        assert species(by_name[r]) == ['Homo sapiens'] and species(by_name[r], 'relatedSpecies') == ['Human herpesvirus 8']
    pw = by_name['KSHV latency']
    assert species(pw) == ['Homo sapiens'] and species(pw, 'relatedSpecies') == ['Human herpesvirus 8']


def test_a_human_only_draft_has_no_related_species():
    d = _virus_draft()
    d.participants['lana'] = EwasSpec(key='lana', name='LANA1', uniprot='Q9QR71')
    out = emit_user_instances(d).user_instances['newInstances']
    assert not any('relatedSpecies' in i['attributes'] for i in out)
