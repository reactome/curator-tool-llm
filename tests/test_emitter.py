from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (CatalystSpec, ComplexSpec, EwasSpec, ExistingRef, GoActivity,
                                         ModifiedResidueSpec, PathwaySpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec, SimpleEntitySpec)
from curator_llm.services import schema_index
from curator_llm.services.emitter import emit_user_instances

OMM = ExistingRef(db_id=70102, display_name='mitochondrial outer membrane', schema_class='Compartment')
PINK1_REF = ExistingRef(db_id=555, display_name='UniProt:Q9BXM7 PINK1', schema_class='ReferenceGeneProduct')


def draft():
    d = ReactomeDraft()
    d.participants = {
        'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7', reference_entity=PINK1_REF, compartment=OMM),
        'ub': EwasSpec(key='ub', name='UB', uniprot='P0CG48', compartment=OMM),
        'pub': EwasSpec(key='pub', name='UB', uniprot='P0CG48', compartment=OMM,
                        modifications=[ModifiedResidueSpec(psi_mod='MOD:00046', mod_label='O-phospho-L-serine',
                                                          residue='S', coordinate=65)]),
        'atp': SimpleEntitySpec(key='atp', name='ATP', chebi='CHEBI:30616', compartment=OMM),
        'adp': SimpleEntitySpec(key='adp', name='ADP', chebi='CHEBI:456216', compartment=OMM),
        'ppk': ComplexSpec(key='ppk', name='', components=['pink1', 'ub'], compartment=OMM),
        'parkin': EwasSpec(key='parkin', name='PRKN', compartment_name='cytosol'),
    }
    d.reactions = [
        ReactionSpec(key='r1', name='PINK1 phosphorylates ubiquitin at Ser65', inputs=['ub', 'atp'],
                     outputs=['pub', 'adp'], pmids=['24751536'], summation='PINK1 phosphorylates Ub.',
                     catalyst=CatalystSpec(entity='pink1',
                                           activity=GoActivity(identifier='GO:0004672', name='protein kinase activity')),
                     regulations=[RegulationSpec(kind='positive', regulator='parkin', note='seen with CCCP')],
                     evidence_ids=['ev-001', 'ev-002']),
        ReactionSpec(key='r2', name='Parkin binds pUb', inputs=['parkin', 'pub'], outputs=['ppk'],
                     preceding=['r1'], pmids=['24751536']),
    ]
    d.pathway = PathwaySpec(name='PINK1-PRKN signaling (proposed)', reactions=['r1', 'r2'])
    return d


def ev():
    return {'ev-001': Evidence(id='ev-001', quote='q1', supports=['catalystActivity', 'output']),
            'ev-002': Evidence(id='ev-002', quote='q2', supports=['regulatedBy[0]'])}


def test_every_emitted_attribute_is_in_the_schema():
    r = emit_user_instances(draft(), ev())          # emitter raises on any non-schema attribute
    assert r.user_instances['newInstances']
    for inst in r.user_instances['newInstances']:
        assert schema_index.check_instance(inst) == []


def test_dbid_and_display_name_are_also_attributes():
    # the frontend's instance view reads them from the attributes, as for an instance a curator creates
    for inst in emit_user_instances(draft(), ev()).user_instances['newInstances']:
        assert inst['attributes']['dbId'] == inst['dbId']
        assert inst['attributes']['displayName'] == inst['displayName']


def test_ids_are_unique_negative_and_references_resolve():
    r = emit_user_instances(draft(), ev())
    new = r.user_instances['newInstances']
    ids = [i['dbId'] for i in new]
    assert len(ids) == len(set(ids)) and all(i < 0 for i in ids)
    new_ids = set(ids)

    def refs(v):
        if isinstance(v, dict) and 'dbId' in v:
            yield v
        elif isinstance(v, list):
            for x in v:
                yield from refs(x)
    for inst in new:
        for val in inst['attributes'].values():
            for ref in refs(val):
                assert ref['dbId'] > 0 or ref['dbId'] in new_ids, (inst['displayName'], ref)
                assert set(ref) == {'dbId', 'displayName', 'schemaClassName'}      # shells only


def test_existing_instances_become_shells_and_are_not_re_emitted():
    r = emit_user_instances(draft(), ev())
    new = r.user_instances['newInstances']
    assert not any(i['dbId'] > 0 for i in new)
    pink1 = next(i for i in new if i['displayName'] == 'PINK1 [mitochondrial outer membrane]')
    assert pink1['attributes']['referenceEntity']['dbId'] == 555
    assert pink1['attributes']['compartment'] == [{'dbId': 70102, 'displayName': 'mitochondrial outer membrane',
                                                   'schemaClassName': 'Compartment'}]


def test_modified_residue_psi_mod_and_coordinate():
    r = emit_user_instances(draft(), ev())
    new = r.user_instances['newInstances']
    pub = next(i for i in new if i['displayName'].startswith('p-S65-UB'))
    mod = next(i for i in new if i['schemaClassName'] == 'ModifiedResidue')
    assert pub['attributes']['hasModifiedResidue'][0]['dbId'] == mod['dbId']
    assert mod['attributes']['coordinate'] == 65 and mod['attributes']['label'] == 'O-phospho-L-serine'
    assert any('PSI-MOD' in w for w in r.warnings)


def test_reaction_structure_catalyst_regulation_and_preceding_event():
    r = emit_user_instances(draft(), ev())
    new = r.user_instances['newInstances']
    by = {i['displayName']: i for i in new}
    r1, r2 = by['PINK1 phosphorylates ubiquitin at Ser65'], by['Parkin binds pUb']
    assert r1['schemaClassName'] == 'Reaction'
    cat = next(i for i in new if i['schemaClassName'] == 'CatalystActivity')
    assert r1['attributes']['catalystActivity'][0]['dbId'] == cat['dbId']
    assert cat['displayName'].startswith('protein kinase activity of PINK1')
    reg = next(i for i in new if i['schemaClassName'] == 'PositiveRegulation')
    assert reg['attributes']['regulatedEntity'][0]['dbId'] == r1['dbId']
    assert r2['attributes']['precedingEvent'][0]['dbId'] == r1['dbId']
    assert len(r1['attributes']['input']) == 2 and len(r1['attributes']['output']) == 2


def test_complex_components_and_shared_reference_entities():
    r = emit_user_instances(draft(), ev())
    new = r.user_instances['newInstances']
    cx = next(i for i in new if i['schemaClassName'] == 'Complex')
    assert [c['displayName'] for c in cx['attributes']['hasComponent']] == [
        'PINK1 [mitochondrial outer membrane]', 'UB [mitochondrial outer membrane]']
    assert sum(i['schemaClassName'] == 'ReferenceGeneProduct' for i in new) == 1     # UB shared by ub and pub
    assert sum(i['schemaClassName'] == 'LiteratureReference' for i in new) == 1      # one PMID, one instance


def test_evidence_links_point_at_the_instance_each_quote_supports():
    r = emit_user_instances(draft(), ev())
    new = {i['dbId']: i for i in r.user_instances['newInstances']}
    links = {(l['evidenceId'], new[l['instanceDbId']]['schemaClassName']): l['field'] for l in r.evidence_links}
    assert links[('ev-001', 'CatalystActivity')] == 'catalystActivity'
    assert links[('ev-001', 'Reaction')] == 'output'
    assert links[('ev-002', 'PositiveRegulation')] == 'regulatedBy[0]'


def test_unresolved_compartment_and_missing_ids_are_reported_not_guessed():
    r = emit_user_instances(draft(), ev())
    assert any('PRKN' in w and 'compartment' in w for w in r.warnings)
    parkin = next(i for i in r.user_instances['newInstances'] if i['displayName'].startswith('PRKN'))
    assert 'compartment' not in parkin['attributes'] and 'referenceEntity' not in parkin['attributes']


def test_pathway_and_existing_reaction_handling():
    d = draft()
    d.reactions[0].existing = ExistingRef(db_id=999, display_name='already curated', schema_class='Reaction')
    r = emit_user_instances(d, ev())
    assert any('already in Reactome' in w for w in r.warnings)
    assert not any(i['displayName'] == 'PINK1 phosphorylates ubiquitin at Ser65' for i in r.user_instances['newInstances'])
    d.pathway.existing = ExistingRef(db_id=1, display_name='Mitophagy', schema_class='Pathway')
    r = emit_user_instances(d, ev())
    assert not any(i['schemaClassName'] == 'Pathway' for i in r.user_instances['newInstances'])
    assert any('Mitophagy' in w for w in r.warnings)


def test_output_is_json_serialisable_user_instances():
    import json
    r = emit_user_instances(draft(), ev())
    assert set(r.user_instances) == {'newInstances', 'updatedInstances', 'deletedInstances', 'bookmarks'}
    json.dumps(r.user_instances)
