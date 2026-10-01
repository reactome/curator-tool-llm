from curator_llm.adapters.ols import OlsClient
from curator_llm.adapters.uniprot import RestUniProtClient
from curator_llm.adapters.ws_lookup import WsInstanceLookup
from curator_llm.models.reactome import (CatalystSpec, EwasSpec, GoActivity, ModifiedResidueSpec,
                                         ReactionSpec, ReactomeDraft, SimpleEntitySpec)
from curator_llm.services.resolvers import Resolver
from tests.fakes.ports import FakeInstanceLookup, FakeOntology, FakeUniProt

GK = [
    {'dbId': 70102, 'displayName': 'mitochondrial outer membrane', 'schemaClassName': 'Compartment'},
    {'dbId': 70101, 'displayName': 'cytosol', 'schemaClassName': 'Compartment'},
    {'dbId': 555, 'displayName': 'UniProt:Q9BXM7 PINK1', 'schemaClassName': 'ReferenceGeneProduct', 'identifier': 'Q9BXM7'},
    {'dbId': 600, 'displayName': 'PINK1 [mitochondrial outer membrane]', 'schemaClassName': 'EntityWithAccessionedSequence'},
    {'dbId': 700, 'displayName': 'ChEBI:30616 ATP', 'schemaClassName': 'ReferenceMolecule', 'identifier': '30616', 'name': ['ATP']},
    {'dbId': 710, 'displayName': 'ATP [mitochondrial outer membrane]', 'schemaClassName': 'SimpleEntity'},
    {'dbId': 800, 'displayName': 'protein kinase activity', 'schemaClassName': 'GO_MolecularFunction', 'identifier': '0004672'},
    {'dbId': 900, 'displayName': 'O-phospho-L-serine', 'schemaClassName': 'PsiMod', 'identifier': '00046'},
    {'dbId': 950, 'displayName': 'Kane LA, et al (2014)', 'schemaClassName': 'LiteratureReference', 'pubMedIdentifier': 24751536},
]
UNIPROT = {'Q9BXM7': {'accession': 'Q9BXM7', 'genes': ['PINK1'], 'names': ['Serine/threonine-protein kinase PINK1, mitochondrial'],
                      'reviewed': True, 'organism': 'Homo sapiens'},
           'P0CG48': {'accession': 'P0CG48', 'genes': ['UBC'], 'names': ['Polyubiquitin-C'], 'reviewed': True,
                      'organism': 'Homo sapiens'}}


def make(draft, ontology=None, genes=None):
    return Resolver(FakeInstanceLookup(GK), FakeUniProt(UNIPROT, genes), ontology), draft


def test_existing_entities_references_and_publications_are_reused():
    d = ReactomeDraft()
    d.participants = {
        'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7', compartment_name='mitochondrial outer membrane'),
        'atp': SimpleEntitySpec(key='atp', name='ATP', chebi='CHEBI:30616', compartment_name='OMM')}
    d.reactions = [ReactionSpec(key='r', name='x', inputs=['pink1', 'atp'], pmids=['24751536'],
                                catalyst=CatalystSpec(entity='pink1', activity=GoActivity(identifier='GO:0004672')))]
    res, d = make(d)
    notes = res.resolve(d)
    p = d.participants['pink1']
    assert p.compartment.db_id == 70102 and p.reference_entity.db_id == 555 and p.existing.db_id == 600
    a = d.participants['atp']
    assert a.compartment.db_id == 70102 and a.reference_entity.db_id == 700 and a.existing.db_id == 710   # OMM synonym
    assert d.reactions[0].catalyst.activity.ref.db_id == 800
    assert d.publications['24751536'].db_id == 950
    assert notes == []


def test_unresolvable_things_are_flagged_not_guessed():
    d = ReactomeDraft()
    d.participants = {'x': EwasSpec(key='x', name='FOO1', compartment_name='the nucleolus-ish place'),
                      'm': SimpleEntitySpec(key='m', name='mystery compound')}
    res, d = make(d)
    res.resolve(d)
    assert d.participants['x'].compartment is None and 'compartment' in d.participants['x'].needs_resolution
    assert d.participants['x'].uniprot is None and 'uniprot' in d.participants['x'].needs_resolution
    assert 'chebi' in d.participants['m'].needs_resolution and d.participants['m'].reference_entity is None


def test_bad_uniprot_is_removed_and_mismatch_is_reported():
    d = ReactomeDraft()
    d.participants = {'a': EwasSpec(key='a', name='PINK1', uniprot='Z9ZZZZ'),
                      'b': EwasSpec(key='b', name='PARKIN', uniprot='Q9BXM7')}
    res, d = make(d)
    notes = res.resolve(d)
    assert d.participants['a'].uniprot is None and 'uniprot' in d.participants['a'].needs_resolution
    assert 'uniprot-mismatch' in d.participants['b'].needs_resolution
    assert any('Z9ZZZZ' in n for n in notes) and any('does not match' in n for n in notes)


def test_gene_symbol_search_fills_accession_only_when_unique():
    d = ReactomeDraft()
    d.participants = {'a': EwasSpec(key='a', name='PINK1'), 'b': EwasSpec(key='b', name='UB')}
    res, d = make(d, genes={'PINK1': 'Q9BXM7'})
    res.resolve(d)
    assert d.participants['a'].uniprot == 'Q9BXM7' and d.participants['a'].reference_entity.db_id == 555
    assert d.participants['b'].uniprot is None


def test_modified_residue_resolution_uses_psi_mod_table_and_lookup():
    d = ReactomeDraft()
    d.participants = {'u': EwasSpec(key='u', name='UB', modifications=[
        ModifiedResidueSpec(psi_mod='MOD:00046', residue='S', coordinate=65)])}
    res, d = make(d)
    res.resolve(d)
    m = d.participants['u'].modifications[0]
    assert m.mod_label == 'O-phospho-L-serine' and m.short == 'p' and m.psi_mod_ref.db_id == 900
    assert d.participants['u'].existing is None          # modified entities are never matched by name


def test_go_and_chebi_labels_resolve_through_ols_only_when_unambiguous():
    onto = FakeOntology({('protein kinase activity', 'go'): [{'identifier': 'GO:0004672', 'label': 'protein kinase activity'}],
                         ('adp', 'chebi'): [{'identifier': 'CHEBI:16761', 'label': 'ADP'}],
                         ('ambiguous', 'chebi'): [{'identifier': 'CHEBI:1', 'label': 'a'}, {'identifier': 'CHEBI:2', 'label': 'a'}]})
    d = ReactomeDraft()
    d.participants = {'adp': SimpleEntitySpec(key='adp', name='ADP'), 'amb': SimpleEntitySpec(key='amb', name='ambiguous'),
                      'k': EwasSpec(key='k', name='PINK1', uniprot='Q9BXM7')}
    d.reactions = [ReactionSpec(key='r', name='x', catalyst=CatalystSpec(entity='k', activity=GoActivity(name='protein kinase activity')))]
    res, d = make(d, ontology=onto)
    notes = res.resolve(d)
    assert d.reactions[0].catalyst.activity.identifier == 'GO:0004672' and d.reactions[0].catalyst.activity.ref.db_id == 800
    assert d.participants['adp'].chebi == 'CHEBI:16761' and any('ADP' in n and 'confirm' in n for n in notes)
    assert d.participants['amb'].chebi is None and 'chebi' in d.participants['amb'].needs_resolution


def test_ambiguous_identifier_match_is_not_guessed():
    lookup = FakeInstanceLookup(GK + [{'dbId': 556, 'displayName': 'dup', 'schemaClassName': 'ReferenceGeneProduct', 'identifier': 'Q9BXM7'}])
    assert Resolver(lookup).reference_gene_product('Q9BXM7') is None


# ── adapters, with canned HTTP ─────────────────────────────────────────────
class _R:
    def __init__(self, status=200, data=None):
        self.status_code, self._d, self.content = status, data, b'x' if data is not None else b''

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Http:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, params=None, **kw):
        self.calls.append((url, params))
        return self.resp


def test_ws_lookup_uses_the_real_ws_parameters_and_normalises():
    http = _Http(_R(200, {'dbId': 5, 'displayName': 'cytosol', 'schemaClassName': 'Compartment', 'attributes': {}}))
    lk = WsInstanceLookup('http://ws/', lambda: 'tok', session=http)
    assert lk.find_by_display_name('cytosol', ['Compartment', 'GO_CellularComponent'])['dbId'] == 5
    assert http.calls[0] == ('http://ws/api/curation/findByDisplayName',
                             {'displayName': 'cytosol', 'classNames': 'Compartment,GO_CellularComponent'})
    http.resp = _R(200, {'instances': [{'dbId': 9, 'displayName': 'x', 'schemaClassName': 'ReferenceGeneProduct'}], 'totalCount': 1})
    assert lk.search('ReferenceGeneProduct', 'identifier', 'Q9BXM7')[0]['dbId'] == 9
    assert http.calls[1] == ('http://ws/api/curation/searchInstances/ReferenceGeneProduct/0/5',
                             {'attributes': 'identifier', 'operands': 'equal', 'searchKeys': 'Q9BXM7'})
    http.resp = _R(404)
    assert lk.find_by_db_id(1) is None


def test_uniprot_client_parses_entry_and_rejects_ambiguous_search():
    entry = {'primaryAccession': 'Q9BXM7', 'entryType': 'UniProtKB reviewed (Swiss-Prot)',
             'genes': [{'geneName': {'value': 'PINK1'}}], 'organism': {'scientificName': 'Homo sapiens'},
             'proteinDescription': {'recommendedName': {'fullName': {'value': 'Serine/threonine-protein kinase PINK1'}}}}
    c = RestUniProtClient(session=_Http(_R(200, entry)))
    info = c.fetch('q9bxm7')
    assert info['genes'] == ['PINK1'] and info['reviewed'] and 'PINK1' in info['names'][0]
    assert RestUniProtClient(session=_Http(_R(404))).fetch('NOPE') is None
    assert RestUniProtClient(session=_Http(_R(200, {'entryType': 'Inactive'}))).fetch('OLD') is None
    assert RestUniProtClient(session=_Http(_R(200, {'results': [{'primaryAccession': 'A'}, {'primaryAccession': 'B'}]}))).search_gene('X') is None
    assert RestUniProtClient(session=_Http(_R(200, {'results': [{'primaryAccession': 'A'}]}))).search_gene('X') == 'A'


def test_ols_client_parses_hits():
    docs = {'response': {'docs': [{'obo_id': 'GO:0004672', 'label': 'protein kinase activity'}]}}
    assert OlsClient(session=_Http(_R(200, docs))).search('protein kinase activity', 'go') == [
        {'identifier': 'GO:0004672', 'label': 'protein kinase activity'}]


def test_a_failing_external_service_degrades_to_a_note_and_is_not_retried():
    class Boom:
        calls = 0
        def search(self, name, ontology):
            Boom.calls += 1
            raise TimeoutError('slow')
    class BoomUniProt(FakeUniProt):
        def fetch(self, accession):
            raise ConnectionError('down')
    d = ReactomeDraft()
    d.participants = {'a': EwasSpec(key='a', name='PINK1', uniprot='Q9BXM7'),
                      'b': SimpleEntitySpec(key='b', name='ADP'), 'c': SimpleEntitySpec(key='c', name='GTP')}
    d.reactions = [ReactionSpec(key='r', name='x', catalyst=CatalystSpec(entity='a', activity=GoActivity(name='kinase')))]
    res = Resolver(FakeInstanceLookup(GK), BoomUniProt({}), Boom())
    notes = res.resolve(d)
    assert Boom.calls == 1                                   # OLS marked down after the first failure
    assert sum('OLS unavailable' in n for n in notes) == 1 and any('UniProt unavailable' in n for n in notes)
    assert 'uniprot-unchecked' in d.participants['a'].needs_resolution and d.participants['a'].uniprot == 'Q9BXM7'
    assert d.participants['a'].reference_entity.db_id == 555           # gk_central lookups still work


def test_ws_login_token_getter_caches_then_relogs_in():
    from curator_llm.adapters.ws_login import WsLoginTokenGetter

    class Http:
        n = 0
        def post(self, url, json=None, timeout=None):
            Http.n += 1
            return type('R', (), {'status_code': 200, 'text': f'"tok{Http.n}"'})()
    now = [0.0]
    g = WsLoginTokenGetter('http://ws/', 'u', 'p', session=Http(), clock=lambda: now[0])
    assert g() == 'tok1' and g() == 'tok1'
    now[0] = 300
    assert g() == 'tok2'


def test_go_function_label_is_found_in_the_graph_before_any_external_call():
    gk = GK + [{'dbId': 801, 'displayName': 'ubiquitin-protein transferase activity', 'schemaClassName': 'GO_MolecularFunction', 'identifier': '0004842'}]
    d = ReactomeDraft()
    d.participants = {'k': EwasSpec(key='k', name='PRKN')}
    d.reactions = [ReactionSpec(key='r', name='x', catalyst=CatalystSpec(entity='k', activity=GoActivity(name='ubiquitin-protein transferase activity')))]

    class NoOls:
        def search(self, *a):
            raise AssertionError('OLS must not be called')
    res = Resolver(FakeInstanceLookup(gk), None, NoOls())
    res.resolve(d)
    a = d.reactions[0].catalyst.activity
    assert a.ref.db_id == 801 and a.identifier == 'GO:0004842' and res.notes == []


def test_mixed_case_protein_names_are_searched_in_uniprot():
    d = ReactomeDraft()
    d.participants = {'p': EwasSpec(key='p', name='Parkin')}
    res = Resolver(FakeInstanceLookup(GK), FakeUniProt(UNIPROT, {'Parkin': 'O60260'}))
    res.resolve(d)
    assert d.participants['p'].uniprot == 'O60260'


def test_uniprot_gene_search_is_cached_and_resolver_notes_are_unique():
    http = _Http(_R(200, {'results': [{'primaryAccession': 'O60260'}]}))
    c = RestUniProtClient(session=http)
    assert c.search_gene('Parkin') == 'O60260' and c.search_gene('parkin') == 'O60260' and len(http.calls) == 1
    d = ReactomeDraft()
    d.participants = {'a': EwasSpec(key='a', name='Parkin'), 'b': EwasSpec(key='b', name='Parkin')}
    notes = Resolver(FakeInstanceLookup(GK), FakeUniProt({}, {'Parkin': 'O60260'})).resolve(d)
    assert len(notes) == 1
