import os

import pytest

from curator_llm.adapters.neo4j_lookup import Neo4jInstanceLookup, search_query
from curator_llm.models.reactome import EwasSpec, ReactomeDraft, SimpleEntitySpec
from curator_llm.services.resolvers import Resolver
from tests.fakes.ports import FakeInstanceLookup


def test_query_builder_validates_names_and_uses_parameters_for_values():
    cypher, extra = search_query('ReferenceGeneProduct', 'identifier', 'equal', 3)
    assert 'n.`identifier`' in cypher and '$v' in cypher and 'Q9BXM7' not in cypher and extra == {'limit': 3}
    cypher, _ = search_query('ReferenceMolecule', 'name', 'equal', 2)         # list attribute
    assert 'any(x IN n.`name`' in cypher
    for bad in [('Foo', 'identifier'), ('ReferenceGeneProduct', 'identifier) DETACH DELETE n //'),
                ('ReferenceGeneProduct` DETACH DELETE n //', 'identifier'), ('ReferenceGeneProduct', 'notAnAttribute')]:
        with pytest.raises(ValueError):
            search_query(*bad, 'equal', 1)
    with pytest.raises(ValueError):
        search_query('ReferenceGeneProduct', 'identifier', 'drop', 1)


def test_resolver_prefers_the_exact_class_over_isoforms_sharing_an_identifier():
    gk = [{'dbId': 1, 'displayName': 'UniProt:Q9BXM7 PINK1', 'schemaClassName': 'ReferenceGeneProduct', 'identifier': 'Q9BXM7'},
          {'dbId': 2, 'displayName': 'UniProt:Q9BXM7-2 PINK1', 'schemaClassName': 'ReferenceIsoform', 'identifier': 'Q9BXM7'}]
    assert Resolver(FakeInstanceLookup(gk)).reference_gene_product('Q9BXM7').db_id == 1


# ── integration: needs the Reactome graph from .env; skipped when it is not reachable ─────
def _live():
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    try:
        lk = Neo4jInstanceLookup.from_env()
        lk.find_by_db_id(48887)
        return lk
    except Exception:
        return None


@pytest.fixture(scope='module')
def live():
    lk = _live()
    if lk is None:
        pytest.skip('Reactome Neo4j graph not reachable')
    yield lk
    lk.close()


def test_live_lookups_resolve_pink1_references(live):
    assert live.find_by_display_name('mitochondrial outer membrane', ['Compartment'])['dbId'] == 17906
    assert live.find_by_db_id(48887)['displayName'] == 'Homo sapiens'
    r = Resolver(live)
    assert r.reference_gene_product('Q9BXM7').db_id == 152115                # canonical, not the isoforms
    assert r.reference_molecule(None, 'ATP').display_name.startswith('ATP(4-)')
    assert r.go_function('GO:0004672', None)[1].db_id == 4030
    assert r.psi_mod('MOD:00046').db_id == 445687
    assert r.publication('24751536').db_id == 9839449


def test_live_draft_resolution_reuses_existing_entities(live):
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7', compartment_name='mitochondrial outer membrane'),
                      'atp': SimpleEntitySpec(key='atp', name='ATP', compartment_name='cytosol')}
    Resolver(live).resolve(d)
    assert d.participants['pink1'].existing.db_id == 5205653 and d.participants['atp'].existing.db_id == 113592
