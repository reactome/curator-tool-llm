import os

import pytest

from curator_llm.models.reactome import (CatalystSpec, ComplexSpec, DefinedSetSpec, EwasSpec, ReactionSpec,
                                         ReactomeDraft, SimpleEntitySpec)
from curator_llm.ports.events import EventRow
from curator_llm.services.existing_events import find_existing, reaction_accessions


class FakeEvents:
    def __init__(self, citing=(), candidates=(), fail=False):
        self.citing, self.candidates, self.fail, self.calls = list(citing), list(candidates), fail, []

    def events_citing(self, pmid):
        if self.fail:
            raise ConnectionError('graph down')
        return self.citing

    def candidate_events(self, cats, accs, limit=100):
        self.calls.append((cats, accs))
        return self.candidates


def draft():
    d = ReactomeDraft()
    d.participants = {
        'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'ub': EwasSpec(key='ub', name='UB', uniprot='P0CG47'),
        'pub': EwasSpec(key='pub', name='UB', uniprot='P0CG47'), 'atp': SimpleEntitySpec(key='atp', name='ATP'),
        'prkn': EwasSpec(key='prkn', name='PRKN', uniprot='O60260'), 'x': EwasSpec(key='x', name='X'),
        'cx': ComplexSpec(key='cx', name='', components=['prkn', 'pub']),
        'set': DefinedSetSpec(key='set', name='S', members=['ub', 'x'])}
    d.reactions = [
        ReactionSpec(key='r0', name='PINK1 phosphorylates ubiquitin', inputs=['ub', 'atp'], outputs=['pub'],
                     catalyst=CatalystSpec(entity='pink1')),
        ReactionSpec(key='r1', name='Parkin binds pUb', inputs=['prkn', 'pub'], outputs=['cx']),
        ReactionSpec(key='r2', name='mystery', inputs=['x'], outputs=['x'])]
    return d


EXISTING = EventRow(9834945, 'PINK1 phosphorylates Ub on MOM proteins', 'R-HSA-9834945', 'Reaction',
                    ['P0CG47'], ['P0CG47'], ['Q9BXM7'])
MOM = EventRow(7, 'PINK1 phosphorylates Ub on many MOM proteins', 'R-HSA-7', 'Reaction',
               ['Q8IWA4', 'P21796', 'Q9Y277', 'Q96E29', 'P45880'], ['Q8IWA4', 'P21796', 'Q9Y277', 'Q96E29', 'P45880'], ['Q9BXM7'])
GENERIC_E3 = EventRow(8, 'Polyubiquitination of substrate', 'R-HSA-8', 'Reaction',
                      ['P0CG47', 'P0CG48', 'P62987', 'P62979', 'X1'], ['P0CG47', 'P0CG48', 'P62987', 'P62979', 'X1'], ['O60260'])
OTHER = EventRow(1, 'PINK1 is autophosphorylated', 'R-HSA-1', 'Reaction', ['Q9BXM7'], ['Q9BXM7'], ['Q9BXM7'])
UNRELATED = EventRow(2, 'Something else', 'R-HSA-2', 'Reaction', ['Z1'], ['Z2'], ['Z3'])
BINDING = EventRow(3, 'PRKN binds p-S-Ub', 'R-HSA-3', 'BlackBoxEvent', ['O60260', 'P0CG47'], ['O60260', 'P0CG47'], [])


def test_accessions_flatten_complexes_and_sets_and_ignore_small_molecules_and_unresolved():
    d = draft()
    cats, allp = reaction_accessions(d, d.reactions[0])
    assert cats == {'Q9BXM7'} and allp == {'Q9BXM7', 'P0CG47'}
    assert reaction_accessions(d, d.reactions[1])[1] == {'O60260', 'P0CG47'}      # complex flattened
    assert reaction_accessions(d, d.reactions[2])[1] == set()                      # X has no accession


def test_same_requires_catalyst_and_overlap_and_unrelated_events_are_dropped():
    m, issues = find_existing(draft(), None, FakeEvents(candidates=[EXISTING, OTHER, UNRELATED]))
    r0 = [x for x in m if x.reaction_key == 'r0']
    assert r0[0].db_id == 9834945 and r0[0].level == 'same' and r0[0].catalyst_match and r0[0].similarity == 1.0
    assert all(x.db_id != 2 for x in m)                                            # unrelated filtered out
    assert [x.level for x in r0][0] == 'same'                                      # best first
    assert any(i.code == 'existing_reaction_match' and i.severity == 'action' and i.reaction_key == 'r0' for i in issues)


def test_citing_the_paper_is_reported_and_helps_the_score():
    m, issues = find_existing(draft(), '24751536', FakeEvents(citing=[EXISTING, BINDING], candidates=[]))
    paper = [i for i in issues if i.code == 'paper_already_curated']
    assert len(paper) == 1 and 'R-HSA-9834945' in paper[0].message and paper[0].severity == 'info'
    by = {}
    for x in m:
        by.setdefault(x.reaction_key, x)                                           # best match per reaction comes first
    assert by['r0'].cites_pmid and 'cites this paper' in by['r0'].reasons and by['r0'].level == 'same'
    assert by['r1'].db_id == 3 and by['r1'].level == 'same'                        # no catalyst on either side


def test_no_catalyst_search_falls_back_to_participants_and_uncomparable_reactions_say_so():
    ev = FakeEvents(candidates=[BINDING])
    m, issues = find_existing(draft(), None, ev)
    assert ev.calls[1] == ([], ['O60260', 'P0CG47'])                               # r1: participants, no catalyst
    assert any(i.code == 'existing_check_skipped' and i.reaction_key == 'r2' for i in issues)


def test_sharing_only_the_catalyst_is_not_a_match():
    m, _ = find_existing(draft(), None, FakeEvents(candidates=[OTHER]))               # PINK1 autophosphorylation vs PINK1->Ub
    assert [x for x in m if x.reaction_key == 'r0'] == []


def test_a_reaction_whose_only_accession_is_its_catalyst_matches_on_that():
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'), 'atp': SimpleEntitySpec(key='atp', name='ATP')}
    d.reactions = [ReactionSpec(key='r0', name='PINK1 autophosphorylation', inputs=['pink1', 'atp'], outputs=['pink1'],
                                catalyst=CatalystSpec(entity='pink1'))]
    m, _ = find_existing(d, None, FakeEvents(candidates=[OTHER]))
    assert m[0].db_id == 1 and m[0].level == 'same'


def test_similar_when_catalyst_matches_and_a_third_of_the_participants_overlap():
    d = draft()
    d.participants['y'] = EwasSpec(key='y', name='Y', uniprot='Y1')
    d.participants['z'] = EwasSpec(key='z', name='Z', uniprot='Z1')
    d.reactions[0].inputs = ['ub', 'y', 'z']
    partial = EventRow(5, 'PINK1 partly overlaps', 'R-HSA-5', '', ['P0CG47'], [], ['Q9BXM7'])
    m, _ = find_existing(d, None, FakeEvents(candidates=[partial]))
    r0 = [x for x in m if x.reaction_key == 'r0']
    assert r0 and r0[0].level == 'similar' and r0[0].overlap == 0.333 and r0[0].similarity == 0.333


def test_a_big_generic_event_does_not_match_just_because_it_contains_the_draft_accessions():
    d = draft()                                      # r1 'Parkin binds pUb': PRKN + UB (and no catalyst)
    d.reactions[0].catalyst = CatalystSpec(entity='prkn')          # r0 now catalysed by Parkin on ubiquitin
    m, _ = find_existing(d, None, FakeEvents(candidates=[GENERIC_E3]))
    assert [x for x in m if x.reaction_key == 'r0'] == []          # Parkin + Ub inside a 5-accession generic event: similarity 0.2


def test_the_curated_paper_event_is_similar_not_same_when_participants_differ():
    m, issues = find_existing(draft(), '24751536', FakeEvents(citing=[MOM], candidates=[]))
    r0 = [x for x in m if x.reaction_key == 'r0'][0]
    assert r0.level == 'similar' and r0.cites_pmid and r0.catalyst_match and r0.similarity == 0.0
    assert any(i.code == 'existing_reaction_similar' and i.severity == 'info' for i in issues)


def test_lookup_failures_degrade_to_an_issue_and_only_filters_reactions():
    m, issues = find_existing(draft(), '1', FakeEvents(fail=True))
    assert m == [] and issues[0].code == 'existing_check_unavailable' and issues[0].severity == 'warning'
    ev = FakeEvents(candidates=[EXISTING])
    m, _ = find_existing(draft(), None, ev, only=['r0'])
    assert {x.reaction_key for x in m} == {'r0'} and len(ev.calls) == 1


# ── live: the real graph, the real PINK1 paper ─────────────────────────────
@pytest.fixture(scope='module')
def live():
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    try:
        from curator_llm.adapters.neo4j_lookup import Neo4jInstanceLookup
        lk = Neo4jInstanceLookup.from_env()
        lk.find_by_db_id(48887)
        return lk
    except Exception:
        pytest.skip('Reactome Neo4j graph not reachable')


def test_live_graph_finds_the_events_that_already_cite_the_pink1_paper(live):
    names = {e.display_name for e in live.events_citing('24751536')}
    assert 'PINK1 phosphorylates Ub on MOM proteins' in names
    row = next(e for e in live.events_citing('24751536') if e.display_name == 'PINK1 phosphorylates Ub on MOM proteins')
    assert row.catalysts == ['Q9BXM7'] and row.st_id == 'R-HSA-9834945'
    cands = {e.display_name for e in live.candidate_events(['Q9BXM7'], ['Q9BXM7'])}
    assert 'PINK1 is autophosphorylated' in cands and not any(n.lower().startswith('clone of') for n in cands)


def test_live_draft_reaction_matches_the_curated_pink1_phosphorylation(live):
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'),
                      'ub': EwasSpec(key='ub', name='UB', uniprot='P0CG47')}
    d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates ubiquitin at Ser65', inputs=['ub'], outputs=['ub'],
                                catalyst=CatalystSpec(entity='pink1'))]
    m, issues = find_existing(d, '24751536', live)
    top = m[0]
    assert top.level in ('same', 'similar') and top.catalyst_match and top.cites_pmid
    assert any(i.code == 'paper_already_curated' for i in issues)
