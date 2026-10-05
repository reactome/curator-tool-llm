from fastapi.testclient import TestClient

from curator_llm.models.reactome import (CatalystSpec, ComplexSpec, EwasSpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec, SimpleEntitySpec)
from curator_llm.models.session import ExistingMatch, Issue
from curator_llm.services.network import build_network
from tests.test_edit_api import H, env  # noqa: F401  (the fixture)


def draft():
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1', uniprot='Q9BXM7'),
                      'ub': EwasSpec(key='ub', name='UB', needs_resolution=['uniprot']),
                      'atp': SimpleEntitySpec(key='atp', name='ATP'),
                      'cccp': SimpleEntitySpec(key='cccp', name='CCCP'),
                      'cx': ComplexSpec(key='cx', name='Parkin:Ub', components=['ub', 'parkin']),
                      'parkin': EwasSpec(key='parkin', name='Parkin'),
                      'unused': SimpleEntitySpec(key='unused', name='not in any reaction')}
    d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates Ub', inputs=['ub', 'atp'], outputs=['ub'],
                                catalyst=CatalystSpec(entity='pink1'),
                                regulations=[RegulationSpec(kind='positive', regulator='cccp')]),
                   ReactionSpec(key='r1', name='Parkin binds Ub', inputs=['ub'], outputs=['cx'], preceding=['r0'])]
    return d


def test_network_has_a_node_per_used_participant_and_reaction_and_an_edge_per_role():
    n = build_network(draft(), [], [])
    ids = {x['id'] for x in n['nodes']}
    assert ids == {'e:pink1', 'e:ub', 'e:atp', 'e:cccp', 'e:cx', 'e:parkin', 'r:r0', 'r:r1'}   # 'unused' left out
    kinds = [(e['type'], e['source'], e['target']) for e in n['edges']]
    assert ('input', 'e:ub', 'r:r0') in kinds and ('output', 'r:r0', 'e:ub') in kinds
    assert ('catalyst', 'e:pink1', 'r:r0') in kinds and ('positive', 'e:cccp', 'r:r0') in kinds
    assert ('component', 'e:parkin', 'e:cx') in kinds       # reached only through the complex
    assert ('precedes', 'r:r0', 'r:r1') in kinds
    assert len({e['id'] for e in n['edges']}) == len(n['edges'])


def test_network_carries_open_issues_unresolved_and_the_best_existing_match():
    issues = [Issue(source='qa', severity='warning', code='c', message='m', reaction_key='r0'),
              Issue(source='qa', severity='info', code='c', message='m', reaction_key='r0', status='resolved'),
              Issue(source='resolver', severity='info', code='c', message='m', participant_key='ub')]
    ex = [ExistingMatch(reaction_key='r0', db_id=1, display_name='x', st_id='R-1', level='similar', score=.5),
          ExistingMatch(reaction_key='r0', db_id=2, display_name='y', st_id='R-2', level='same', score=.9)]
    by = {x['id']: x for x in build_network(draft(), issues, ex, {'r0': -5})['nodes']}
    assert by['r:r0']['openIssues'] == 1 and by['r:r0']['existingMatch'] == 'same' and by['r:r0']['dbId'] == -5
    assert by['e:ub']['openIssues'] == 1 and by['e:ub']['unresolved'] == ['uniprot']
    assert by['r:r1']['existingMatch'] is None


def test_network_of_a_session_without_a_draft_is_empty():
    assert build_network(None, [], []) == {'nodes': [], 'edges': []}


def test_network_route_returns_the_graph_and_is_private(env):  # noqa: F811
    c, sid, _ = env
    r = c.get(f'/api/llm/sessions/{sid}/network', headers=H())
    assert r.status_code == 200
    assert {n['id'] for n in r.json()['nodes']} >= {'e:pink1', 'r:r0'}
    assert c.get(f'/api/llm/sessions/{sid}/network', headers=H('bob')).status_code == 404
    assert c.get(f'/api/llm/sessions/{sid}/network').status_code == 401
