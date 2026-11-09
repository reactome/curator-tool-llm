"""An annotation as a Cytoscape network: drug groups and papers per reaction from the pathway hierarchy, a shared
reaction counted once with all its papers, and a well-formed style."""
import json
import xml.dom.minidom

from curator_llm.cytoscape_export import export, style_xml, to_cyjs
from curator_llm.services.combine import combine
from tests.test_combine import judge, parts


def test_a_combined_annotation_becomes_a_network_with_groups_and_shared_reactions(tmp_path):
    s = combine(parts(), 'AML drug response', 'wgm', judge=judge)
    doc = to_cyjs(s)
    nodes = {n['data']['id']: n['data'] for n in doc['elements']['nodes']}
    shared = nodes['r:001_r0']
    assert shared['n_papers'] == 2 and shared['groups'] == 'GIL' and shared['pmids'] == '111; 222'
    assert shared['papers'] == 'Paper one (row 1); Paper two (row 2)'
    assert nodes['r:003_r0']['groups'] == 'VEN' and nodes['e:gsk3b']['kind'] == 'protein'
    assert nodes['e:gsk3b']['uniprot'] == 'P49841' and 'e:gil' not in nodes          # used by no reaction
    assert all(e['data']['source'] in nodes and e['data']['target'] in nodes for e in doc['elements']['edges'])
    assert {e['data']['role'] for e in doc['elements']['edges']} >= {'input', 'output', 'catalyst'}
    f = tmp_path / 'AML.annotation.json'
    f.write_text(s.model_dump_json())
    net, style = export(str(f), say=lambda m: None)
    assert net.endswith('AML.cyjs') and json.load(open(net))['data']['name'] == 'AML drug response'
    xml.dom.minidom.parseString(style_xml())
