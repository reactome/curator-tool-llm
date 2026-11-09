"""An annotation as a network for Cytoscape desktop: a .cyjs network file and a .style.xml visual style.

    python -m curator_llm.cytoscape_export annotations/AML_drug_response.annotation.json      (or scripts/export_cytoscape.sh)
    --out DIR    where to write (default: next to the annotation file)

In Cytoscape: File > Import > Network from File... (the .cyjs), then File > Import > Styles from File... (the .style.xml)
and pick the style "Paper2Path" in the Style panel; apply a layout (Layout > Prefuse Force Directed, or yFiles Organic).

Nodes are entities and reactions; edges run input -> reaction -> output, catalyst and regulators point at the reaction,
parts point at their complex or set (services/network.py). Columns to filter and colour by:
  reactions: name, groups (drug group or other --group-by value), papers, pmids, n_papers (above 1: a shared reaction,
             drawn with a thick border), qa_priority, open_issues, existing_match, reaction_type, summation
  entities:  name, kind (protein | gene | rna | small_molecule | complex | set), uniprot, chebi, species, compartment,
             n_reactions (how many reactions use it; filter out hubs like ATP)
Lists are joined with "; " so every column is plain text.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set
from xml.sax.saxutils import quoteattr

from curator_llm.models.reactome import EwasSpec, SimpleEntitySpec
from curator_llm.models.session import Session
from curator_llm.services.network import build_network

STYLE = 'Paper2Path'


def _where(session: Session) -> Dict[str, Dict[str, List[str]]]:
    """reaction key -> {'groups': [...], 'papers': [...]} from the pathway hierarchy: the top pathway, then a pathway per
    group, then one per paper (services/combine.py). A one-paper annotation has a single pathway: no group, its paper."""
    out: Dict[str, Dict[str, List[str]]] = {}
    top = session.draft.pathway if session.draft else None

    def add(key, field, value):
        lst = out.setdefault(key, {'groups': [], 'papers': []})[field]
        if value and value not in lst:
            lst.append(value)

    def paper_level(pw, group):
        for k in pw.reactions:
            add(k, 'papers', pw.name)
            add(k, 'groups', group)
        for sub in pw.subpathways:
            paper_level(sub, group)

    if top is None:
        return out
    if not top.subpathways:
        for k in top.reactions:
            add(k, 'papers', (session.papers[0].name if session.papers else top.name))
        return out
    for k in top.reactions:
        add(k, 'papers', top.name)
    for sub in top.subpathways:
        if sub.subpathways:                                  # a group: its sub-pathways are papers
            for k in sub.reactions:
                add(k, 'groups', sub.name)
            for paper in sub.subpathways:
                paper_level(paper, sub.name)
        else:                                                # no groups: papers directly under the top
            paper_level(sub, None)
    return out


def _kind(p) -> str:
    if isinstance(p, EwasSpec):
        return {'dna': 'gene', 'rna': 'rna'}.get(p.molecule, 'protein')
    if isinstance(p, SimpleEntitySpec):
        return 'small_molecule'
    return p.kind


def to_cyjs(session: Session) -> dict:
    d = session.draft
    net = build_network(d, session.issues, session.existing, session.key_to_db_id)
    where = _where(session)
    reactions = {r.key: r for r in d.reactions} if d else {}
    uses: Dict[str, Set[str]] = {}
    for e in net['edges']:
        if e['type'] in ('component', 'member', 'precedes'):
            continue
        ent, rx = (e['source'], e['target']) if e['target'].startswith('r:') else (e['target'], e['source'])
        uses.setdefault(ent, set()).add(rx)

    nodes = []
    for n in net['nodes']:
        data = {'id': n['id'], 'name': n['label'], 'node_type': n['type'], 'open_issues': n.get('openIssues', 0)}
        if n['type'] == 'reaction':
            r = reactions[n['key']]
            w = where.get(r.key, {'groups': [], 'papers': []})
            qa = session.qa.get(r.key)
            data.update({'kind': 'reaction', 'key': r.key, 'reaction_type': r.reaction_type,
                         'groups': '; '.join(w['groups']), 'papers': '; '.join(w['papers']),
                         'n_papers': max(len(w['papers']), len(set(r.pmids)), 1), 'pmids': '; '.join(r.pmids),
                         'qa_priority': qa.priority if qa else '', 'existing_match': n.get('existingMatch') or '',
                         'summation': r.summation or ''})
        else:
            p = d.participants[n['key']]
            data.update({'kind': _kind(p), 'key': p.key, 'compartment': n.get('compartment') or '',
                         'uniprot': getattr(p, 'uniprot', None) or '', 'chebi': getattr(p, 'chebi', None) or '',
                         'species': getattr(p, 'species', '') or '',
                         'unresolved': '; '.join(n.get('unresolved') or []),
                         'n_reactions': len(uses.get(n['id'], ()))})
        nodes.append({'data': data})
    edges = [{'data': {'id': e['id'], 'source': e['source'], 'target': e['target'], 'interaction': e['type'],
                       'role': e['type'], 'label': e.get('label') or ''}} for e in net['edges']]
    title = (d.pathway.name if d and d.pathway else None) or (session.papers[0].name if session.papers else session.id)
    return {'format_version': '1.0', 'generated_by': 'curator-tool-llm',
            'data': {'name': title, 'shared_name': title},
            'elements': {'nodes': nodes, 'edges': edges}}


# ── visual style ─────────────────────────────────────────────────────────────
_SHAPE = {'reaction': 'RECTANGLE', 'protein': 'ROUND_RECTANGLE', 'gene': 'PARALLELOGRAM', 'rna': 'HEXAGON',
          'small_molecule': 'ELLIPSE', 'complex': 'OCTAGON', 'set': 'DIAMOND'}
_FILL = {'reaction': '#FFFFFF', 'protein': '#8FC1E3', 'gene': '#C6A9E0', 'rna': '#E8B4D8', 'small_molecule': '#A8D5A2',
         'complex': '#F2C57C', 'set': '#D9D9D9'}
_EDGE = {'input': '#555555', 'output': '#555555', 'catalyst': '#1F77B4', 'positive': '#2CA02C', 'negative': '#D62728',
         'requirement': '#9467BD', 'component': '#BBBBBB', 'member': '#BBBBBB', 'precedes': '#FF7F0E'}
_ARROW = {'input': 'DELTA', 'output': 'DELTA', 'catalyst': 'CIRCLE', 'positive': 'DELTA', 'negative': 'T',
          'requirement': 'DIAMOND', 'component': 'NONE', 'member': 'NONE', 'precedes': 'DELTA'}
_LINE = {'component': 'EQUAL_DASH', 'member': 'DOT', 'precedes': 'EQUAL_DASH'}
_QA = {'fix_first': '#D62728', 'fix': '#FF7F0E', 'review': '#BCBD22', 'ok': '#2CA02C'}


def _discrete(prop: str, default: str, column: str, values: Dict[str, str]) -> str:
    entries = ''.join(f'\n          <discreteMappingEntry attributeValue={quoteattr(k)} value={quoteattr(v)}/>'
                      for k, v in values.items())
    return (f'\n      <visualProperty name="{prop}" default={quoteattr(default)}>'
            f'\n        <discreteMapping attributeName="{column}" attributeType="string">{entries}'
            f'\n        </discreteMapping>\n      </visualProperty>')


def _plain(prop: str, value: str) -> str:
    return f'\n      <visualProperty name="{prop}" default={quoteattr(value)}/>'


def style_xml() -> str:
    """A Cytoscape 3 visual style: shape and colour by kind, a reaction's border by QA priority and thickness by how
    many papers report it, edge colour/arrow/line by role."""
    node = (_plain('NODE_LABEL_FONT_SIZE', '10') + _plain('NODE_WIDTH', '45') + _plain('NODE_HEIGHT', '25')
            + '\n      <visualProperty name="NODE_LABEL" default="">'
              '\n        <passthroughMapping attributeName="name" attributeType="string"/>\n      </visualProperty>'
            + '\n      <visualProperty name="NODE_TOOLTIP" default="">'
              '\n        <passthroughMapping attributeName="name" attributeType="string"/>\n      </visualProperty>'
            + _discrete('NODE_SHAPE', 'ROUND_RECTANGLE', 'kind', _SHAPE)
            + _discrete('NODE_FILL_COLOR', '#DDDDDD', 'kind', _FILL)
            + _discrete('NODE_BORDER_PAINT', '#666666', 'qa_priority', _QA)
            + '\n      <visualProperty name="NODE_BORDER_WIDTH" default="1.0">'
              '\n        <continuousMapping attributeName="n_papers" attributeType="integer">'
              '\n          <continuousMappingPoint attrValue="1" equalValue="1.5" greaterValue="1.5" lesserValue="1.0"/>'
              '\n          <continuousMappingPoint attrValue="2" equalValue="5.0" greaterValue="5.0" lesserValue="5.0"/>'
              '\n        </continuousMapping>\n      </visualProperty>')
    edge = (_plain('EDGE_WIDTH', '1.5')
            + _discrete('EDGE_STROKE_UNSELECTED_PAINT', '#555555', 'role', _EDGE)
            + _discrete('EDGE_TARGET_ARROW_UNSELECTED_PAINT', '#555555', 'role', _EDGE)
            + _discrete('EDGE_TARGET_ARROW_SHAPE', 'DELTA', 'role', _ARROW)
            + _discrete('EDGE_LINE_TYPE', 'SOLID', 'role', _LINE)
            + '\n      <visualProperty name="EDGE_TOOLTIP" default="">'
              '\n        <passthroughMapping attributeName="role" attributeType="string"/>\n      </visualProperty>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<vizmap id="VizMap-{STYLE}" documentVersion="3.1">
  <visualStyle name="{STYLE}">
    <network>
      <visualProperty name="NETWORK_BACKGROUND_PAINT" default="#FFFFFF"/>
    </network>
    <node>
      <dependency name="nodeSizeLocked" value="false"/>
      <dependency name="nodeCustomGraphicsSizeSync" value="true"/>{node}
    </node>
    <edge>
      <dependency name="arrowColorMatchesEdge" value="false"/>{edge}
    </edge>
  </visualStyle>
</vizmap>
'''


def export(path: str, out_dir: Optional[str] = None, say=print) -> tuple:
    with open(path, encoding='utf-8') as f:
        session = Session.model_validate_json(f.read())
    base = os.path.basename(path)
    for suffix in ('.annotation.json', '.json'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    doc = to_cyjs(session)
    net_path = os.path.join(out_dir, f'{base}.cyjs')
    style_path = os.path.join(out_dir, f'{STYLE}.style.xml')
    with open(net_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    with open(style_path, 'w', encoding='utf-8') as f:
        f.write(style_xml())
    nodes = doc['elements']['nodes']
    n_rx = sum(1 for n in nodes if n['data']['node_type'] == 'reaction')
    shared = sum(1 for n in nodes if n['data']['node_type'] == 'reaction' and n['data']['n_papers'] > 1)
    say(f'{path}: {n_rx} reactions ({shared} reported by more than one paper), {len(nodes) - n_rx} entities, '
        f'{len(doc["elements"]["edges"])} edges -> {net_path} (style: {style_path})')
    return net_path, style_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Write an annotation as a Cytoscape network (.cyjs) and visual style.')
    ap.add_argument('annotations', nargs='+', help='<name>.annotation.json file(s), or a session file')
    ap.add_argument('--out', help='output folder (default: next to each annotation file)')
    a = ap.parse_args(argv)
    for p in a.annotations:
        export(p, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
