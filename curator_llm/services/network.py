"""The extracted reactions as a network: one node per participant and per reaction, one edge per role.

Nodes are `entity` (kind ewas | simple | complex | set) or `reaction`. Edges run input -> reaction -> output; a
catalyst or regulator points at the reaction (`type` catalyst | positive | negative | requirement); `component` and
`member` edges point from a part to the complex or set it belongs to; `precedes` links one reaction to the next.
Ids are prefixed (`e:` entity, `r:` reaction) because participant and reaction keys come from different spaces.
Only participants a reaction uses, directly or through a complex or set, are included."""
from typing import Dict, List, Optional

from curator_llm.models.reactome import ComplexSpec, DefinedSetSpec, ReactomeDraft
from curator_llm.models.session import ExistingMatch, Issue

_RANK = {'same': 2, 'similar': 1}


def build_network(draft: Optional[ReactomeDraft], issues: List[Issue], existing: List[ExistingMatch],
                  key_to_db_id: Optional[Dict[str, int]] = None) -> dict:
    if draft is None:
        return {'nodes': [], 'edges': []}
    key_to_db_id = key_to_db_id or {}
    open_by_reaction: Dict[str, int] = {}
    open_by_participant: Dict[str, int] = {}
    for i in issues:
        if i.status != 'open':
            continue
        if i.reaction_key:
            open_by_reaction[i.reaction_key] = open_by_reaction.get(i.reaction_key, 0) + 1
        if i.participant_key:
            open_by_participant[i.participant_key] = open_by_participant.get(i.participant_key, 0) + 1
    match: Dict[str, str] = {}
    for m in existing:
        if _RANK.get(m.level, 0) > _RANK.get(match.get(m.reaction_key, ''), 0):
            match[m.reaction_key] = m.level

    edges: List[dict] = []

    def edge(kind: str, source: str, target: str, **extra):
        edges.append({'id': f'{kind}:{len(edges)}', 'type': kind, 'source': source, 'target': target, **extra})

    used: List[str] = []             # participant keys, in first-use order

    def use(key: str):
        if key in draft.participants and key not in used:
            used.append(key)

    for r in draft.reactions:
        for k in r.inputs:
            use(k)
            edge('input', f'e:{k}', f'r:{r.key}')
        for k in r.outputs:
            use(k)
            edge('output', f'r:{r.key}', f'e:{k}')
        if r.catalyst:
            use(r.catalyst.entity)
            edge('catalyst', f'e:{r.catalyst.entity}', f'r:{r.key}',
                 label=(r.catalyst.activity.name if r.catalyst.activity else None))
        for g in r.regulations:
            use(g.regulator)
            edge(g.kind, f'e:{g.regulator}', f'r:{r.key}', label=g.note)
        for prev in r.preceding:
            edge('precedes', f'r:{prev}', f'r:{r.key}')

    i = 0
    while i < len(used):               # components and members may bring in participants no reaction names
        p = draft.participants[used[i]]
        parts = p.components if isinstance(p, ComplexSpec) else p.members if isinstance(p, DefinedSetSpec) else []
        for k in parts:
            use(k)
            edge('component' if isinstance(p, ComplexSpec) else 'member', f'e:{k}', f'e:{p.key}')
        i += 1

    nodes: List[dict] = []
    for k in used:
        p = draft.participants[k]
        nodes.append({'id': f'e:{k}', 'type': 'entity', 'key': k, 'kind': p.kind, 'label': p.name,
                      'compartment': p.compartment_name,
                      'unresolved': list(p.needs_resolution),
                      'openIssues': open_by_participant.get(k, 0)})
    for r in draft.reactions:
        nodes.append({'id': f'r:{r.key}', 'type': 'reaction', 'key': r.key, 'label': r.name,
                      'reactionType': r.reaction_type, 'dbId': key_to_db_id.get(r.key),
                      'openIssues': open_by_reaction.get(r.key, 0), 'existingMatch': match.get(r.key)})
    known = {n['id'] for n in nodes}      # a reaction can name a key that is not a participant (or a preceding one that is gone)
    edges = [e for e in edges if e['source'] in known and e['target'] in known]
    return {'nodes': nodes, 'edges': edges}
