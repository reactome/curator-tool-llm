"""Combine several annotations (one per paper) into one, under a pathway hierarchy:

    top pathway
    └─ one sub-pathway per group (say, a drug), if groups are given
       └─ one sub-pathway per paper, holding that paper's reactions

Done in code from the finished per-paper annotations, so nothing is extracted again and no single model call has to
cover hundreds of reactions. Entities are shared: the same protein (UniProt accession and modifications), molecule
(ChEBI) or complex is one participant, wherever it came from. Reactions keep a per-paper prefix on their key. Within a
group, reactions that the merge's duplicate judge calls the same become ONE reaction citing every paper that reports it,
listed under each of those papers' pathways (the judge's verdicts are cached, as in run_merge.py).

Quotes, issues, QA results and the papers' text come along; the caller re-runs the automatic checks and suggestions.
"""
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (ComplexSpec, DefinedSetSpec, EwasSpec, PathwaySpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec, SimpleEntitySpec)
from curator_llm.models.session import ChangeLogEntry, Issue, PaperInfo, Session

_REGULATION = {'positive': 'positiveRegulation', 'negative': 'negativeRegulation', 'requirement': 'requirement'}


@dataclass
class Part:
    session: Session        # one finished annotation (usually of one paper)
    prefix: str             # put before its reaction keys and quote ids, e.g. "039"
    title: str              # its pathway's name (the paper's title)
    group: str = ''         # the sub-pathway it goes under ('' = directly under the top pathway)


# ── entities: one participant per real entity ────────────────────────────────
def _compartment(p) -> object:
    return p.compartment.db_id if p.compartment else (p.compartment_name or '').strip().lower()


def _identity(p, by_key: Dict[str, object], memo: Dict[str, tuple]) -> tuple:
    if p.key in memo:
        return memo[p.key]
    memo[p.key] = ('cycle', p.key)                     # a malformed complex that contains itself stays apart
    # the name, ignoring case and punctuation ("BCL-2" = "BCL2"), is part of every identity: within one paper two
    # participants with the same accession but different names are different states ("CDK2 (active)", "CDK2
    # (inhibited)"), and must stay two
    name = re.sub(r'[^a-z0-9]+', '', p.name.lower())
    if isinstance(p, EwasSpec):
        mods = tuple(sorted((m.psi_mod or '', m.residue or '', m.coordinate or 0) for m in p.modifications))
        ident = ('ewas', p.molecule, (p.uniprot or '').upper(), name, mods, p.start_coordinate, p.end_coordinate,
                 _compartment(p), p.species)
    elif isinstance(p, SimpleEntitySpec):
        ident = ('simple', (p.chebi or '').upper(), name, _compartment(p))
    else:
        parts = p.components if isinstance(p, ComplexSpec) else p.members
        inner = tuple(sorted(str(_identity(by_key[k], by_key, memo)) for k in parts if k in by_key))
        ident = ('complex' if isinstance(p, ComplexSpec) else 'set', name, inner, _compartment(p))
    memo[p.key] = ident
    return ident


# ── what the duplicate judge reads ───────────────────────────────────────────
def as_record(d: ReactomeDraft, r: ReactionSpec, source: str) -> dict:
    """A draft reaction in the shape run_merge's records have, for the judge and the candidate filters."""
    nm = lambda k: d.participants[k].name if k in d.participants else k
    a = {'name': r.name, 'reactionType': r.reaction_type, 'input': [nm(k) for k in r.inputs],
         'output': [nm(k) for k in r.outputs],
         'catalystActivity': ({'catalyst': nm(r.catalyst.entity),
                               'molecularFunction': (r.catalyst.activity.name if r.catalyst.activity else None)}
                              if r.catalyst else None),
         'regulatedBy': [{'regulationType': _REGULATION.get(g.kind, g.kind), 'regulator': nm(g.regulator), 'note': g.note}
                         for g in r.regulations],
         'compartment': r.compartment.display_name if r.compartment else None}
    return {'source': source, 'annotation_result': a}


def shared_reactions(d: ReactomeDraft, groups: Dict[str, List[Tuple[str, str]]], judge: Callable,
                     map_fn: Callable = map, max_candidates: int = 10) -> Tuple[Dict[str, str], dict]:
    """Within each group (name -> [(reaction key, paper source)]), which reactions are the same as one from another
    paper. Returns ({reaction key -> the key it becomes}, stats)."""
    from curator_llm.services.dedup_prefilter import may_be_duplicates
    from curator_llm.services.merge_support import TfidfSimilarity, merge_across_papers
    by_key = {r.key: r for r in d.reactions}
    alias: Dict[str, str] = {}
    total = {'judged': 0, 'merged': 0, 'groups': 0}
    for _, members in groups.items():
        records = [dict(as_record(d, by_key[k], src), key=k) for k, src in members if k in by_key]
        if len({x['source'] for x in records}) < 2:
            continue
        sim = TfidfSimilarity([x['annotation_result'] for x in records])
        found, stats = merge_across_papers(
            records, prefilter=lambda a, b: may_be_duplicates(a['annotation_result'], b['annotation_result']),
            similar=lambda a, b: sim(a['annotation_result'], b['annotation_result']),
            judge=lambda a, b: judge(a['annotation_result'], b['annotation_result']),
            combine=lambda rep, r: rep, max_candidates=max_candidates, map_fn=map_fn)
        total['judged'] += stats['judged']
        for g in found:
            keys = [records[i]['key'] for i in g['members']]
            if len(keys) > 1:
                total['merged'] += len(keys) - 1
                total['groups'] += 1
                for k in keys[1:]:
                    alias[k] = keys[0]
    return alias, total


def _fold(into: ReactionSpec, other: ReactionSpec) -> None:
    """`other` is the same reaction from another paper: `into` now cites that paper's quotes and references too."""
    into.evidence_ids = list(dict.fromkeys(into.evidence_ids + other.evidence_ids))
    into.pmids = list(dict.fromkeys(into.pmids + other.pmids))
    have = {(g.kind, g.regulator) for g in into.regulations}
    into.regulations += [RegulationSpec(**g.model_dump()) for g in other.regulations if (g.kind, g.regulator) not in have]
    into.preceding = list(dict.fromkeys(into.preceding + other.preceding))
    if other.summation.strip() and other.summation.strip() not in into.summation:
        into.summation = (into.summation.rstrip() + '\n\n' + other.summation.strip()).strip()


# ── combining ────────────────────────────────────────────────────────────────
def combine(parts: List[Part], title: str, owner: str, judge: Optional[Callable] = None,
            map_fn: Callable = map, say: Callable[[str], None] = lambda m: None) -> Session:
    d = ReactomeDraft()
    evidence: List[Evidence] = []
    issues: List[Issue] = []
    qa, papers, texts, existing, usage = {}, [], {}, [], []
    global_key: Dict[tuple, str] = {}             # entity identity -> its participant key in the combined draft
    paper_reactions: List[Tuple[Part, List[str], str]] = []    # (part, its reaction keys, its main source)
    for part in parts:
        s, pre = part.session, part.prefix
        if s.draft is None:
            continue
        src = s.draft.participants
        memo: Dict[str, tuple] = {}
        pkey: Dict[str, str] = {}
        created = set()
        for k, p in src.items():
            ident = _identity(p, src, memo)
            if ident not in global_key:
                new = k if k not in d.participants else next(f'{k}_{n}' for n in range(2, 10 ** 6)
                                                              if f'{k}_{n}' not in d.participants)
                global_key[ident] = new
                d.participants[new] = p.model_copy(deep=True)
                d.participants[new].key = new
                created.add(new)
            pkey[k] = global_key[ident]
        for k, p in src.items():                  # a complex or set made here points at the combined keys of its parts
            g = d.participants[pkey[k]]
            if g.key in created and isinstance(g, ComplexSpec):
                g.components = [pkey.get(x, x) for x in p.components]
            elif g.key in created and isinstance(g, DefinedSetSpec):
                g.members = [pkey.get(x, x) for x in p.members]
        rkey = {r.key: f'{pre}_{r.key}' for r in s.draft.reactions}
        ekey = {e.id: f'{pre}_{e.id}' for e in s.evidence}
        for r in s.draft.reactions:
            n = r.model_copy(deep=True)
            n.key, n.inputs, n.outputs = rkey[r.key], [pkey.get(x, x) for x in r.inputs], [pkey.get(x, x) for x in r.outputs]
            if n.catalyst:
                n.catalyst.entity = pkey.get(n.catalyst.entity, n.catalyst.entity)
            for g in n.regulations:
                g.regulator = pkey.get(g.regulator, g.regulator)
            n.evidence_ids = [ekey.get(x, x) for x in r.evidence_ids]
            n.preceding = [rkey[x] for x in r.preceding if x in rkey]
            d.reactions.append(n)
        for e in s.evidence:
            evidence.append(e.model_copy(update={'id': ekey[e.id]}))
        for i in s.issues:
            if i.source in ('checks', 'emitter'):  # remade for the combined draft
                continue
            issues.append(i.model_copy(update={'id': '', 'reaction_key': rkey.get(i.reaction_key, i.reaction_key),
                                               'participant_key': pkey.get(i.participant_key, i.participant_key),
                                               'instance_db_id': None, 'blocked_by': None, 'proposal_id': None}))
        qa.update({rkey[k]: v for k, v in s.qa.items() if k in rkey})
        existing += [m.model_copy(update={'reaction_key': rkey[m.reaction_key]}) for m in s.existing if m.reaction_key in rkey]
        d.publications.update(s.draft.publications)
        papers += [p for p in s.papers if p.source not in {q.source for q in papers}]
        texts.update(s.paper_texts)
        usage += s.usage
        main = s.papers[0].source if s.papers else (s.source or pre)
        paper_reactions.append((part, [rkey[r.key] for r in s.draft.reactions], main))

    # the same reaction reported by several papers of a group: one reaction, listed under each of them
    alias: Dict[str, str] = {}
    if judge is not None:
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for part, keys, main in paper_reactions:
            groups.setdefault(part.group, []).extend((k, main) for k in keys)
        alias, stats = shared_reactions(d, groups, judge, map_fn)
        say(f'[combine] {stats["judged"]} pair(s) judged; {stats["merged"]} reaction(s) folded into {stats["groups"]} '
            f'shared reaction(s)')
    if alias:
        by_key = {r.key: r for r in d.reactions}
        for k, to in alias.items():
            _fold(by_key[to], by_key[k])
        d.reactions = [r for r in d.reactions if r.key not in alias]
        for r in d.reactions:
            r.preceding = [p for p in dict.fromkeys(alias.get(x, x) for x in r.preceding) if p != r.key]
        issues = [i for i in issues if i.reaction_key not in alias]
        existing = [m for m in existing if m.reaction_key not in alias]
        qa = {k: v for k, v in qa.items() if k not in alias}

    # the pathways: top -> group -> paper
    def pmids_of(keys):
        by_key = {r.key: r for r in d.reactions}
        return sorted({p for k in keys if k in by_key for p in by_key[k].pmids})
    group_pw: Dict[str, PathwaySpec] = {}
    top = PathwaySpec(name=title)
    for part, keys, _ in paper_reactions:
        mine = list(dict.fromkeys(alias.get(k, k) for k in keys))
        pw = PathwaySpec(name=part.title, reactions=mine, pmids=pmids_of(mine),
                         summation=(part.session.draft.pathway.summation if part.session.draft.pathway else ''))
        if part.group:
            if part.group not in group_pw:
                group_pw[part.group] = PathwaySpec(name=part.group)
                top.subpathways.append(group_pw[part.group])
            group_pw[part.group].subpathways.append(pw)
        else:
            top.subpathways.append(pw)
    for g in group_pw.values():
        g.pmids = sorted({p for sp in g.subpathways for p in sp.pmids})
    top.pmids = sorted({p for _, pw in top.walk() for p in pw.pmids})
    d.pathway = top

    s = Session(owner=owner, source=title, papers=papers, draft=d, evidence=evidence, issues=issues, existing=existing,
                usage=usage, usage_source='saved', paper_texts=texts, status='ready', progress='done', qa=qa,
                pmid=next((p.pmid for p in papers if p.pmid), None))
    s.change_log.append(ChangeLogEntry(actor='combine', action='combined',
                                       detail=f'{len(paper_reactions)} annotation(s), {len(d.reactions)} reactions'))
    s.created = time.time()
    return s
