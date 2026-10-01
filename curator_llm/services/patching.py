"""Validated edits to a ReactomeDraft.

Edits are RFC 6902 JSON Patch against the draft's JSON, with two conveniences for the model and the
UI: a path segment under /reactions can be the reaction KEY ("/reactions/r2/summation") instead of an
array index, and evidence cannot be edited through a patch at all (it only enters through a proposal's
`evidence`, where every quote is verified against the paper first).

A patch is applied to a COPY and the result must (1) still be a valid ReactomeDraft and (2) keep every
reference intact, so a bad edit is refused with the reason, never half-applied.
"""
import copy
from typing import List

import jsonpatch

from curator_llm.models.reactome import ComplexSpec, DefinedSetSpec, ReactomeDraft

FORBIDDEN_SEGMENTS = {'evidence_ids'}


class PatchError(ValueError):
    pass


def _resolve_paths(doc: dict, ops: List[dict]) -> List[dict]:
    keys = [r['key'] for r in doc.get('reactions', [])]
    out = []
    for op in ops:
        op = dict(op)
        for field in ('path', 'from'):
            if field in op:
                parts = op[field].split('/')
                if len(parts) > 2 and parts[1] == 'reactions' and parts[2] in keys:
                    parts[2] = str(keys.index(parts[2]))
                op[field] = '/'.join(parts)
        out.append(op)
    return out


def integrity_errors(d: ReactomeDraft) -> List[str]:
    errs, rkeys, pkeys = [], [r.key for r in d.reactions], set(d.participants)
    for k, p in d.participants.items():
        if p.key != k:
            errs.append(f'participant under "{k}" has key "{p.key}"')
        refs = p.components if isinstance(p, ComplexSpec) else p.members if isinstance(p, DefinedSetSpec) else []
        errs += [f'{p.name}: unknown component/member "{x}"' for x in refs if x not in pkeys]
    if len(set(rkeys)) != len(rkeys):
        errs.append('duplicate reaction keys')
    for r in d.reactions:
        used = r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else [])
        errs += [f'reaction {r.key}: unknown participant "{x}"' for x in used if x not in pkeys]
        errs += [f'reaction {r.key}: unknown preceding reaction "{x}"' for x in r.preceding if x not in rkeys]
        if r.key in r.preceding:
            errs.append(f'reaction {r.key} precedes itself')
    if d.pathway:
        errs += [f'pathway: unknown reaction "{x}"' for x in d.pathway.reactions if x not in rkeys]
    return errs


def apply_patch(draft: ReactomeDraft, ops: List[dict]) -> ReactomeDraft:
    doc = draft.model_dump(mode='json')
    ops = _resolve_paths(doc, ops)
    for op in ops:
        for field in ('path', 'from'):
            if field in op and FORBIDDEN_SEGMENTS & set(op[field].split('/')):
                raise PatchError('evidence cannot be edited through a patch; attach it to the proposal')
    try:
        patched = jsonpatch.apply_patch(copy.deepcopy(doc), ops)
    except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException, KeyError, IndexError, TypeError) as e:
        raise PatchError(f'patch does not apply: {e}') from e
    try:
        new = ReactomeDraft.model_validate(patched)
    except Exception as e:
        raise PatchError(f'result is not a valid draft: {str(e)[:300]}') from e
    errs = integrity_errors(new)
    if errs:
        raise PatchError('; '.join(errs[:5]))
    return new


def touched_reactions(before: ReactomeDraft, after: ReactomeDraft) -> List[str]:
    """Keys of reactions that were added, removed or changed (participant edits count for the
    reactions that use them)."""
    b = {r.key: r.model_dump(mode='json') for r in before.reactions}
    a = {r.key: r.model_dump(mode='json') for r in after.reactions}
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    pchanged = {k for k in set(before.participants) | set(after.participants)
                if before.participants.get(k) != after.participants.get(k)}
    for r in after.reactions:
        used = set(r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else []))
        if used & pchanged:
            changed.add(r.key)
    return sorted(changed)


def describe_changes(before: ReactomeDraft, after: ReactomeDraft) -> List[str]:
    """Human-readable diff for the accept/reject card."""
    lines: List[str] = []
    bp, ap = before.participants, after.participants
    for k in ap:
        if k not in bp:
            lines.append(f'add {ap[k].kind} "{ap[k].name}" ({k})')
    for k in bp:
        if k not in ap:
            lines.append(f'remove {bp[k].kind} "{bp[k].name}" ({k})')
    for k in ap:
        if k in bp and ap[k] != bp[k]:
            x, y = bp[k].model_dump(mode='json'), ap[k].model_dump(mode='json')
            for f in y:
                if x.get(f) != y[f] and f not in ('needs_resolution',):
                    lines.append(f'{ap[k].name} ({k}).{f}: {x.get(f)!r} -> {y[f]!r}')
    br = {r.key: r for r in before.reactions}
    ar = {r.key: r for r in after.reactions}
    for k, r in ar.items():
        if k not in br:
            lines.append(f'add reaction {k}: "{r.name}"')
    for k, r in br.items():
        if k not in ar:
            lines.append(f'remove reaction {k}: "{r.name}"')
    for k, r in ar.items():
        if k in br and r != br[k]:
            x, y = br[k].model_dump(mode='json'), r.model_dump(mode='json')
            for f in y:
                if x.get(f) != y[f] and f != 'evidence_ids':
                    lines.append(f'reaction {k}.{f}: {x.get(f)!r} -> {y[f]!r}')
    return lines
