"""Per-reaction QA of a draft: deterministic checks (always) plus an optional LLM review.

Deterministic checks are structural and evidence-based, the things code can know for certain: a reaction
with nothing consumed or produced, a catalyst or regulation with no quote behind it, a regulator that is
also a substrate. The LLM review judges biology (is the catalyst really a catalyst, does the summation
overstate the evidence) and sees the same facts plus the quotes.
"""
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.models.reactome import ReactionSpec, ReactomeDraft, SimpleEntitySpec


class QAFinding(BaseModel):
    severity: str                       # info | warning | action
    code: str
    message: str
    field: Optional[str] = None         # e.g. catalyst, regulations[1], summation
    source: str = 'rule'                # rule | llm


class QAResult(BaseModel):
    reaction_key: str
    verdict: str                        # ok | needs_work
    score: Optional[float] = None       # LLM's 0-1 confidence in the reaction, if it reviewed
    findings: List[QAFinding] = Field(default_factory=list)
    llm_used: bool = False


def _state(draft: ReactomeDraft, key: str):
    p = draft.participants[key]
    mods = tuple(sorted((m.psi_mod, m.coordinate) for m in getattr(p, 'modifications', [])))
    return (p.name.lower(), mods)


def rule_findings(draft: ReactomeDraft, r: ReactionSpec, evidence: Dict[str, Evidence]) -> List[QAFinding]:
    out: List[QAFinding] = []
    F = lambda sev, code, msg, field=None: out.append(QAFinding(severity=sev, code=code, message=msg, field=field))
    blackbox = r.reaction_type in ('blackBoxEvent', 'omitted')
    if not r.inputs and not blackbox:
        F('action', 'qa_no_inputs', 'the reaction consumes nothing', 'inputs')
    if not r.outputs and not blackbox:
        F('action', 'qa_no_outputs', 'the reaction produces nothing', 'outputs')
    if r.inputs and r.outputs and r.reaction_type not in ('binding', 'dissociation'):
        if sorted(_state(draft, k) for k in r.inputs) == sorted(_state(draft, k) for k in r.outputs):
            F('warning', 'qa_no_change', 'inputs and outputs are the same entities in the same state, so nothing '
              'happens in this reaction; check the modified or bound form of the output', 'outputs')
    if r.catalyst:
        cat = draft.participants[r.catalyst.entity]
        if isinstance(cat, SimpleEntitySpec):
            F('warning', 'qa_catalyst_not_protein', f'the catalyst "{cat.name}" is a small molecule; check it is not '
              f'a regulator or cofactor', 'catalyst')
        if r.catalyst.entity in r.outputs and r.catalyst.entity not in r.inputs:
            F('warning', 'qa_catalyst_is_output', f'"{cat.name}" is both catalyst and a produced entity', 'catalyst')
    ins = set(r.inputs)
    for i, g in enumerate(r.regulations):
        name = draft.participants[g.regulator].name
        if g.regulator in ins:
            F('warning', 'qa_regulator_is_input', f'"{name}" is both a regulator and a substrate', f'regulations[{i}]')
        same = [x for j, x in enumerate(r.regulations) if j != i and x.regulator == g.regulator]
        if any(x.kind != g.kind for x in same) and not g.note:
            F('warning', 'qa_conflicting_regulation', f'"{name}" regulates in more than one direction without a '
              f'condition note', f'regulations[{i}]')
    ev = [evidence[i] for i in r.evidence_ids if i in evidence]
    if not ev:
        F('action', 'qa_no_evidence', 'no quote from the paper supports this reaction', 'evidence')
    else:
        supported = {s for e in ev for s in e.supports}
        if r.catalyst and 'catalystActivity' not in supported:
            F('warning', 'qa_unsupported_catalyst', 'no quote is marked as supporting the catalyst', 'catalyst')
        for i, g in enumerate(r.regulations):
            if f'regulatedBy[{i}]' not in supported:
                F('warning', 'qa_unsupported_regulation',
                  f'no quote is marked as supporting the regulation by "{draft.participants[g.regulator].name}"',
                  f'regulations[{i}]')
        if all(e.claim_origin == ClaimOrigin.CURATOR_ASSERTION for e in ev):
            F('warning', 'qa_only_assertion', 'the only evidence is a curator assertion, not the paper', 'evidence')
        elif all(e.claim_origin == ClaimOrigin.CITED for e in ev):
            F('warning', 'qa_only_cited', 'every quote is a result this paper cites from earlier work, not shown here',
              'evidence')
        elif all(e.verified == Verification.FUZZY for e in ev):
            F('info', 'qa_fuzzy_evidence', 'every quote matched the paper only approximately; check the wording', 'evidence')
        species = {e.experimental_species for e in ev if e.experimental_species}
        if species and 'Homo sapiens' not in species:
            F('info', 'qa_non_human_evidence',
              f'the evidence is from {", ".join(sorted(species))}, not human; consider inferredFrom', 'evidence')
    if not r.summation.strip():
        F('info', 'qa_no_summation', 'no summation text', 'summation')
    if not r.pmids:
        F('warning', 'qa_no_literature_reference', 'no literature reference', 'pmids')
    return out


QA_SYSTEM = """You are a senior Reactome curator reviewing ONE reaction of a draft annotation. Judge only what the
reaction and its quotes support. Check: is the catalyst really the enzyme that performs the reaction (others are
regulators); are consumed and produced entities in the right modification state; does the summation claim more than
the quotes show; is a regulator's condition recorded; is ortholog or cited evidence presented as if shown here.
Reply with ONLY a JSON object: {"verdict": "ok" | "needs_work", "score": <0-1 confidence the reaction is correct and
well supported>, "findings": [{"severity": "info" | "warning" | "action", "message": "...", "field": "<field or null>"}]}.
Do not repeat the rule findings you are given. Say nothing about identifiers or formatting."""


def _llm_prompt(draft: ReactomeDraft, r: ReactionSpec, evidence: Dict[str, Evidence], rules: List[QAFinding]) -> str:
    used = set(r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else []))
    return json.dumps({
        'reaction': r.model_dump(mode='json', exclude={'evidence_ids'}),
        'participants': {k: draft.participants[k].model_dump(mode='json', exclude={'needs_resolution', 'existing'}) for k in used},
        'quotes': [{'quote': e.quote, 'supports': e.supports, 'system': e.system, 'species': e.experimental_species,
                    'origin': e.claim_origin.value, 'section': e.section}
                   for e in (evidence[i] for i in r.evidence_ids if i in evidence)],
        'rule_findings_already_reported': [f.message for f in rules]}, indent=1)


def parse_llm_review(text: str) -> Dict[str, Any]:
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        raise ValueError('no JSON in the review')
    d = json.loads(m.group(0))
    findings = []
    for f in d.get('findings') or []:
        sev = f.get('severity') if f.get('severity') in ('info', 'warning', 'action') else 'warning'
        if f.get('message'):
            findings.append(QAFinding(severity=sev, code='qa_llm', message=str(f['message'])[:500],
                                      field=f.get('field') or None, source='llm'))
    score = d.get('score')
    return {'verdict': d.get('verdict') if d.get('verdict') in ('ok', 'needs_work') else None,
            'score': float(score) if isinstance(score, (int, float)) and 0 <= score <= 1 else None,
            'findings': findings}


def qa_reaction(draft: ReactomeDraft, reaction_key: str, evidence: List[Evidence], model=None) -> QAResult:
    r = next((x for x in draft.reactions if x.key == reaction_key), None)
    if r is None:
        raise KeyError(reaction_key)
    ev = {e.id: e for e in evidence}
    findings = rule_findings(draft, r, ev)
    score, verdict_llm, used = None, None, False
    if model is not None:
        try:
            turn = model.turn(QA_SYSTEM, [{'role': 'user', 'content': _llm_prompt(draft, r, ev, findings)}], [], lambda t: None)
            rev = parse_llm_review(turn.text)
            findings += rev['findings']
            score, verdict_llm, used = rev['score'], rev['verdict'], True
        except Exception as e:                                   # the rule findings still stand
            findings.append(QAFinding(severity='info', code='qa_llm_unavailable',
                                      message=f'the LLM review could not be completed ({type(e).__name__})'))
    needs = any(f.severity == 'action' for f in findings) or verdict_llm == 'needs_work'
    return QAResult(reaction_key=reaction_key, verdict='needs_work' if needs else 'ok', score=score,
                    findings=findings, llm_used=used)
