"""Per-reaction QA of a draft: deterministic checks (always) plus an optional LLM review.

Deterministic checks are structural and evidence-based, the things code can know for certain: a reaction
with nothing consumed or produced, a catalyst or regulation with no quote behind it, a regulator that is
also a substrate. The LLM review judges biology (is the catalyst really a catalyst, does the summation
overstate the evidence) and sees the same facts plus the quotes.
"""
import json
import re
from typing import Any, Collection, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.models.reactome import ReactionSpec, ReactomeDraft, SimpleEntitySpec
from curator_llm.services.dedup_prefilter import side_tokens, tokens


class QAFinding(BaseModel):
    severity: str                       # info | warning | action
    code: str
    message: str
    field: Optional[str] = None         # e.g. catalyst, regulations[1], summation
    source: str = 'rule'                # rule | llm


class QAResult(BaseModel):
    reaction_key: str
    reaction_name: Optional[str] = None  # for lists: the reaction's name
    verdict: str                        # ok | needs_work (kept for older callers; see `priority`)
    priority: str = 'ok'                # fix_first (3+ action findings) | fix | review (warnings only) | ok
    summary: Optional[str] = None       # one sentence: the most important thing to do, from the reviewer
    score: Optional[float] = None       # LLM's 0-1 confidence in the reaction, if it reviewed
    findings: List[QAFinding] = Field(default_factory=list)
    llm_used: bool = False
    usage: Optional[Dict[str, int]] = None   # tokens the model review spent, when it ran
    is_reaction: Optional[bool] = None       # the model's answer to "is this a molecular reaction at all" (None: not asked)
    not_reaction_reason: Optional[str] = None


def _state(draft: ReactomeDraft, key: str):
    p = draft.participants[key]
    mods = tuple(sorted((m.psi_mod, m.coordinate) for m in getattr(p, 'modifications', [])))
    return (p.name.lower(), mods)


# What a quote's wording says about it. Patterns, so they can be wrong: they raise warnings for a curator to look at and
# only fire when EVERY quote of a reaction matches.
_REF_MARKER = re.compile(r'\(\s*(?:see\s+)?refs?\.\s*\d|\bet al\.', re.I)            # "(ref. 43)", "(refs. 1-3)", "Smith et al."
_HEDGE = re.compile(r'\b(?:may|might|could|possibly|potentially|perhaps|suggests?|suggesting|hypothes\w*|propose[sd]?|'
                    r'predicts?|predicted|we considered|it is (?:possible|likely))\b', re.I)
_NEGATIVE = re.compile(r'\b(?:negligible|did not|does not|do not|was not|were not|no (?:significant |detectable )?'
                       r'(?:change|effect|difference|increase|decrease)|not (?:detected|observed|affected|altered|changed)|'
                       r'unchanged|unaffected|failed to|fails to)\b', re.I)
# the first words of a figure legend that describes a panel rather than stating a result
_LEGEND = re.compile(r'^\s*(?:\(?[a-z]\)?[.,]?\s+)?(?:western blots?|immunoblots?|graphs?|images?|quantification|schematic|'
                     r'cartoon|representative|plots?|heat ?map|dose[\u2013-]response|time[- ]course|time-lapse|growth curves?|'
                     r'validation of)\b', re.I)
_METHODS_SECTIONS = ('method', 'materials', 'experimental procedure', 'supplementary')


def is_cited_quote(e: Evidence) -> bool:
    """States earlier work: tagged cited, or tagged as this paper's but carrying a reference such as "(ref. 43)"."""
    return e.claim_origin == ClaimOrigin.CITED or (e.claim_origin == ClaimOrigin.THIS_PAPER
                                                   and bool(_REF_MARKER.search(e.quote or '')))


def describes_method_or_figure(e: Evidence) -> bool:
    """From a Methods section, or a figure legend describing a panel ("Western blot showing ...")."""
    section = (e.section or '').lower()
    return any(section.startswith(m) for m in _METHODS_SECTIONS) or bool(_LEGEND.match(e.quote or ''))


def rule_findings(draft: ReactomeDraft, r: ReactionSpec, evidence: Dict[str, Evidence],
                  reviews: Collection[str] = ()) -> List[QAFinding]:
    """The deterministic checks of one reaction. `reviews`: the source tags of the papers that are review articles, whose
    cited statements are their evidence (a research paper's are not)."""
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
        if r.catalyst.entity in r.inputs and r.catalyst.entity not in r.outputs:
            F('warning', 'qa_catalyst_is_consumed', f'"{cat.name}" is the catalyst but is listed as an input and not '
              f'as an output, so the reaction consumes its own enzyme; an enzyme is not consumed', 'inputs')
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
        research = [e for e in ev if e.source not in reviews and e.claim_origin != ClaimOrigin.CURATOR_ASSERTION]
        if all(e.claim_origin == ClaimOrigin.CURATOR_ASSERTION for e in ev):
            F('warning', 'qa_only_assertion', 'the only evidence is a curator assertion, not the paper', 'evidence')
        elif research and len(research) == len(ev) and all(is_cited_quote(e) for e in ev):
            marked = [e for e in ev if e.claim_origin == ClaimOrigin.THIS_PAPER]
            F('warning', 'qa_only_cited', 'every quote states earlier work' +
              (' (cited, or carrying a reference such as "(ref. N)")' if marked else '') +
              ', not a result shown in this paper; Reactome needs direct evidence: remove the reaction unless the paper '
              'also shows it', 'evidence')
        elif all(e.verified == Verification.FUZZY for e in ev):
            F('info', 'qa_fuzzy_evidence', 'every quote matched the paper only approximately; check the wording', 'evidence')
        quotes = [e for e in ev if e.claim_origin != ClaimOrigin.CURATOR_ASSERTION]
        results = [e for e in quotes if not describes_method_or_figure(e)]     # the quotes that state something
        if results and all(_NEGATIVE.search(e.quote or '') for e in results):
            F('warning', 'qa_negative_only', 'every quote stating a result reports a negative or negligible one ("did not", '
              '"negligible", "no change"), which is not evidence that this reaction happens: remove it, or find the '
              'sentence that shows it', 'evidence')
        elif results and all(_HEDGE.search(e.quote or '') for e in results):
            F('warning', 'qa_hedged_only', 'every quote stating a result is hedged ("may", "could", "suggests", a hypothesis or a '
              'model prediction): it is not shown as a result; find a sentence that states it, or remove it', 'evidence')
        if quotes and all(describes_method_or_figure(e) for e in quotes):
            F('warning', 'qa_methods_only', 'every quote comes from Methods or describes a figure panel ("Western blot '
              'showing ..."), none states a result: check that the paper reports this as a finding', 'evidence')
        species = {e.experimental_species for e in ev if e.experimental_species}
        if species and 'Homo sapiens' not in species:
            F('info', 'qa_non_human_evidence',
              f'the evidence is from {", ".join(sorted(species))}, not human; consider inferredFrom', 'evidence')
    if not r.summation.strip():
        F('info', 'qa_no_summation', 'no summation text', 'summation')
    if not r.pmids:
        F('warning', 'qa_no_literature_reference', 'no literature reference', 'pmids')
    return out


# The rules that need no model: the draft's structure, what the quotes say about themselves, and reactions that look
# like each other. They run on every refresh and appear in the issue list without the curator asking for a QA run.
AUTO_RULE_CODES = frozenset({'qa_no_inputs', 'qa_no_outputs', 'qa_no_change', 'qa_catalyst_not_protein',
                             'qa_catalyst_is_output', 'qa_catalyst_is_consumed', 'qa_regulator_is_input',
                             'qa_conflicting_regulation', 'qa_only_cited', 'qa_negative_only', 'qa_hedged_only',
                             'qa_methods_only', 'qa_possible_duplicate'})


def _jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _overlap(a, b) -> float:
    """Shared share of the smaller set: "CDK1" vs "CDK1, cyclin B" is 1 (less detail, as the merge judge treats it)."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def duplicate_findings(draft: ReactomeDraft) -> List[Tuple[str, QAFinding]]:
    """Reactions the merge left apart that look like the same event: the same catalyst (or none for both), and inputs
    and outputs that name mostly the same entities, one side possibly with less detail (names compared by tokens,
    ignoring modification words and cofactors, see dedup_prefilter). With no catalyst there is less to go on, so both
    must produce something and their regulators must agree. Reported on the later reaction of each pair, as a warning:
    the merge's judge had called them different, so a curator decides."""
    name = lambda k: draft.participants[k].name if k in draft.participants else k
    sides = []
    for r in draft.reactions:
        cat = tokens(name(r.catalyst.entity)) if r.catalyst else None
        sides.append((r, cat, side_tokens(name(k) for k in r.inputs), side_tokens(name(k) for k in r.outputs),
                      side_tokens(name(g.regulator) for g in r.regulations)))
    out = []
    for j, (b, cb, ib, ob, gb) in enumerate(sides):
        for a, ca, ia, oa, ga in sides[:j]:
            if (ca is None) != (cb is None) or (ca is not None and _jaccard(ca, cb) < 0.5):
                continue
            if ca is None and (not (oa and ob) or (ga and gb and _overlap(ga, gb) < 0.5)):
                continue
            if _overlap(ia, ib) >= 0.67 and _overlap(oa, ob) >= 0.67:
                out.append((b.key, QAFinding(severity='warning', code='qa_possible_duplicate', field=None,
                            message=f'may describe the same event as {a.key} "{a.name}" (same catalyst, mostly the same '
                                    f'inputs and outputs): merge them if they are one reaction')))
                break
    return out


def auto_findings(draft: ReactomeDraft, evidence: Optional[List[Evidence]] = None,
                  reviews: Collection[str] = ()) -> List[Tuple[str, QAFinding]]:
    """(reaction key, finding) for every automatic rule that fires, in reaction order."""
    ev = {e.id: e for e in evidence or []}
    found = [(r.key, f) for r in draft.reactions for f in rule_findings(draft, r, ev, reviews) if f.code in AUTO_RULE_CODES]
    dups = dict(duplicate_findings(draft))
    order = {r.key: i for i, r in enumerate(draft.reactions)}
    return sorted(found + list(dups.items()), key=lambda kf: order.get(kf[0], 0))


QA_SYSTEM = """You are a senior Reactome curator reviewing ONE reaction of a draft annotation, to tell a human curator
what to fix. Judge only what the reaction and its quotes support. Check: is the catalyst really the enzyme that performs
the reaction (others are regulators); are consumed and produced entities in the right modification state; does the
summation claim more than the quotes show; is a regulator's condition recorded; is ortholog or cited evidence presented
as if shown here.

First of all decide whether it is a molecular reaction at all, and answer in "is_reaction". It is NOT one (false) when it
is a cell-level outcome (arrest, a longer phase, death, drug sensitivity), a structural contact inside a complex (a
hydrogen bond), a hypothesis or model prediction, the reason for an experimental step ("X was added to prevent Y"), an
event of a tool (a degron, a reporter), or a negative result; then give the reason in "not_reaction_reason" (one
sentence) and list NO findings, since the curator will first decide whether to keep it. A real reaction with problems
(a wrong catalyst, a missing quote, a regulator to remove) IS a reaction: true, with findings.
Reply with ONLY a JSON object: {"is_reaction": true | false, "not_reaction_reason": "<one sentence, or null>",
"verdict": "ok" | "needs_work", "score": <0-1>, "summary": "<one sentence>",
"findings": [{"severity": "info" | "warning" | "action", "message": "...", "field": "<field or null>"}]}.

Report PROBLEMS only. Never say what is correct, reasonable or acceptable, and do not restate the reaction: if nothing is
wrong, return no findings. Each message says what is wrong and what to change, in one or two sentences. At most 5
findings, the most important first. Do not repeat the rule findings you are given. Say nothing about identifiers or
formatting.

Severity. "action": an error that makes the reaction wrong or unsupported as written (the wrong catalyst, a wrong consumed
or produced entity or state, a claim no quote supports, a regulator in the wrong direction). "warning": a real weakness
that does not make the reaction wrong. "info": a minor note or polish.
Verdict: "needs_work" only when at least one finding is an "action"; otherwise "ok", however many warnings or notes.
Score: 0.9 or more when sound with at most notes; 0.7-0.9 sound with warnings; 0.4-0.7 an easily fixed action; below 0.4
mostly unsupported or wrong.
Summary: the single most important thing for the curator to do about this reaction, or "No problems found"."""

MAX_LLM_FINDINGS = 5
_SEVERITY_RANK = {'action': 0, 'warning': 1, 'info': 2}
_PRIORITY_RANK = {'fix_first': 0, 'fix': 1, 'review': 2, 'ok': 3}
FIX_FIRST_ACTIONS = 3        # this many errors in one reaction and it goes to the top of the list


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
    summary = d.get('summary')
    is_reaction = d.get('is_reaction') if isinstance(d.get('is_reaction'), bool) else None
    why = d.get('not_reaction_reason')
    return {'is_reaction': is_reaction,
            'not_reaction_reason': str(why).strip()[:400] if is_reaction is False and isinstance(why, str) and why.strip() else None,
            'verdict': d.get('verdict') if d.get('verdict') in ('ok', 'needs_work') else None,
            'score': float(score) if isinstance(score, (int, float)) and 0 <= score <= 1 else None,
            'summary': str(summary).strip()[:300] if isinstance(summary, str) and summary.strip() else None,
            'findings': findings}


def priority_of(findings: List[QAFinding], llm_verdict: Optional[str]) -> str:
    """Where this reaction goes in the curator's list. A binary verdict was "needs_work" for every reaction on real
    drafts (an automatic first draft nearly always has something wrong), so rank by how much is wrong: fix_first (several
    errors), fix (one or two), review (warnings only, or the reviewer said needs_work without naming an action), ok."""
    actions = sum(1 for f in findings if f.severity == 'action')
    if actions >= FIX_FIRST_ACTIONS:
        return 'fix_first'
    if actions:
        return 'fix'
    if llm_verdict == 'needs_work' or any(f.severity == 'warning' for f in findings):
        return 'review'
    return 'ok'


def rank_results(results: List['QAResult']) -> List['QAResult']:
    """Most urgent first: by priority, then more action findings, then more warnings, then lower score."""
    def key(r: 'QAResult'):
        n = lambda sev: sum(1 for f in r.findings if f.severity == sev)
        return (_PRIORITY_RANK.get(r.priority, 3), -n('action'), -n('warning'), r.score if r.score is not None else 1.0)
    return sorted(results, key=key)


def qa_reaction(draft: ReactomeDraft, reaction_key: str, evidence: List[Evidence], model=None,
                reviews: Collection[str] = ()) -> QAResult:
    r = next((x for x in draft.reactions if x.key == reaction_key), None)
    if r is None:
        raise KeyError(reaction_key)
    ev = {e.id: e for e in evidence}
    findings = rule_findings(draft, r, ev, reviews)
    score, verdict_llm, summary, used, spent, is_reaction, why = None, None, None, False, None, None, None
    if model is not None:
        try:
            turn = model.turn(QA_SYSTEM, [{'role': 'user', 'content': _llm_prompt(draft, r, ev, findings)}], [], lambda t: None)
            u = getattr(turn, 'usage', None)           # recorded before parsing: a reply that cannot be parsed still cost tokens
            if u is not None:
                spent = {'input_tokens': u.input_tokens, 'output_tokens': u.output_tokens,
                         'cache_read_tokens': u.cache_read_tokens, 'cache_write_tokens': u.cache_write_tokens}
            rev = parse_llm_review(turn.text)
            findings += sorted(rev['findings'], key=lambda f: _SEVERITY_RANK[f.severity])[:MAX_LLM_FINDINGS]
            score, verdict_llm, summary, used = rev['score'], rev['verdict'], rev['summary'], True
            is_reaction, why = rev['is_reaction'], rev['not_reaction_reason']
            if is_reaction is False:              # nothing else matters until the curator decides to keep it
                findings = [f for f in findings if f.source != 'llm']
                summary = summary or why
        except Exception as e:                                   # the rule findings still stand
            findings.append(QAFinding(severity='info', code='qa_llm_unavailable',
                                      message=f'the LLM review could not be completed ({type(e).__name__})'))
    needs = any(f.severity == 'action' for f in findings) or verdict_llm == 'needs_work' or is_reaction is False
    findings.sort(key=lambda f: _SEVERITY_RANK[f.severity])             # stable: most serious first, rules before the LLM
    return QAResult(reaction_key=reaction_key, reaction_name=r.name, verdict='needs_work' if needs else 'ok', score=score, summary=summary,
                    priority='fix_first' if is_reaction is False else priority_of(findings, verdict_llm),
                    findings=findings, llm_used=used, usage=spent, is_reaction=is_reaction, not_reaction_reason=why)
