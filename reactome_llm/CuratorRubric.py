"""Curator-rubric LLM scorer + final-selection judge, shared by the standalone evaluation
(`retrieval_eval.py`) and the live pipeline (`CrewAILiteratureAnnotator`).

Scores each paper for ANNOTATION USEFULNESS (can it support a Reactome-style entity/reaction/
complex?), not mere topical relevance, via specificity -> evidence type (DIRECT/INDIRECT/
INSUFFICIENT/NONE) -> annotatability. Factored out of retrieval_eval.py so the live pipeline can
reuse the exact validated rubric without importing the eval harness.

Public API:
  score_retrieval_relevance(gene, description, papers)  -> per-paper scores + mean + distribution
  judge_select(gene, description, candidates, max_papers, min_score)
                                                        -> final selection (score-threshold + top-N)
"""
import re
import json
from collections import Counter

from ModelConfig import create_reactome_chat_model

EVIDENCE_TYPES = ("DIRECT", "INDIRECT", "INSUFFICIENT", "NONE")

# Curator-style rubric: score ANNOTATION USEFULNESS (can this paper support a Reactome-style
# entity/reaction/complex?), not mere topical relevance to the gene name. Reuses the project's
# direct-vs-indirect evidence framing (ReactomeAgents/ReactomeTasks), extended into the explicit
# DIRECT / INDIRECT / INSUFFICIENT tiers with assay examples.
_SCORE_PROMPT = """You are an expert Reactome curator judging whether each retrieved paper is USEFUL
for building a Reactome-style annotation of the human gene {gene} -- NOT merely whether it is
topically about {gene}.

Gene biological context (reference only, do not score it):
{description}

For EACH paper, reason as a curator through three questions:
  1. SPECIFICITY: Does it make a SPECIFIC mechanistic claim -- a concrete molecular interaction,
     reaction, complex, or regulatory event -- rather than general/topical background? The claim
     counts as specific if it is EITHER:
       (a) gene-named -- it explicitly involves {gene}; OR
       (b) pathway/partner-context-specific -- it describes a specific mechanism within {gene}'s
           immediate pathway, or among {gene}'s interaction partners, EVEN IF {gene} itself is never
           named in the title/abstract.
     CRITICAL: a paper that never mentions {gene} can still be fully specific and highly useful.
     Curators routinely cite mechanism papers about a gene's pathway or partners that do not name the
     gene at all (e.g. a paper on a BRAF mutation in the RAF/MAP-kinase cascade cited as evidence for
     a gene placed in that cascade). These gene-absent mechanism papers are the MAJORITY of real
     curator-cited evidence. Do NOT downgrade a paper because {gene} is absent -- judge the
     specificity of its mechanistic claim within {gene}'s pathway/partner context, using the gene
     context above to decide whether the paper's mechanism belongs to that context.
  2. EVIDENCE TYPE: What is the strongest experimental evidence it presents for that claim?
       DIRECT       -- co-immunoprecipitation, in vitro binding, crystallography, SPR/ITC
       INDIRECT     -- knockout, overexpression, inhibitor treatment
       INSUFFICIENT -- microarray, bulk proteomics, or computational prediction alone
       NONE         -- presents no experimental evidence for a specific mechanistic claim
  3. ANNOTATABILITY: Given that evidence, is it strong enough to plausibly support a Reactome
     annotation (a specific entity / reaction / complex) in {gene}'s pathway context, or only
     background context?

Then assign an OVERALL USEFULNESS score from 1 to 10. Score on SPECIFICITY + EVIDENCE TYPE ONLY --
gene-name presence must NOT change the score: a pathway/partner-context-specific claim scores exactly
the same as a gene-named one at the same evidence level.
  9-10 = specific mechanistic claim (gene-named OR pathway/partner-context-specific) + DIRECT evidence
         -> directly supports an annotation
  6-8  = specific mechanistic claim (either specificity type) + INDIRECT evidence, OR DIRECT evidence
         with slightly less certain specificity -> plausibly supports one, needs validation
  3-5  = genuinely vague / general topical relevance, OR INSUFFICIENT evidence type -> weak background
  1-2  = unrelated -> not usable

Papers:
{papers_block}

Return ONLY a JSON array, exactly one object per paper, no prose before or after:
[{{"index": <int matching the [n] label>, "score": <int 1-10>,
   "evidence_type": "DIRECT|INDIRECT|INSUFFICIENT|NONE",
   "specific_interaction": <true|false>, "annotatable": <true|false>,
   "justification": "<one sentence tying evidence type + specificity to the score>"}}]
"""


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return None


def _parse_scores(text: str) -> dict:
    """Robustly pull the curator-rubric array out of an LLM reply -> {index: {...}}.
    Tolerates code fences and surrounding prose; returns {} if no array is parseable."""
    if not text:
        return {}
    t = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    m = re.search(r"\[.*\]", t, flags=re.S)
    arr = None
    if m:
        try:
            arr = json.loads(m.group(0))
        except Exception:
            arr = None
    if not isinstance(arr, list):
        # Truncated/partial reply (e.g. the output-token cap clipped the closing `]`): salvage every
        # complete top-level {...} object individually so a cut-off tail costs only its own papers.
        arr = []
        for obj in re.findall(r"\{[^{}]*\}", t, flags=re.S):
            try:
                arr.append(json.loads(obj))
            except Exception:
                continue
    out = {}
    for e in arr if isinstance(arr, list) else []:
        if not isinstance(e, dict):
            continue
        try:
            idx = int(e.get("index"))
        except (TypeError, ValueError):
            continue
        try:
            sc = min(10, max(1, int(e.get("score"))))
        except (TypeError, ValueError):
            sc = None
        et = str(e.get("evidence_type", "")).strip().upper()
        et = et if et in EVIDENCE_TYPES else None
        out[idx] = {
            "score": sc,
            "evidence_type": et,
            "specific_interaction": _coerce_bool(e.get("specific_interaction")),
            "annotatable": _coerce_bool(e.get("annotatable")),
            "justification": str(e.get("justification", "")).strip(),
        }
    return out


def score_retrieval_relevance(gene: str, description: str, papers: list,
                              max_abstract_chars: int = 1200) -> dict:
    """ONE batched LLM call scoring each paper's ANNOTATION USEFULNESS (1-10) for `gene`, judged
    as an expert curator via specificity -> evidence type (DIRECT/INDIRECT/INSUFFICIENT/NONE) ->
    annotatability. `description` gives the gene's biological context for reference.

    Returns per-paper {score, evidence_type, specific_interaction, annotatable, justification},
    plus mean_score, score distribution, and evidence_type_counts. Standalone: `papers` is any
    list of dicts with pmid / title / abstract."""
    empty = {"gene": gene, "n": 0, "mean_score": None, "distribution": {},
             "evidence_type_counts": {}, "annotatable_count": 0, "per_paper": []}
    if not papers:
        return empty

    blocks = []
    for i, p in enumerate(papers, 1):
        title = (p.get("title") or "").strip() or "(no title available)"
        abstract = " ".join((p.get("abstract") or "").split())[:max_abstract_chars] or "(no abstract)"
        blocks.append(f"[{i}] PMID {p.get('pmid', '?')} | {title}\n{abstract}")

    prompt = _SCORE_PROMPT.format(gene=gene, description=(description or "(none provided)"),
                                  papers_block="\n\n".join(blocks))
    try:
        # One object per paper (score + evidence type + one-sentence justification) easily exceeds
        # ChatAnthropic's default 1024-token output cap for a full candidate pool, truncating the JSON
        # array -> nothing parses. Raise the cap so the whole array comes back intact.
        model = create_reactome_chat_model().bind(max_tokens=8000)
        content = model.invoke(prompt).content
    except Exception as e:
        print(f"  [warn] scorer LLM call failed for {gene}: {e}")
        content = ""
    parsed = _parse_scores(content)

    per_paper = []
    for i, p in enumerate(papers, 1):
        e = parsed.get(i, {})
        per_paper.append({
            "pmid": p.get("pmid", ""),
            "title": p.get("title", ""),
            "score": e.get("score"),
            "evidence_type": e.get("evidence_type"),
            "specific_interaction": e.get("specific_interaction"),
            "annotatable": e.get("annotatable"),
            "justification": e.get("justification") or "(unscored — parse/scoring gap)",
        })

    valid = [pp["score"] for pp in per_paper if isinstance(pp["score"], (int, float))]
    mean = round(sum(valid) / len(valid), 2) if valid else None
    distribution = {s: c for s, c in sorted(Counter(valid).items())}
    evidence_type_counts = {t: c for t, c in
                            sorted(Counter(pp["evidence_type"] for pp in per_paper
                                           if pp["evidence_type"]).items())}
    annotatable_count = sum(1 for pp in per_paper if pp["annotatable"])
    return {"gene": gene, "n": len(papers), "mean_score": mean, "distribution": distribution,
            "evidence_type_counts": evidence_type_counts, "annotatable_count": annotatable_count,
            "per_paper": per_paper}


def judge_select(gene: str, description: str, candidates: list, max_papers: int = 5,
                 min_score: int = 3) -> dict:
    """Final selection stage: an LLM curator-judge reads the cross-encoder candidate pool and picks
    the papers to keep, using the SAME validated rubric as score_retrieval_relevance (specificity ->
    evidence type -> annotatability) rather than a new criterion.

    Policy (score-threshold + top-N): score every candidate, DROP any below `min_score` (default 3 =
    below "weak, mostly background"; 1-2 are "not usable for a specific annotation"), then return up
    to `max_papers` of the survivors, highest score first, ties broken by candidate (cross-encoder)
    order. May return FEWER than max_papers -- an honest signal that a gene has few usable papers.

    `candidates`: list of dicts with pmid / title / abstract (cross-encoder-ranked order).
    Returns {"selected": [candidate dicts kept], "scored": <full score_retrieval_relevance output>,
             "dropped_below_threshold": <int>}."""
    if not candidates:
        return {"selected": [], "scored": score_retrieval_relevance(gene, description, []),
                "dropped_below_threshold": 0}

    scored = score_retrieval_relevance(gene, description, candidates)
    per = scored["per_paper"]  # aligned 1:1 with `candidates`

    ranked = []
    for orig_i, (cand, sc) in enumerate(zip(candidates, per)):
        s = sc["score"]
        s_val = s if isinstance(s, (int, float)) else -1
        ranked.append((s_val, orig_i, cand))
    # highest score first; ties -> lower cross-encoder rank (smaller orig_i) first
    ranked.sort(key=lambda t: (-t[0], t[1]))

    selected, dropped = [], 0
    for s_val, _, cand in ranked:
        if s_val < min_score:
            dropped += 1
            continue
        if len(selected) < max_papers:
            selected.append(cand)
    return {"selected": selected, "scored": scored, "dropped_below_threshold": dropped}
