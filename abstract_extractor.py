"""Abstract-level reaction extraction for MISS papers (no full text available).

When FullTextResolver marks a retrieved PMID as ``miss`` (no local PDF and no PMC full
text), we still try to pull reactions from its ABSTRACT rather than dropping the paper.
The output uses the SAME reaction schema as the full-text extractor
(run_extraction.py / FullTextPDFPrompts.build_extraction_prompt) so abstract reactions drop
straight into the same downstream pipeline -- but:

  * the prompt is ABSTRACT-AWARE: an abstract summarizes results with little mechanistic
    detail, so the model is told to extract ONLY explicitly-stated molecular events and NOT
    to invent intermediate species, sites, or mechanisms;
  * every emitted reaction is LABELED ``provenance: "abstract"`` (and its wrapper ``source``
    carries a ``(abstract)`` tag) so a curator can always tell abstract-derived reactions
    from full-text ones, and can weight them accordingly.

This module is deliberately SEPARATE from the partner's run_extraction.py (which stays
untouched, so re-pulling her branch remains a clean merge). It makes one bounded Anthropic
call per abstract -- no agent, no tool loop.
"""
import os
import re
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REACTOME_LLM = os.path.join(PROJECT_ROOT, "reactome_llm")
if _REACTOME_LLM not in sys.path:
    sys.path.insert(0, _REACTOME_LLM)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

import anthropic
from PubMedFetcher import fetch_abstract

logger = logging.getLogger(__name__)

# Match run_extraction.py's model so abstract and full-text extractions are judged by the same
# model. (The partner's scripts hardcode this; the user confirmed it's valid.)
MODEL_NAME = "claude-sonnet-5"
_MAX_TOKENS = 16000

# Cross-model OpenAI reviewer, mirroring run_review.py so an abstract-derived reaction set gets a
# score comparable to a full-text one. run_review.py itself can't be reused here: it re-derives the
# source chunks by calling load_source on the PMID, which fails for a MISS paper (no full text).
MODEL_REVIEW = "gpt-5.6-luna"

_client = None
_oai = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=120.0)
    return _client


def _get_oai():
    global _oai
    if _oai is None:
        from openai import OpenAI
        _oai = OpenAI(timeout=300.0, max_retries=2)
    return _oai


# ---- token accounting (folded into fulltext_extractor._FT_USAGE by the caller) --------------
USAGE = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def reset_usage():
    for k in USAGE:
        USAGE[k] = 0


def get_usage():
    return dict(USAGE)


def _track_usage(msg):
    u = getattr(msg, "usage", None)
    if not u:
        return
    USAGE["calls"] += 1
    USAGE["input"] += getattr(u, "input_tokens", 0) or 0
    USAGE["output"] += getattr(u, "output_tokens", 0) or 0
    USAGE["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
    USAGE["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0


def build_abstract_extraction_prompt(abstract_text, gene=None):
    """Abstract-aware extraction prompt. Emits the SAME reaction JSON schema as the full-text
    extractor, but instructs the model to stay conservative because it is reading an abstract,
    not a Results section."""
    gene_line = (f"The paper concerns the gene/protein {gene}. " if gene else "")
    return f"""You are an expert Reactome biocurator. Extract the biochemical reactions
EXPLICITLY described in the PubMed ABSTRACT below.

IMPORTANT — THIS IS AN ABSTRACT, NOT A FULL PAPER. {gene_line}An abstract summarizes findings
with little mechanistic detail, so you must be CONSERVATIVE:
  - Extract ONLY molecular events the abstract states outright (an enzyme modifies a substrate;
    molecules bind; a complex forms or dissociates; a molecule is transported, synthesized, or
    degraded).
  - Do NOT infer intermediate species, modification sites, catalysts, compartments, or
    mechanistic steps that the abstract does not explicitly state. Leave such fields null.
  - Prefer FEWER, well-supported reactions over speculative ones. If the abstract only states a
    general or hedged claim ("may", "might", "could", "suggests", "implicated in"), do NOT emit
    a reaction for it.
  - Set each reaction's "confidence" modestly (typically <= 0.6): abstract-level support is
    weaker than full-text support.

A biochemical reaction requires at least two named biological entities and a directional
molecular relationship between them (phosphorylates, binds, cleaves, activates, inhibits,
ubiquitinates, translocates, recruits, etc.). STRIP experimental framing (assay, tag, method)
from the reaction itself; the raw wording lives in the "evidence" excerpts.

CATALYST vs REGULATOR — assign an entity as "catalyst" ONLY when the abstract explicitly says it
directly performs the reaction. An entity that merely facilitates, promotes, or is required for
a reaction is a REGULATOR (put it in "regulatedBy", leave "catalyst" null). Many events
(binding, transport, dissociation) have no catalyst — null is correct.

DO NOT extract negative/failed results, pure measurements, or comparative statements.

The "evidence" field is your CITATION — a LIST of the VERBATIM sentence(s) from the abstract you
drew the reaction from, copied exactly. Every non-null subsection you fill (regulatedBy,
catalyst, compartment, condition, input/output) must be supported by an excerpt here.

Return ONLY a JSON object, no markdown:
{{
  "reactions": [
    {{
      "name": "<display name, e.g. 'PINK1 phosphorylates Parkin'>",
      "reactionType": "<transition | omitted | binding | dissociation | blackBoxEvent>",
      "input": ["<PhysicalEntity consumed — reactant or substrate>"],
      "output": ["<PhysicalEntity produced — product>"],
      "catalystActivity": {{
        "catalyst": "<the enzyme that DIRECTLY performs this reaction, or null>",
        "molecularFunction": "<GO molecular function term if explicitly stated, or null>"
      }},
      "regulatedBy": [
        {{
          "regulationType": "<positiveRegulation | negativeRegulation | requirement>",
          "regulator": "<gene, protein, or small molecule>",
          "note": "<null, OR a curator flag when this regulator's effect is condition-dependent>"
        }}
      ],
      "compartment": "<cellular compartment if explicitly stated, or null>",
      "condition": "<general biological state/context only, coarse and reusable, or null>",
      "summation": {{
        "text": "<factual description of the molecular event; keep it about the biology>",
        "literatureReference": ["<PMID or citation string if mentioned, else empty list>"]
      }},
      "relationships": ["EntityA - relationship_type -> EntityB"],
      "evidence": ["<verbatim excerpt(s) from the abstract supporting this reaction and each filled subsection>"],
      "context_used": ["none"],
      "confidence": <float 0-1, modest for abstract-level evidence>
    }}
  ]
}}

If no biochemical reaction is explicitly present in the abstract, return {{"reactions": []}}.
Do not speculate or merely list interacting genes if there is no stated molecular event.

=== ABSTRACT — extract reactions from THIS text only ===
{abstract_text}"""


def _parse_reactions(raw):
    """Mirror run_extraction.py's tolerant JSON parse: strip fences, isolate the object."""
    raw = (raw or "").replace("```json", "").replace("```", "").strip()
    if not raw.startswith("{"):
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("abstract extraction JSON parse failed: %s", e)
        return []
    return obj.get("reactions", []) or []


# ---- cross-model review (OpenAI) --------------------------------------------------------
# Rubric mirrors run_review.py's REVIEW_SYSTEM so an abstract score is calibrated the same way as
# a full-text score. KEEP IN SYNC with run_review.py if that rubric changes. The one difference is
# the "this is an ABSTRACT" framing: an abstract states conclusions with little mechanistic detail,
# so a record that omits detail the abstract never gave is NOT a defect.
_CONVENTIONS = (
    "Two Reactome conventions worth holding them to:\n"
    "- the CATALYST directly performs the chemistry. An entity that facilitates, promotes, enables "
    "or is merely required for a reaction is a REGULATOR, not a catalyst. Transport, binding, "
    "translocation and conformational change usually have no catalyst at all.\n"
    "- regulation is additive across experiments — two records testing different regulators of the "
    "same event are still the same event.")

_REVIEW_SYSTEM = f"""You are an experienced Reactome biocurator giving a colleague a second opinion.
Another AI read this paper's ABSTRACT and pulled reaction records out of it. You are seeing the
same abstract it saw and the records it produced.

IMPORTANT: this is an ABSTRACT, not a full paper. An abstract states conclusions with little
mechanistic detail. A record that omits a catalyst, compartment, site, or intermediate the
abstract never states is NOT incomplete — judge only whether the records faithfully reflect what
the abstract actually says. Fault a record only when it misrepresents the abstract or asserts an
event the abstract does not state.

Judge against the abstract text only. Something the abstract does not say is not supported, however
plausible. Be objective, not exacting: every criticism must point at a specific record by number
and quote the text that contradicts it. If you cannot point at the text, it is a preference, not a
defect.

This output is a STARTING POINT a curator will edit, not a finished record.
- a real defect changes the biology: a participant the abstract never names, a catalyst that is
  really a regulator, a direction reversed, two distinct events fused, one event split in two, a
  claim with no support in the abstract.
- not a defect: imprecise-but-correct naming, a summation you would phrase differently, a missing
  detail the abstract does not supply.

Score on this scale, and use the middle of it — most competent output lands at 6-8:
- 9-10: you would hand it to a curator as-is; nothing you found changes the biology.
- 7-8: sound. Real biology, with small fixes a curator makes in minutes.
- 5-6: usable but needs work — one or two records are wrong or a real event was missed.
- 3-4: several records misrepresent the abstract.
- 0-2: faster to start over.

{_CONVENTIONS}"""


def _brief_reaction(ar, i):
    """One abstract reaction, trimmed to what a reviewer needs (mirrors run_review.brief)."""
    ca = ar.get("catalystActivity") or {}
    if isinstance(ca, list):
        ca = {}
    regs = ar.get("regulatedBy") or []
    s = ar.get("summation")
    summ = s.get("text") if isinstance(s, dict) else (s if isinstance(s, str) else "")
    lines = [f"### {i}. {ar.get('name', '')}",
             f"- type: {ar.get('reactionType', '')}",
             f"- input: {', '.join(ar.get('input') or []) or 'none'}",
             f"- output: {', '.join(ar.get('output') or []) or 'none'}",
             f"- catalyst: {ca.get('catalyst') or 'none'} ({ca.get('molecularFunction') or 'none'})",
             "- regulation: " + (", ".join(
                 f"{r.get('regulationType')} by {r.get('regulator')}"
                 for r in regs if isinstance(r, dict)) or "none"),
             f"- compartment: {ar.get('compartment') or 'none'}",
             f"- summation: {summ or 'none'}"]
    for q in ar.get("evidence") or []:
        lines.append(f'- evidence: "{q}"')
    return "\n".join(lines)


def review_abstract(pmid, gene, abstract_text, wrapped_reactions, model=MODEL_REVIEW):
    """Cross-model (OpenAI) second opinion on abstract-extracted reactions -> (score, review_md).

    Mirrors run_review.py but with the abstract as the source text. Best-effort: returns
    (None, None) if OPENAI_API_KEY is missing or the call fails. Folds token usage into USAGE."""
    if not wrapped_reactions:
        return None, None
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("abstract review skipped for %s: OPENAI_API_KEY not set", pmid)
        return None, None

    reactions_md = "\n\n".join(
        _brief_reaction(r.get("annotation_result", r), i)
        for i, r in enumerate(wrapped_reactions, 1))
    user = (f"## SOURCE TEXT — the abstract for {gene} (PMID {pmid})\n\n{abstract_text}\n\n"
            f"## FINAL OUTPUT — {len(wrapped_reactions)} reaction(s) extracted from the abstract\n\n"
            f"{reactions_md}\n\n"
            f"## Your review\n\nStart with a single line, exactly:\n\nSCORE: <0-10>\n\n"
            f"That is your overall confidence in this output as a starting point for curation. "
            f"Then, in markdown, briefly: **Overall** (would you trust it?), **Extraction** (did "
            f"it catch the real biology the abstract states; call out anything invented or "
            f"overstated, naming the record and quoting the abstract), and **Look twice** (records "
            f"where the biology may be wrong — skip if none). Every point needs a quoted example.")
    try:
        resp = _get_oai().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _REVIEW_SYSTEM},
                      {"role": "user", "content": user}])
    except Exception as e:
        logger.warning("abstract review call failed for %s: %s", pmid, e)
        return None, None

    review = (resp.choices[0].message.content or "").strip()
    u = getattr(resp, "usage", None)
    if u:
        USAGE["calls"] += 1
        USAGE["input"] += getattr(u, "prompt_tokens", 0) or 0
        USAGE["output"] += getattr(u, "completion_tokens", 0) or 0
    m = re.search(r"SCORE:\s*([\d.]+)", review)
    score = float(m.group(1)) if m else None
    return score, review


def extract_abstract(pmid, gene, fallback_text=None, review=True):
    """Extract reactions from ONE paper's abstract. Returns a per-paper result dict shaped like
    fulltext_extractor's per-paper results (pmid, source, n_extracted, n_merged, reactions,
    review_score, review_path, ...), so the caller can treat abstract and full-text papers
    uniformly. Never raises.

    Each reaction is wrapped in the same envelope the full-text extractor writes
    ({source, chunk_index, word_count, annotation_result}) plus provenance="abstract". With
    review=True and OPENAI_API_KEY set, a cross-model OpenAI reviewer scores the extraction the
    same way run_review scores full text, and the review markdown is written to results/.
    """
    pmid = str(pmid)
    result = {"pmid": pmid, "source": "abstract", "n_extracted": 0, "n_merged": 0,
              "reactions": [], "review_score": None, "review_path": None,
              "extraction_path": None, "merged_path": None,
              "provenance": "abstract", "ok": False}

    text = fetch_abstract(pmid, fallback_text=fallback_text)
    if not text:
        logger.info("abstract: no abstract available for %s -- dropped", pmid)
        return result

    prompt = build_abstract_extraction_prompt(text, gene=gene)
    try:
        msg = _get_client().messages.create(
            model=MODEL_NAME, max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        logger.warning("abstract extraction API call failed for %s: %s", pmid, e)
        return result
    _track_usage(msg)

    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    reactions = _parse_reactions(raw)
    word_count = len(text.split())
    wrapped = [
        {"source": f"{pmid} (abstract)", "chunk_index": 0, "word_count": word_count,
         "annotation_result": r, "provenance": "abstract"}
        for r in reactions
    ]
    result["reactions"] = wrapped
    result["n_extracted"] = len(wrapped)
    result["ok"] = True
    logger.info("abstract %s: %d reaction(s) extracted from abstract", pmid, len(wrapped))

    # Cross-model OpenAI review, comparable to the full-text run_review score.
    if review and wrapped:
        score, review_md = review_abstract(pmid, gene, text, wrapped)
        result["review_score"] = score
        if review_md:
            os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
            review_path = os.path.join(PROJECT_ROOT, "results", f"{pmid}_abstract_review.md")
            header = (f"# Second opinion (abstract) — PMID {pmid} · {gene}\n\n"
                      f"reviewer: {MODEL_REVIEW} | {len(wrapped)} reaction(s) from abstract\n\n---\n\n")
            with open(review_path, "w") as f:
                f.write(header + review_md + "\n")
            result["review_path"] = review_path
        logger.info("abstract %s: review score %s", pmid,
                    f"{score}/10" if score is not None else "n/a")
    return result
