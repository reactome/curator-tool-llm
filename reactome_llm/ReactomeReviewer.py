"""ReactomeReviewer — sufficiency judge + retry proposer (agent 2 of 3).

The domain-expert reviewer: reads a CuratorResult and returns a Verdict
(sufficient | retry(+adjustment) | give_up).

Two implementations behind ONE interface:
  - LLM (default): a language model reasons over the retrieval scores, full-text hit-rate,
    partner review scores, extracted reactions, and predicted placement, and decides on its
    own — including inventing a concrete adjustment from the honored knob vocabulary.
  - Rule-based fallback: a transparent if/else ladder. Used when the LLM call fails, or when
    use_llm=False (free, deterministic — good for testing the loop).

Both return the SAME Verdict shape, so run_curator.py's loop never changes.

Safety: the Reviewer only PROPOSES. The orchestrator owns the retry cap, so no verdict can
cause an infinite loop. As a belt-and-suspenders, `_finalize` converts a "retry" issued on
the final allowed attempt (or a "retry" with no usable adjustment) into a terminal decision.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---- the Verdict run_curator consumes (unchanged) ----
@dataclass
class Verdict:
    decision: str                       # "sufficient" | "retry" | "give_up"
    adjustment: Optional[Dict[str, Any]]  # knobs for the NEXT Curator run (None unless retry)
    reason: str


# ---- structured-output schema the LLM fills ----
class Adjustment(BaseModel):
    """Knobs for the next Curator retrieval run. Set ONLY the ones you want to change."""
    additional_terms: Optional[str] = Field(
        None, description="Extra terms OR'd into the PubMed query to broaden/redirect the search.")
    avoid_pmids: List[str] = Field(
        default_factory=list, description="PMIDs to exclude so the next-best candidates surface "
        "(e.g. papers that had no full text).")
    min_score: Optional[int] = Field(
        None, description="Lower to keep weaker candidates, raise to be stricter (rubric floor).")
    candidate_pool: Optional[int] = Field(
        None, description="Widen the cross-encoder pool the judge sees.")
    max_papers: Optional[int] = Field(None, description="Select more/fewer final papers.")


class ReviewVerdict(BaseModel):
    decision: str = Field(..., description="One of: sufficient | retry | give_up")
    reason: str = Field(..., description="Concise expert rationale for the decision.")
    adjustment: Optional[Adjustment] = Field(
        None, description="REQUIRED when decision=retry: the knobs to change for the next attempt. "
        "Omit for sufficient/give_up.")


REVIEW_PROMPT = """You are a Reactome domain-expert reviewer: you know this field and its \
literature extremely well. A deterministic Curator just produced ONE annotation attempt for \
gene {gene}. Decide whether it is good enough to hand to QA, or whether the Curator should try \
again with a change you specify.

This is attempt {attempt_num} of {max_attempts} ({retries_left} retr{ies_word} left after this).

=== What the Curator produced this attempt ===
{context}

=== Adjustments already tried in earlier attempts ===
{history}

=== Your decision ===
Choose ONE decision:
- "sufficient": there are usable extracted reactions AND either the placement is confident OR \
the evidence is strong enough that a curator could place it. Hand off to QA.
- "retry": there is a FIXABLE shortfall and retries remain. You MUST provide an `adjustment`. \
Pick the change most likely to help; do NOT repeat an adjustment already tried:
    additional_terms  — broaden/redirect the query (e.g. add the pathway or a partner-complex term)
    avoid_pmids       — drop specific PMIDs (e.g. all that missed full text) so next-best surface
    min_score         — lower to admit weaker candidates, raise to be stricter
    candidate_pool    — widen the pool the judge scores
    max_papers        — select more/fewer papers
- "give_up": no usable evidence and no productive change left — flag for manual curator review.

Judge on substance (is the evidence specific and sufficient for a real annotation?), not just \
counts. If you are out of retries, prefer a terminal decision (sufficient if anything usable was \
extracted, else give_up)."""


class ReactomeReviewer:
    MIN_MEAN_SCORE = 5.0   # rule-fallback threshold: pool mean below this = weak

    def __init__(self, use_llm: bool = True, model: Any = None) -> None:
        self.use_llm = use_llm
        self.model = model

    def review(self, result, attempt: int, max_attempts: int,
               history: Optional[List] = None) -> Verdict:
        if self.use_llm:
            try:
                verdict = self._llm_review(result, attempt, max_attempts, history or [])
            except Exception as e:
                logger.warning(f"LLM reviewer failed ({e}); falling back to rules.")
                verdict = self._rule_review(result, attempt, max_attempts)
        else:
            verdict = self._rule_review(result, attempt, max_attempts)
        return self._finalize(verdict, result, attempt, max_attempts)

    # ------------------------------------------------------------------ LLM reviewer
    def _llm_review(self, result, attempt, max_attempts, history) -> Verdict:
        from ModelConfig import create_reactome_chat_model
        import token_profiler

        retries_left = max(0, max_attempts - 1 - attempt)
        prompt = REVIEW_PROMPT.format(
            gene=result.gene, attempt_num=attempt + 1, max_attempts=max_attempts,
            retries_left=retries_left, ies_word=("y" if retries_left == 1 else "ies"),
            context=self._context(result), history=self._history(history))

        model = self.model or create_reactome_chat_model()
        with token_profiler.label("reviewer_llm"):
            rv: ReviewVerdict = model.with_structured_output(ReviewVerdict).invoke(prompt)

        decision = (rv.decision or "").strip().lower()
        if decision not in ("sufficient", "retry", "give_up"):
            raise ValueError(f"LLM returned invalid decision: {rv.decision!r}")

        adj = None
        if decision == "retry" and rv.adjustment:
            a = rv.adjustment
            adj = {}
            if a.additional_terms:
                adj["additional_terms"] = a.additional_terms
            if a.avoid_pmids:
                adj["avoid_pmids"] = a.avoid_pmids
            if a.min_score is not None:
                adj["min_score"] = a.min_score
            if a.candidate_pool is not None:
                adj["candidate_pool"] = a.candidate_pool
            if a.max_papers is not None:
                adj["max_papers"] = a.max_papers
            adj = adj or None
        return Verdict(decision, adj, rv.reason or "")

    # ------------------------------------------------------------------ context builders
    @staticmethod
    def _context(result) -> str:
        r, ft, pl = result.retrieval or {}, result.fulltext or {}, result.placement or {}
        scores = {str(s.get("pmid")): s.get("score") for s in (r.get("per_paper_scores") or [])}
        lines = [
            f"accession: {result.accession}",
            f"retrieval: pool={r.get('pool_size')} candidates={r.get('candidate_pool')} "
            f"mean_score={r.get('mean_score')} dropped_below_floor={r.get('dropped_below_threshold')}",
            "selected papers (pmid | rubric score | title):",
        ]
        for p in (r.get("papers") or []):
            pid = str(p.get("pmid"))
            lines.append(f"  - {pid} | {scores.get(pid)} | {(p.get('title') or '')[:90]}")
        c = ft.get("counts") or {}
        lines.append(f"full text: {c.get('pdf', 0)} PDF / {c.get('xml', 0)} XML / {c.get('miss', 0)} miss")
        for pp in (ft.get("per_paper") or []):
            lines.append(f"  - {pp.get('pmid')} [{pp.get('source')}] extracted={pp.get('n_extracted')} "
                         f"merged={pp.get('n_merged')} review_score={pp.get('review_score')}")
        rxs = ft.get("reactions") or []
        lines.append(f"extracted reactions: {len(rxs)}")
        for rx in rxs[:5]:
            name = rx.get("name") or rx.get("displayName") or rx.get("reaction") or str(rx)[:80]
            lines.append(f"  - {str(name)[:110]}")
        lines.append(f"placement (deterministic): predicted={pl.get('predicted_pathway')} "
                     f"confident={pl.get('confident')} status={pl.get('status')}")
        return "\n".join(lines)

    @staticmethod
    def _history(history) -> str:
        if not history:
            return "(none — this is the first attempt)"
        out = []
        for h in history:
            out.append(f"  attempt {h.get('attempt')}: decision={h.get('decision')} "
                       f"adjustment={h.get('adjustment_applied') or 'none'}")
        return "\n".join(out)

    # ------------------------------------------------------------------ finalize / safety net
    @staticmethod
    def _finalize(verdict: Verdict, result, attempt: int, max_attempts: int) -> Verdict:
        """Guarantee termination and non-no-op retries regardless of what produced the verdict."""
        last = attempt >= max_attempts - 1
        if verdict.decision == "retry" and (last or not verdict.adjustment):
            if result.has_reactions():
                return Verdict("sufficient", None,
                               verdict.reason + " (no retries left / no adjustment; accepting "
                               "extracted reactions).")
            return Verdict("give_up", None,
                           verdict.reason + " (no retries left / no adjustment; flag for manual review).")
        return verdict

    # ------------------------------------------------------------------ rule-based fallback
    def _rule_review(self, result, attempt: int, max_attempts: int) -> Verdict:
        r, ft, pl = result.retrieval or {}, result.fulltext or {}, result.placement or {}
        last = attempt >= max_attempts - 1

        if not r.get("papers"):
            if last:
                return Verdict("give_up", None, "No candidate papers found; flag for manual review.")
            return Verdict("retry", self._broaden(result), "No candidates; broaden the query.")

        if result.has_reactions() and pl.get("confident"):
            return Verdict("sufficient", None,
                           f"{len(ft.get('reactions', []))} reaction(s); placement confident "
                           f"({pl.get('predicted_pathway')}).")

        if result.has_reactions() and not pl.get("confident"):
            if last:
                return Verdict("sufficient", None,
                               "Reactions extracted but placement weak; return unplaced for manual placement.")
            return Verdict("retry", self._broaden(result),
                           "Reactions found but placement weak; broaden to strengthen evidence.")

        counts = ft.get("counts") or {}
        total = sum(counts.values()) if counts else 0
        if total and counts.get("miss", 0) == total and not result.has_reactions():
            if last:
                return Verdict("give_up", None, "No full text and no reactions after retries; flag for review.")
            missed = [str(p.get("pmid")) for p in r["papers"] if p.get("pmid")]
            return Verdict("retry", {"avoid_pmids": missed, **self._broaden(result)},
                           "All selected papers missed full text; exclude them and pull next-best.")

        mean = r.get("mean_score")
        if isinstance(mean, (int, float)) and mean < self.MIN_MEAN_SCORE and not result.has_reactions():
            if last:
                return Verdict("give_up", None, f"Pool mean {mean} stayed weak; flag for review.")
            return Verdict("retry", {**self._broaden(result), "min_score": 2},
                           f"Weak pool (mean {mean}); broaden and lower the floor.")

        if not result.has_reactions():
            if last:
                return Verdict("give_up", None, "No reactions after retries; flag for review.")
            return Verdict("retry", self._broaden(result), "No reactions yet; broaden and retry.")

        return Verdict("sufficient", None, "Reactions extracted.")

    @staticmethod
    def _broaden(result) -> Dict[str, Any]:
        pw = (result.placement or {}).get("predicted_pathway")
        return {"additional_terms": pw} if pw else {"min_score": 2}
