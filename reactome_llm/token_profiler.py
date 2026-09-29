"""
Opt-in token-usage profiler for the Reactome annotation pipeline.

Purpose: give visibility into which LLM API calls consume the most tokens, so spend
can be reasoned about as usage scales. It is entirely GATED behind the TOKEN_PROFILE
environment variable (default OFF) -- when off, every hook here is a no-op and the
pipeline runs byte-for-byte unchanged.

Two capture mechanisms, because the pipeline has two LLM paths:

  A. CrewAI phases (Phases 1-5) run on CrewAI's native litellm-backed `LLM`. CrewAI
     exposes per-crew `usage_metrics` after a kickoff. `profile_kickoff(...)` snapshots
     a crew's metrics around one kickoff and records the delta (shared crew) or the raw
     value (a fresh crew, whose metrics belong entirely to that one kickoff).

  B. The description / gate / summary tools run on langchain_anthropic.ChatAnthropic.
     `TokenProfileCallback` (a LangChain BaseCallbackHandler) reads token counts off each
     response and attributes them to whatever `label(...)` context is active. Attach it
     once in ModelConfig.create_reactome_chat_model() so it covers every direct call.

All records land in one process-global registry; `emit_report(gene)` writes a per-call-site
CSV under data/ and prints a ranked summary with disproportion flags.
"""

import os
import csv
import logging
import threading
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sonnet-tier pricing (both claude-sonnet-4-5 and 4-6): $/1M tokens.
_INPUT_USD_PER_MTOK = 3.0
_OUTPUT_USD_PER_MTOK = 15.0


def enabled() -> bool:
    """True when TOKEN_PROFILE is set to a truthy value. Cheap; safe to call on hot paths."""
    return os.environ.get("TOKEN_PROFILE", "").strip().lower() not in ("", "0", "false", "no")


# --- run-scoped registry -------------------------------------------------------------

@dataclass
class UsageRecord:
    """One captured unit of token usage (one CrewAI kickoff, or one LangChain call)."""
    label: str          # call-site / phase identifier, e.g. "phase_1_literature_extraction"
    phase: str          # coarse grouping, e.g. "phase_1" / "phase_5" / "precompute"
    model: str
    input_tokens: int
    output_tokens: int
    calls: int          # underlying API requests (CrewAI: successful_requests; LangChain: 1)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


_RECORDS: List[UsageRecord] = []
_LOCK = threading.Lock()


def reset() -> None:
    """Clear the registry at the start of a run so a reused annotator can't bleed across genes."""
    with _LOCK:
        _RECORDS.clear()


def totals() -> dict:
    """Programmatic snapshot of everything recorded so far -> {calls, input, output, total}.

    Used by run_analysis.py to read the in-process (retrieval-side) token spend and combine it
    with the partner extractor's subprocess usage. Returns zeros when profiling is off/empty."""
    with _LOCK:
        calls = sum(r.calls for r in _RECORDS)
        inp = sum(r.input_tokens for r in _RECORDS)
        out = sum(r.output_tokens for r in _RECORDS)
    return {"calls": calls, "input": inp, "output": out, "total": inp + out}


def _add(record: UsageRecord) -> None:
    with _LOCK:
        _RECORDS.append(record)


# --- attribution label (LangChain path) ----------------------------------------------

_UNSET = "unattributed"
_current_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "token_profile_label", default=_UNSET)


@contextmanager
def label(name: str, force: bool = True):
    """Tag LangChain calls made within this block with `name`.

    force=True always sets the label. force=False sets it only if none is active yet, so an
    inner hub (GenePathwayAnnotator.invoke_llm) doesn't overwrite a more specific label set by
    its caller. Propagates across asyncio.to_thread (which copies the context), and LangChain
    fires on_llm_end synchronously in the same thread, so the label is visible to the callback.
    """
    if not enabled() or (not force and _current_label.get() != _UNSET):
        yield
        return
    token = _current_label.set(name)
    try:
        yield
    finally:
        _current_label.reset(token)


# --- CrewAI path: per-kickoff usage_metrics ------------------------------------------

def _snapshot(crew: Any) -> Tuple[int, int, int, int]:
    """(prompt_tokens, completion_tokens, total_tokens, successful_requests) from a crew, or zeros."""
    m = getattr(crew, "usage_metrics", None)
    if m is None:
        return (0, 0, 0, 0)
    return (
        int(getattr(m, "prompt_tokens", 0) or 0),
        int(getattr(m, "completion_tokens", 0) or 0),
        int(getattr(m, "total_tokens", 0) or 0),
        int(getattr(m, "successful_requests", 0) or 0),
    )


@contextmanager
def profile_kickoff(label_name: str, crew: Any, phase: Optional[str] = None):
    """Record the token cost of one CrewAI kickoff by diffing the crew's usage_metrics.

    Robust to both accumulate- and reset-per-kickoff semantics: if the "after" totals exceed
    "before", the crew accumulates and we take the delta; otherwise the crew reset and the
    "after" value IS this kickoff's cost. Fresh crews (before == 0) yield delta == after either way.
    """
    if not enabled():
        yield
        return
    before = _snapshot(crew)
    try:
        yield
    finally:
        after = _snapshot(crew)
        # Per-field: accumulate -> after-before; reset -> after. See docstring.
        pin, cin, tot, req = (
            (a - b) if a >= b else a for a, b in zip(after, before)
        )
        # Raw values logged so the accumulate-vs-reset semantics can be verified on a real run.
        logger.info(
            "[token_profile] %s: before=%s after=%s -> in=%d out=%d calls=%d",
            label_name, before, after, pin, cin, req,
        )
        _add(UsageRecord(
            label=label_name,
            phase=phase or _phase_of(label_name),
            model="claude-sonnet-4-5 (crewai)",
            input_tokens=pin,
            output_tokens=cin,
            calls=req,
        ))


def _phase_of(label_name: str) -> str:
    """Coarse phase bucket from a label like 'phase_5_vote:reviewer' -> 'phase_5'."""
    if label_name.startswith("phase_"):
        return "_".join(label_name.split("_")[:2]).split(":")[0]
    return "precompute"


# --- LangChain path: callback handler -------------------------------------------------

def langchain_callbacks() -> Optional[list]:
    """Callbacks to attach to a ChatAnthropic model, or None when profiling is off."""
    return [TokenProfileCallback()] if enabled() else None


def _extract_usage(response: Any) -> Tuple[int, int, str]:
    """Pull (input_tokens, output_tokens, model) out of a LangChain LLMResult.

    langchain-core 0.1.x stores Anthropic usage in llm_output['usage'] and/or on each
    generation's message.response_metadata['usage'] -- NOT in the newer .usage_metadata.
    We try each location and normalize the (input/prompt, output/completion) naming.
    """
    model = "claude-sonnet-4-6 (langchain)"

    def _norm(u: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        if not isinstance(u, dict):
            return None
        i = u.get("input_tokens", u.get("prompt_tokens"))
        o = u.get("output_tokens", u.get("completion_tokens"))
        if i is None and o is None:
            return None
        return int(i or 0), int(o or 0)

    # 1. llm_output (aggregate for the call)
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        model = llm_output.get("model") or llm_output.get("model_name") or model
        for key in ("usage", "token_usage"):
            got = _norm(llm_output.get(key, {}))
            if got:
                return got[0], got[1], model

    # 2. per-generation message metadata
    for gen_list in getattr(response, "generations", []) or []:
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg is None:
                continue
            meta = getattr(msg, "response_metadata", None) or {}
            model = meta.get("model") or model
            got = _norm(meta.get("usage", {}))
            if got:
                return got[0], got[1], model
            um = getattr(msg, "usage_metadata", None)  # newer langchain-core, just in case
            got = _norm(um or {})
            if got:
                return got[0], got[1], model
    return 0, 0, model


try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - langchain always present in this project
    BaseCallbackHandler = object  # type: ignore


class TokenProfileCallback(BaseCallbackHandler):
    """Records token usage from every ChatAnthropic response, tagged by the active label()."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            in_tok, out_tok, model = _extract_usage(response)
            if in_tok == 0 and out_tok == 0:
                logger.warning("[token_profile] LangChain call under '%s' reported no usage",
                               _current_label.get())
            _add(UsageRecord(
                label=_current_label.get(),
                phase="precompute",
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                calls=1,
            ))
        except Exception as e:  # never let profiling break a real call
            logger.warning("[token_profile] failed to record LangChain usage: %s", e)


# --- aggregation + report -------------------------------------------------------------

def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * _INPUT_USD_PER_MTOK + (output_tokens / 1e6) * _OUTPUT_USD_PER_MTOK


@dataclass
class _Agg:
    label: str
    phase: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return _cost_usd(self.input_tokens, self.output_tokens)


def _aggregate() -> List[_Agg]:
    """Collapse raw records to one row per (label, phase, model), summing tokens and calls."""
    by_key: Dict[Tuple[str, str, str], _Agg] = {}
    with _LOCK:
        for r in _RECORDS:
            key = (r.label, r.phase, r.model)
            a = by_key.get(key)
            if a is None:
                a = _Agg(label=r.label, phase=r.phase, model=r.model)
                by_key[key] = a
            a.calls += r.calls
            a.input_tokens += r.input_tokens
            a.output_tokens += r.output_tokens
    return sorted(by_key.values(), key=lambda a: a.total_tokens, reverse=True)


def emit_report(gene: str, out_dir: str = "data") -> Optional[str]:
    """Write per-call-site CSV + print a ranked summary. Returns the CSV path (or None if off/empty)."""
    if not enabled():
        return None
    aggs = _aggregate()
    if not aggs:
        logger.warning("[token_profile] enabled but no usage was recorded for %s", gene)
        return None

    grand_total = sum(a.total_tokens for a in aggs)
    grand_in = sum(a.input_tokens for a in aggs)
    grand_out = sum(a.output_tokens for a in aggs)
    grand_cost = _cost_usd(grand_in, grand_out)

    # --- CSV ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(out_dir) / f"token_usage_{gene}_{date.today().isoformat()}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["call_site", "phase", "model", "calls",
                    "input_tokens", "output_tokens", "total_tokens", "est_cost_usd"])
        for a in aggs:
            w.writerow([a.label, a.phase, a.model, a.calls,
                        a.input_tokens, a.output_tokens, a.total_tokens, f"{a.cost_usd:.4f}"])
        w.writerow(["TOTAL", "", "", sum(a.calls for a in aggs),
                    grand_in, grand_out, grand_total, f"{grand_cost:.4f}"])

    # --- ranked console summary ---
    lines = [
        "",
        f"===== TOKEN USAGE BREAKDOWN: {gene} =====",
        f"{'call_site':<38}{'total':>10}{'in':>10}{'out':>10}{'calls':>7}{'%':>7}  $",
    ]
    for a in aggs:
        share = 100.0 * a.total_tokens / grand_total if grand_total else 0.0
        lines.append(
            f"{a.label[:37]:<38}{a.total_tokens:>10,}{a.input_tokens:>10,}"
            f"{a.output_tokens:>10,}{a.calls:>7}{share:>6.1f}%  ${a.cost_usd:.4f}"
        )
    lines.append(f"{'GRAND TOTAL':<38}{grand_total:>10,}{grand_in:>10,}{grand_out:>10,}"
                 f"{'':>7}{'100.0%':>7}  ${grand_cost:.4f}")

    # --- disproportion flags ---
    flags = _disproportion_flags(aggs, grand_total)
    if flags:
        lines.append("")
        lines.append("FLAGS (potentially disproportionate):")
        lines.extend(f"  - {fl}" for fl in flags)
    lines.append(f"CSV written to {csv_path}")
    lines.append("=" * 46)

    report = "\n".join(lines)
    logger.info(report)
    print(report)
    return csv_path


def _disproportion_flags(aggs: List[_Agg], grand_total: int) -> List[str]:
    """Heuristics: any single call-site >30% of the run, and description-gen rivaling Phase-1."""
    flags: List[str] = []
    if grand_total <= 0:
        return flags
    for a in aggs:
        share = 100.0 * a.total_tokens / grand_total
        if share > 30.0:
            flags.append(f"{a.label} is {share:.1f}% of the whole run ({a.total_tokens:,} tokens).")
    phase1 = sum(a.total_tokens for a in aggs if a.phase == "phase_1")
    for a in aggs:
        if "description" in a.label.lower() or "desc" in a.label.lower():
            if phase1 > 0 and a.total_tokens >= phase1:
                flags.append(
                    f"{a.label} ({a.total_tokens:,}) >= Phase-1 extraction ({phase1:,}) "
                    f"-- a support call rivaling the main extraction.")
    return flags
