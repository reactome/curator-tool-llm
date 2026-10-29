"""The one place that says which LLM each step uses. Stdlib only, so every script can import it.

Each step has a ROLE. To change a model, edit DEFAULTS below, or override it without touching code:

    LLM_MODEL_<ROLE>=claude-sonnet-5        which model the role calls     (e.g. LLM_MODEL_DRAFT)
    LLM_THINKING_<ROLE>=on|off              extended thinking, where the role sets it explicitly

Older names still work: CHAT_MODEL, MERGE_MODEL, MERGE_THINKING, EXTRACT_THINKING.

Notes from measuring on PINK1 (Kane et al. 2014):
  * Thinking is most of the latency of a claude-sonnet-5 call. Off, extraction took 32s instead of 119s and merge
    44s instead of 77s, with the same quotes verified and the same merges.
  * claude-sonnet-5 rejects `temperature`, so a role with temperature=None never sends one.
  * Draft: claude-sonnet-5-5 with thinking off took 17s (4-6: 46s) and typed ubiquitin as a protein. Plain
    claude-sonnet-5 was fast too (25s) but typed every ubiquitin as a small molecule (5 duplicate "UB" entries).
  * Extraction stays on claude-sonnet-5: 5.5 found fewer reactions on PINK1 (18 thinking off / 23 on, against 34),
    with far fewer regulators (9 / 8, against 28). It is faster (17s), but recall matters more than seconds here.
  * Merge: 5.5 as the duplicate judge grouped the same as 5 (24 of 25 merged pairs) but was no faster (54s vs 47s).
  * Confirmed on two more papers (clock_nature, clock_science): 5.5 extraction found 10-14 and 8-11 reactions against
    22 and 21 for sonnet-5 thinking off; the 5.5 draft was 2x faster than 4-6 with the same coverage (plain 5 failed
    schema validation on one draft).
  * Haiku was not a safe duplicate judge (missed 10 of 24 merges).
  * claude-sonnet-5-5 turns thinking off with {"type": "between_tools"}, not "disabled" (see thinking_off).
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    model: str
    thinking: Optional[bool] = None          # False: send thinking disabled. None: leave it to the model's default.
    temperature: Optional[float] = None      # None: do not send one


DEFAULTS: Dict[str, ModelSpec] = {
    # --- paper annotation service and its scripts (Anthropic) ---
    'extraction':  ModelSpec('claude-sonnet-5', thinking=False),   # run_extraction.py, abstract_extractor.py
    'sections':    ModelSpec('claude-sonnet-5', thinking=False),   # section-title checks (FullTextPDFSections always sends thinking off: max_tokens is 100-300)
    'merge':       ModelSpec('claude-sonnet-5', thinking=False),   # run_merge.py duplicate judge / consolidate / validate
    'draft':       ModelSpec('claude-sonnet-5-5', thinking=False),   # services/draft_builder.py (LangChain)
    'chat':        ModelSpec('claude-sonnet-5-5'),                 # chat agent and per-reaction review
    # --- gene-first pipeline (run_curator.py, LangChain) ---
    'gene':        ModelSpec('claude-sonnet-4-6', temperature=1.0),  # curator, reviewer, QA
    'convert':     ModelSpec('claude-sonnet-4-6', temperature=0.1),  # reaction_to_instances.py
    # --- cross-model review (OpenAI) ---
    'review':      ModelSpec('gpt-5.6-luna'),                      # run_review.py, abstract_extractor.py
}

_LEGACY_MODEL = {'chat': 'CHAT_MODEL', 'merge': 'MERGE_MODEL'}
_LEGACY_THINKING = {'extraction': 'EXTRACT_THINKING', 'merge': 'MERGE_THINKING'}


# claude-sonnet-5-5 does not accept {"type": "disabled"}: the API says to send "between_tools" instead (the model
# then does not think before answering). Add a prefix here when another model asks for the same.
_BETWEEN_TOOLS = ('claude-sonnet-5-5',)


def thinking_off(model: str) -> Dict[str, str]:
    return {'type': 'between_tools' if model.startswith(_BETWEEN_TOOLS) else 'disabled'}


def spec(role: str) -> ModelSpec:
    """The role's settings with environment overrides applied."""
    if role not in DEFAULTS:
        raise KeyError(f'unknown model role {role!r}; known: {sorted(DEFAULTS)}')
    base = DEFAULTS[role]
    key = role.upper()
    model = os.getenv(f'LLM_MODEL_{key}') or os.getenv(_LEGACY_MODEL.get(role, '-')) or base.model
    flag = os.getenv(f'LLM_THINKING_{key}') or os.getenv(_LEGACY_THINKING.get(role, '-'))
    thinking = base.thinking if flag is None or flag == '' else (flag.strip().lower() == 'on')
    return ModelSpec(model, thinking if thinking is not True else None, base.temperature)


def model_name(role: str) -> str:
    return spec(role).model


def anthropic_kwargs(role: str) -> Dict[str, Any]:
    """Keyword arguments for anthropic `client.messages.create` / `.stream`: the model, and thinking when disabled."""
    s = spec(role)
    kw: Dict[str, Any] = {'model': s.model}
    if s.thinking is False:
        kw['thinking'] = thinking_off(s.model)
    return kw


def langchain_kwargs(role: str) -> Dict[str, Any]:
    """Keyword arguments for `ChatAnthropic(...)`: the model, a temperature if the role has one, thinking if disabled."""
    s = spec(role)
    kw: Dict[str, Any] = {'model': s.model}
    if s.temperature is not None:
        kw['temperature'] = s.temperature
    if s.thinking is False:
        kw['thinking'] = thinking_off(s.model)
    return kw
