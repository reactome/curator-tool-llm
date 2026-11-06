"""What the language-model calls cost, from the token counts the usage records keep (models/usage.py).

Prices are US dollars per million tokens, Anthropic first-party API rates as published on 2026-10-06 (Claude API docs,
"Current Models"). Prompt caching: a cache write (5-minute TTL, the one the pipeline uses) costs 1.25 x input, a cache
read 0.1 x input. Change them without touching code with LLM_PRICES, a JSON object of model -> {"input", "output",
"cache_write", "cache_read"} (any field left out keeps its default), e.g.

    LLM_PRICES='{"claude-sonnet-5": {"input": 2.0, "output": 10.0}}'

These are estimates from the tool's own counts: the bill in the Anthropic Console is the authority.
"""
import json
import os
from typing import Dict, Iterable, Optional

CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.10

_BASE = {                      # input, output per million tokens
    'claude-sonnet-5': (2.00, 10.00),
    'claude-sonnet-5-5': (2.00, 10.00),
    'claude-sonnet-4-6': (3.00, 15.00),
    'claude-opus-5-5': (4.00, 20.00),
    'claude-opus-5': (5.00, 25.00),
    'claude-opus-4-8': (5.00, 25.00),
    'claude-haiku-5-5': (0.10, 0.50),
    'claude-haiku-4-5': (1.00, 5.00),
}

# usage step -> the model_settings role whose model that step calls (when a usage entry does not name its model)
STEP_ROLE = {'extraction': 'extraction', 'merge': 'merge', 'draft': 'draft', 'qa': 'qa', 'chat': 'chat',
             'combine': 'merge', 'sections': 'sections', 'review': 'review'}


def prices() -> Dict[str, Dict[str, float]]:
    out = {m: {'input': i, 'output': o, 'cache_write': i * CACHE_WRITE_FACTOR, 'cache_read': i * CACHE_READ_FACTOR}
           for m, (i, o) in _BASE.items()}
    try:
        extra = json.loads(os.getenv('LLM_PRICES') or '{}')
    except ValueError:
        extra = {}
    for model, p in extra.items():
        base = out.get(model, {})
        i = float(p.get('input', base.get('input', 0.0)))
        out[model] = {'input': i, 'output': float(p.get('output', base.get('output', 0.0))),
                      # cache prices follow the input price unless given: a new input price moves them too
                      'cache_write': float(p.get('cache_write', i * CACHE_WRITE_FACTOR)),
                      'cache_read': float(p.get('cache_read', i * CACHE_READ_FACTOR))}
    return out


def model_of_step(step: str, model: Optional[str] = None) -> Optional[str]:
    if model:
        return model
    role = STEP_ROLE.get(step)
    if role is None:
        return None
    try:
        from curator_llm.model_settings import model_name
        return model_name(role)
    except Exception:
        return None


def cost(counts: Dict[str, int], model: Optional[str]) -> Optional[float]:
    """Dollars for one set of token counts on `model`, or None when its price is unknown."""
    p = prices().get(model or '')
    if p is None:
        return None
    return (counts.get('input_tokens', 0) * p['input'] + counts.get('output_tokens', 0) * p['output']
            + counts.get('cache_write_tokens', 0) * p['cache_write']
            + counts.get('cache_read_tokens', 0) * p['cache_read']) / 1_000_000


def cost_of_steps(steps: Iterable[dict]) -> Dict[str, object]:
    """Totals and dollars for usage rows ({'step', 'model'?, 'calls', 'input_tokens', ...}, as summarize() writes them).
    A step whose model has no price is counted in the tokens and listed under 'unpriced'."""
    keys = ('calls', 'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')
    total = {k: 0 for k in keys}
    usd, unpriced = 0.0, []
    for s in steps:
        for k in keys:
            total[k] += int(s.get(k) or 0)
        c = cost(s, model_of_step(s.get('step', ''), s.get('model')))
        if c is None:
            unpriced.append(s.get('step', '?'))
        else:
            usd += c
    return {**total, 'usd': usd, 'unpriced': unpriced}
