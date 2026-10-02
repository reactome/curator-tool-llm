"""Language-model token usage, recorded per step so a curator can see where the tokens went.

Steps that call no model (checking quotes against the paper, resolving identifiers, looking for existing reactions,
exporting) are not listed: they spend nothing.

`input_tokens` and `output_tokens` are what the provider reports for the calls; cache reads and writes are reported
separately by the provider and are kept separate here. `total_tokens` in a summary is input plus output.
"""
import time
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

Step = Literal['extraction', 'merge', 'review', 'draft', 'qa', 'chat']
STEP_ORDER: List[str] = ['extraction', 'merge', 'review', 'draft', 'qa', 'chat']
# the steps that produce the annotation; a replayed session carries their usage from the saved run, while checks and
# chat done in the session are always live
PIPELINE_STEPS = ('extraction', 'merge', 'review', 'draft')
_COUNTERS = ('calls', 'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')


class UsageEntry(BaseModel):
    step: Step
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: Optional[str] = None
    detail: Optional[str] = None          # what it was for: a reaction key for a check, "turn 3" for a chat turn
    at: float = Field(default_factory=time.time)

    @property
    def empty(self) -> bool:
        return not any(getattr(self, c) for c in _COUNTERS)


def entry_from_counts(step: str, counts: Dict[str, int], model: Optional[str] = None, detail: Optional[str] = None) -> UsageEntry:
    """From the dict shape the scripts' output is parsed into (calls / input / output / cache_read / cache_write)."""
    return UsageEntry(step=step, calls=counts.get('calls', 0), input_tokens=counts.get('input', 0),
                      output_tokens=counts.get('output', 0), cache_read_tokens=counts.get('cache_read', 0),
                      cache_write_tokens=counts.get('cache_write', 0), model=model, detail=detail)


def _totals(rows: List[Dict[str, object]]) -> Dict[str, int]:
    t = {c: sum(r[c] for r in rows) for c in _COUNTERS}
    t['total_tokens'] = t['input_tokens'] + t['output_tokens']
    return t


def summarize(entries: Iterable[UsageEntry], saved_pipeline: bool = False) -> Dict[str, object]:
    """Per-step totals in pipeline order (only steps that appear), the totals of everything listed, and the totals
    actually spent in this session.

    With `saved_pipeline` (the result was replayed from a snapshot) the steps that produce the annotation are flagged
    `saved`: their numbers are from the original run and nothing was spent on them now. `spent_now` leaves those out.
    """
    agg: Dict[str, Dict[str, int]] = {}
    for e in entries:
        a = agg.setdefault(e.step, {c: 0 for c in _COUNTERS})
        for c in _COUNTERS:
            a[c] += getattr(e, c)
    steps = [{'step': s, **agg[s], 'total_tokens': agg[s]['input_tokens'] + agg[s]['output_tokens'],
              'saved': saved_pipeline and s in PIPELINE_STEPS}
             for s in STEP_ORDER if s in agg]
    return {'steps': steps, 'totals': _totals(steps), 'spent_now': _totals([s for s in steps if not s['saved']])}
