from __future__ import annotations

import threading
from typing import Any, Callable, Optional


_EventSink = Callable[[dict[str, Any]], None]
_context = threading.local()


def set_event_sink(sink: _EventSink, job_id: Optional[str] = None) -> None:
    _context.sink = sink
    _context.job_id = job_id


def clear_event_sink() -> None:
    _context.sink = None
    _context.job_id = None


def current_job_id() -> Optional[str]:
    return getattr(_context, "job_id", None)


def emit_event(event_type: str, status: str, **fields: Any) -> None:
    sink = getattr(_context, "sink", None)
    if sink is None:
        return
    event = {
        "event_type": event_type,
        "status": status,
    }
    event.update({key: value for key, value in fields.items() if value is not None})
    sink(event)


def emit_job_event(status: str, **fields: Any) -> None:
    emit_event("job", status, **fields)


def emit_agent_event(agent: str, status: str, **fields: Any) -> None:
    emit_event("agent", status, agent=agent, **fields)


def emit_tool_event(tool: str, status: str, **fields: Any) -> None:
    emit_event("tool", status, tool=tool, **fields)
