from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ModelTurn:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = 'end_turn'
    content: List[Dict[str, Any]] = field(default_factory=list)   # assistant blocks, replayed as history


class ChatModel(Protocol):
    def turn(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             on_text: Callable[[str], None]) -> ModelTurn:
        """One model call. `on_text` receives text deltas as they stream."""
        ...
