from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ModelTurn:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = 'end_turn'
    content: List[Dict[str, Any]] = field(default_factory=list)   # assistant blocks, replayed as history
    usage: Optional[TokenUsage] = None                            # what the provider reported for this call


class ChatModel(Protocol):
    def turn(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             on_text: Callable[[str], None]) -> ModelTurn:
        """One model call. `on_text` receives text deltas as they stream."""
        ...
