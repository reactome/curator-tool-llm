"""ChatModel on the Anthropic Messages API (streaming)."""
import os
from typing import Any, Callable, Dict, List

from curator_llm.ports.chat_model import ModelTurn, TokenUsage, ToolCall


class AnthropicChatModel:
    def __init__(self, client=None, model: str = None, max_tokens: int = 4096):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=120.0)
        self.client = client
        self.model = model or os.getenv('CHAT_MODEL', 'claude-sonnet-5-5')
        self.max_tokens = max_tokens

    def turn(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
             on_text: Callable[[str], None]) -> ModelTurn:
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system,
                                         messages=messages, tools=tools) as stream:
            for delta in stream.text_stream:
                on_text(delta)
            final = stream.get_final_message()
        text, calls, content = [], [], []
        for b in final.content:
            if b.type == 'text':
                text.append(b.text)
                content.append({'type': 'text', 'text': b.text})
            elif b.type == 'tool_use':
                calls.append(ToolCall(b.id, b.name, dict(b.input)))
                content.append({'type': 'tool_use', 'id': b.id, 'name': b.name, 'input': dict(b.input)})
        u = getattr(final, 'usage', None)
        usage = TokenUsage(getattr(u, 'input_tokens', 0) or 0, getattr(u, 'output_tokens', 0) or 0,
                           getattr(u, 'cache_read_input_tokens', 0) or 0,
                           getattr(u, 'cache_creation_input_tokens', 0) or 0) if u is not None else None
        return ModelTurn(''.join(text), calls, final.stop_reason or 'end_turn', content, usage)
