from typing import Tuple
from langchain_anthropic import ChatAnthropic
import token_profiler

REACTOME_MODEL_NAME = "claude-sonnet-4-6"
REACTOME_MODEL_TEMPERATURE = 1.0

def get_reactome_model_settings() -> Tuple[str, float]:
    """
    Resolve base Reactome LLM model settings.
    Model settings are intentionally hard-coded in this module.
    """
    return REACTOME_MODEL_NAME, REACTOME_MODEL_TEMPERATURE

def create_reactome_chat_model() -> ChatAnthropic:
    """Create ChatAnthropic instance for the base Reactome pipeline."""
    model_name, temperature = get_reactome_model_settings()
    # Token-usage profiling is opt-in (TOKEN_PROFILE env var); returns None when off, so the
    # model is constructed exactly as before with no callback attached.
    callbacks = token_profiler.langchain_callbacks()
    return ChatAnthropic(temperature=temperature, model=model_name, callbacks=callbacks)
