from typing import Tuple
from langchain_anthropic import ChatAnthropic

REACTOME_MODEL_NAME = "claude-sonnet-4-6"
REACTOME_MODEL_TEMPERATURE = 1.0
# CrewAI 1.14.7 only recognizes Claude 4.5 models for NATIVE structured output (output_pydantic).
# On 4.6 it silently falls back to a forced-tool call that collides with CrewAI's own assistant-
# message prefill in the tool loop -> "this model does not support assistant message prefill".
# 4.5 takes the clean native path, so structured output works. (Base model above is unaffected —
# it's used by the LangChain annotator, not CrewAI.)
CREWAI_MODEL_NAME = "anthropic/claude-sonnet-4-5-20250929"
CREWAI_MODEL_TEMPERATURE = 0.1

def get_reactome_model_settings() -> Tuple[str, float]:
    """
    Resolve base Reactome LLM model settings.
    Model settings are intentionally hard-coded in this module.
    """
    return REACTOME_MODEL_NAME, REACTOME_MODEL_TEMPERATURE

def get_crewai_model_settings() -> Tuple[str, float]:
    """
    Resolve CrewAI-specific model settings.
    Model settings are intentionally hard-coded in this module.
    """
    return CREWAI_MODEL_NAME, CREWAI_MODEL_TEMPERATURE

def create_reactome_chat_model() -> ChatAnthropic:
    """Create ChatAnthropic instance for the base Reactome pipeline."""
    model_name, temperature = get_reactome_model_settings()
    return ChatAnthropic(temperature=temperature, model=model_name)

def create_crewai_chat_model() -> ChatAnthropic:
    """Create ChatAnthropic instance for the CrewAI pipeline."""
    model_name, temperature = get_crewai_model_settings()
    return ChatAnthropic(temperature=temperature, model=model_name)