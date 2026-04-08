from typing import Tuple

from langchain_openai import ChatOpenAI

REACTOME_MODEL_NAME = "gpt-5-mini"
REACTOME_MODEL_TEMPERATURE = 1.0
CREWAI_MODEL_NAME = "gpt-5-nano"
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


def create_reactome_chat_model() -> ChatOpenAI:
    """Create ChatOpenAI instance for the base Reactome pipeline."""
    model_name, temperature = get_reactome_model_settings()
    return ChatOpenAI(temperature=temperature, model=model_name)


def create_crewai_chat_model() -> ChatOpenAI:
    """Create ChatOpenAI instance for the CrewAI pipeline."""
    model_name, temperature = get_crewai_model_settings()
    return ChatOpenAI(temperature=temperature, model=model_name)