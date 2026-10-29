import pytest

from curator_llm import model_settings as ms


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os
    for k in list(os.environ):
        if k.startswith('LLM_MODEL_') or k.startswith('LLM_THINKING_') or k in (
                'CHAT_MODEL', 'MERGE_MODEL', 'MERGE_THINKING', 'EXTRACT_THINKING'):
            monkeypatch.delenv(k)


def test_defaults_per_role():
    assert ms.anthropic_kwargs('extraction') == {'model': 'claude-sonnet-5', 'thinking': {'type': 'disabled'}}
    assert ms.anthropic_kwargs('chat') == {'model': 'claude-sonnet-5-5'}
    assert ms.langchain_kwargs('draft') == {'model': 'claude-sonnet-5-5', 'thinking': {'type': 'between_tools'}}
    assert ms.langchain_kwargs('gene') == {'model': 'claude-sonnet-4-6', 'temperature': 1.0}
    assert ms.langchain_kwargs('convert')['temperature'] == 0.1
    assert 'temperature' not in ms.langchain_kwargs('extraction')        # sonnet-5 rejects it


def test_env_overrides_model_and_thinking(monkeypatch):
    monkeypatch.setenv('LLM_MODEL_DRAFT', 'claude-sonnet-5')
    monkeypatch.setenv('LLM_THINKING_EXTRACTION', 'on')
    assert ms.model_name('draft') == 'claude-sonnet-5'
    assert 'thinking' not in ms.anthropic_kwargs('extraction')           # on = the model's own default
    monkeypatch.setenv('LLM_THINKING_DRAFT', 'off')
    assert ms.langchain_kwargs('draft')['thinking'] == {'type': 'disabled'}


def test_legacy_names_still_work(monkeypatch):
    monkeypatch.setenv('CHAT_MODEL', 'claude-x')
    monkeypatch.setenv('MERGE_THINKING', 'on')
    monkeypatch.setenv('MERGE_MODEL', 'claude-y')
    assert ms.model_name('chat') == 'claude-x'
    assert ms.anthropic_kwargs('merge') == {'model': 'claude-y'}


def test_unknown_role_is_an_error():
    with pytest.raises(KeyError):
        ms.spec('nope')


def test_thinking_off_syntax_depends_on_the_model(monkeypatch):
    assert ms.thinking_off('claude-sonnet-5') == {'type': 'disabled'}
    assert ms.thinking_off('claude-sonnet-5-5') == {'type': 'between_tools'}
    monkeypatch.setenv('LLM_MODEL_EXTRACTION', 'claude-sonnet-5-5')
    assert ms.anthropic_kwargs('extraction')['thinking'] == {'type': 'between_tools'}
