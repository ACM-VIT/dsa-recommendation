"""Settings tests."""

import pytest

from app.config.settings import Settings

ENV_VARS = [
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LOG_LEVEL",
    "MAX_SOURCE_CODE_CHARS",
    "MAX_PROBLEM_STATEMENT_CHARS",
    "PROMPT_MAX_CHARS",
    "RULE_ENGINE_ENABLED",
    "MAX_CONCEPT_GAPS",
    "SOLUTION_LEAK_LINE_THRESHOLD",
]


def test_settings_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Settings expose the documented defaults."""

    for env_var in ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "vllm"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5-coder:7b"
    assert settings.vllm_base_url == "https://ascuvum64parbo-8000.proxy.runpod.net"
    assert settings.vllm_model == "qwen2.5-coder:7b"
    assert settings.llm_timeout_seconds == 600
    assert settings.log_level == "INFO"
    assert settings.max_source_code_chars == 20000
    assert settings.max_problem_statement_chars == 4000
    assert settings.prompt_max_chars == 12000
    assert settings.rule_engine_enabled is True
    assert settings.max_concept_gaps == 8
    assert settings.solution_leak_line_threshold == 3
    assert settings.vllm_api_key == "test-key"


def test_settings_load_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Environment variables override defaults with typed values."""

    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("MAX_PROBLEM_STATEMENT_CHARS", "1200")
    monkeypatch.setenv("RULE_ENGINE_ENABLED", "false")
    monkeypatch.setenv("SOLUTION_LEAK_LINE_THRESHOLD", "4")
    monkeypatch.setenv("VLLM_API_KEY", "test-env-key")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "vllm"
    assert settings.llm_timeout_seconds == 8
    assert settings.max_problem_statement_chars == 1200
    assert settings.rule_engine_enabled is False
    assert settings.solution_leak_line_threshold == 4


def test_ollama_only_starts_without_provider_keys(monkeypatch) -> None:
    """Settings load cleanly with LLM_PROVIDER=ollama and no API keys at all."""

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.vllm_api_key is None


def test_vllm_without_api_key_raises(monkeypatch) -> None:
    """Settings raise a clear error when LLM_PROVIDER=vllm but VLLM_API_KEY is missing."""

    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    with pytest.raises(Exception, match="VLLM_API_KEY"):
        Settings(_env_file=None)
