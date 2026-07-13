"""LLM client and provider tests."""

from types import SimpleNamespace

import httpx
import pytest

from app.llm import client as client_module
from app.llm import ollama_provider, vllm_provider
from app.llm.base import LLMError, LLMTimeoutError
from app.llm.client import LLMClient, get_llm_provider
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider
from app.models.domain import LLMPrompt


class _FakeAsyncClient:
    """Minimal async client fake for provider tests."""

    response: httpx.Response | None = None
    exception: Exception | None = None

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict | None = None) -> httpx.Response:
        self.url = url
        self.json = json
        self.headers = headers
        if self.exception is not None:
            raise self.exception
        if self.response is None:
            msg = "test response not configured"
            raise AssertionError(msg)
        return self.response


def _response(status_code: int, payload: dict) -> httpx.Response:
    """Build an httpx response with a request attached."""

    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "http://test.local"),
    )


@pytest.mark.asyncio
async def test_ollama_provider_success(monkeypatch) -> None:
    """Ollama provider returns raw assistant content."""

    _FakeAsyncClient.response = _response(
        200,
        {"message": {"content": '{"feedback_text":"ok"}'}},
    )
    _FakeAsyncClient.exception = None
    monkeypatch.setattr(ollama_provider.httpx, "AsyncClient", _FakeAsyncClient)

    result = await OllamaProvider().generate("system", "user", timeout_seconds=3)

    assert result == '{"feedback_text":"ok"}'


@pytest.mark.asyncio
async def test_ollama_provider_timeout(monkeypatch) -> None:
    """Ollama timeout errors are normalized."""

    _FakeAsyncClient.response = None
    _FakeAsyncClient.exception = httpx.TimeoutException("slow")
    monkeypatch.setattr(ollama_provider.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(LLMTimeoutError):
        await OllamaProvider().generate("system", "user", timeout_seconds=3)


@pytest.mark.asyncio
async def test_ollama_provider_non_2xx(monkeypatch) -> None:
    """Ollama non-success responses raise LLMError."""

    _FakeAsyncClient.response = _response(500, {"error": "boom"})
    _FakeAsyncClient.exception = None
    monkeypatch.setattr(ollama_provider.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(LLMError):
        await OllamaProvider().generate("system", "user", timeout_seconds=3)


@pytest.mark.asyncio
async def test_vllm_provider_success(monkeypatch) -> None:
    """vLLM provider returns raw assistant content."""

    _FakeAsyncClient.response = _response(
        200,
        {"choices": [{"message": {"content": '{"feedback_text":"ok"}'}}]},
    )
    _FakeAsyncClient.exception = None
    monkeypatch.setattr(vllm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    result = await VLLMProvider().generate("system", "user", timeout_seconds=3)

    assert result == '{"feedback_text":"ok"}'


@pytest.mark.asyncio
async def test_vllm_provider_timeout(monkeypatch) -> None:
    """vLLM timeout errors are normalized."""

    _FakeAsyncClient.response = None
    _FakeAsyncClient.exception = httpx.TimeoutException("slow")
    monkeypatch.setattr(vllm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(LLMTimeoutError):
        await VLLMProvider().generate("system", "user", timeout_seconds=3)


@pytest.mark.asyncio
async def test_vllm_provider_non_2xx(monkeypatch) -> None:
    """vLLM non-success responses raise LLMError."""

    _FakeAsyncClient.response = _response(503, {"error": "unavailable"})
    _FakeAsyncClient.exception = None
    monkeypatch.setattr(vllm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(LLMError):
        await VLLMProvider().generate("system", "user", timeout_seconds=3)


def test_get_llm_provider_selects_ollama(monkeypatch) -> None:
    """Provider factory selects Ollama from settings."""

    get_llm_provider.cache_clear()
    monkeypatch.setattr(
        client_module,
        "get_settings",
        lambda: SimpleNamespace(llm_provider="ollama"),
    )

    assert isinstance(get_llm_provider(), OllamaProvider)


def test_get_llm_provider_selects_vllm(monkeypatch) -> None:
    """Provider factory selects vLLM from settings."""

    get_llm_provider.cache_clear()
    monkeypatch.setattr(
        client_module,
        "get_settings",
        lambda: SimpleNamespace(llm_provider="vllm"),
    )

    assert isinstance(get_llm_provider(), VLLMProvider)


@pytest.mark.asyncio
async def test_llm_client_uses_configured_timeout(monkeypatch) -> None:
    """LLMClient passes configured timeout to the provider."""

    class Provider:
        async def generate(
            self,
            system: str,
            user: str,
            timeout_seconds: float,
        ) -> str:
            assert system == "system"
            assert user == "user"
            assert timeout_seconds == 4
            return "raw"

    monkeypatch.setattr(
        client_module,
        "get_settings",
        lambda: SimpleNamespace(llm_timeout_seconds=4),
    )

    result = await LLMClient(Provider()).get_structured_completion(
        LLMPrompt(system="system", user="user"),
    )

    assert result == "raw"


@pytest.mark.asyncio
async def test_vllm_log_metadata_only_no_raw_content(monkeypatch, caplog) -> None:
    """vLLM provider logs metadata only — raw content must not appear in the log extra."""

    import logging

    raw_content = '{"feedback_text":"ok"}'

    _FakeAsyncClient.response = _response(
        200,
        {"choices": [{"message": {"content": raw_content}}]},
    )
    _FakeAsyncClient.exception = None
    monkeypatch.setattr(vllm_provider.httpx, "AsyncClient", _FakeAsyncClient)

    with caplog.at_level(logging.DEBUG, logger="app.llm.vllm_provider"):
        await VLLMProvider().generate("system", "user", timeout_seconds=3)

    # Find the llm_response_received log record
    debug_records = [r for r in caplog.records if r.getMessage() == "llm_response_received"]
    assert debug_records, "Expected a 'llm_response_received' debug log record"

    log_extra = debug_records[0].__dict__
    assert raw_content not in str(log_extra), "Raw LLM content must not be logged"
    assert "response_length_chars" in log_extra
    assert "model" in log_extra


def test_strip_thinking_block_basic() -> None:
    """Basic case: input with a <think>...</think> block followed by valid JSON."""
    raw = "<think>\nThinking...\n</think>\n\n{\"key\": \"value\"}"
    expected = "{\"key\": \"value\"}"
    assert vllm_provider.strip_thinking_block(raw) == expected


def test_strip_thinking_block_no_think_block() -> None:
    """No think block: input is already just the JSON, no <think> tag anywhere."""
    raw = "{\"key\": \"value\"}"
    assert vllm_provider.strip_thinking_block(raw) == raw


def test_strip_thinking_block_with_braces() -> None:
    """Think block containing braces: confirm these do not leak into the output."""
    raw = "<think>\n{0:1}, {1:1}\n</think>\n{\"key\": \"value\"}"
    expected = "{\"key\": \"value\"}"
    assert vllm_provider.strip_thinking_block(raw) == expected


def test_strip_thinking_block_unterminated() -> None:
    """Unterminated think block (no closing tag): returns input completely unchanged."""
    raw = "<think>\nUnfinished thought...\n{\"key\": \"value\"}"
    assert vllm_provider.strip_thinking_block(raw) == raw


def test_strip_thinking_block_multiple_occurrences() -> None:
    """Multiple think-like substrings: only the first real block is stripped."""
    raw = "<think>\nDone thinking.\n</think>\n{\"feedback_text\": \"Use <think> tags\"}"
    expected = "{\"feedback_text\": \"Use <think> tags\"}"
    assert vllm_provider.strip_thinking_block(raw) == expected
