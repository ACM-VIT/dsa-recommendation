"""Thin LLM client and provider factory."""

import asyncio
from functools import lru_cache

from app.config.settings import get_settings
from app.llm.base import RETRYABLE_LLM_ERRORS, LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider
from app.logging.logger import get_logger
from app.models.domain import LLMPrompt

logger = get_logger(__name__)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider."""

    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    return VLLMProvider()


def get_active_model_name() -> str:
    """Return the configured model name for the active provider."""

    settings = get_settings()
    if settings.llm_provider == "ollama":
        return settings.ollama_model
    return settings.vllm_model


class LLMClient:
    """Client used by the orchestrator to request raw structured completions."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        """Initialize the client with an optional provider override."""

        self._provider = provider or get_llm_provider()

    async def get_structured_completion(self, prompt: LLMPrompt) -> str:
        """Return raw model output for a strict structured prompt.

        Transient provider failures (timeouts, connection errors, HTTP 5xx) are
        retried with exponential backoff, up to ``llm_max_retries`` additional
        attempts. Non-transient failures (4xx, malformed responses) propagate
        immediately without a retry.
        """

        settings = get_settings()
        attempts = settings.llm_max_retries + 1

        for attempt in range(attempts):
            try:
                return await self._provider.generate(
                    prompt.system,
                    prompt.user,
                    timeout_seconds=settings.llm_timeout_seconds,
                )
            except RETRYABLE_LLM_ERRORS:
                if attempt == attempts - 1:
                    raise
                delay = settings.llm_retry_backoff_seconds * (2**attempt)
                logger.warning(
                    "llm request failed, retrying",
                    extra={
                        "attempt": attempt + 1,
                        "max_attempts": attempts,
                        "delay_seconds": delay,
                    },
                )
                await asyncio.sleep(delay)

        msg = "unreachable: retry loop exited without returning or raising"
        raise AssertionError(msg)
