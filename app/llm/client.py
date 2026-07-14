"""Thin LLM client and provider factory."""

from functools import lru_cache

from app.config.settings import get_settings
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.vllm_provider import VLLMProvider
from app.models.domain import LLMPrompt


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
        """Return raw model output for a strict structured prompt."""

        settings = get_settings()
        return await self._provider.generate(
            prompt.system,
            prompt.user,
            timeout_seconds=settings.llm_timeout_seconds,
        )
