"""Shared LLM provider contracts and exceptions."""

from typing import Protocol


class LLMError(Exception):
    """Raised when an LLM provider request fails."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider request exceeds its timeout."""


class LLMProvider(Protocol):
    """Protocol implemented by concrete LLM providers."""

    async def generate(self, system: str, user: str, timeout_seconds: float) -> str:
        """Generate raw model output for a system and user prompt."""
