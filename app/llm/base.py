"""Shared LLM provider contracts and exceptions."""

from typing import Protocol


class LLMError(Exception):
    """Raised when an LLM provider request fails."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider request exceeds its timeout. Transient — retryable."""


class LLMConnectionError(LLMError):
    """Raised when an LLM provider request fails at the network/connection level.

    Transient — retryable.
    """


class LLMServerError(LLMError):
    """Raised when an LLM provider responds with an HTTP 5xx status. Transient — retryable."""


# Exception types the LLM client layer retries with backoff. Anything else raised by a
# provider (plain LLMError for 4xx/malformed responses) is treated as non-transient and
# propagates immediately without a retry.
RETRYABLE_LLM_ERRORS: tuple[type[LLMError], ...] = (
    LLMTimeoutError,
    LLMConnectionError,
    LLMServerError,
)


class LLMProvider(Protocol):
    """Protocol implemented by concrete LLM providers."""

    async def generate(self, system: str, user: str, timeout_seconds: float) -> str:
        """Generate raw model output for a system and user prompt."""
