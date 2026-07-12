"""Ollama LLM provider implementation."""

from typing import Any

import httpx

from app.config.settings import get_settings
from app.llm.base import LLMError, LLMTimeoutError


class OllamaProvider:
    """Generate completions through Ollama's chat API."""

    async def generate(self, system: str, user: str, timeout_seconds: float) -> str:
        """Return raw assistant content from Ollama."""

        settings = get_settings()
        payload: dict[str, Any] = {
            "model": settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            msg = "Ollama request timed out"
            raise LLMTimeoutError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"Ollama returned HTTP {exc.response.status_code}"
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "Ollama request failed"
            raise LLMError(msg) from exc

        data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            msg = "Ollama response did not contain assistant content"
            raise LLMError(msg)
        return content
