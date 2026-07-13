"""vLLM OpenAI-compatible provider implementation."""

import re
from typing import Any

import httpx

from app.config.settings import get_settings
from app.llm.base import LLMError, LLMTimeoutError
from app.logging.logger import get_logger

logger = get_logger(__name__)

_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_thinking_block(raw_output: str) -> str:
    """Remove a <think>...</think> reasoning block from raw LLM output, if present.

    This is a safe no-op if no <think> block exists in the input. Any whitespace
    immediately following the closing </think> tag is also consumed, which is safe
    since JSON parsing ignores leading/trailing whitespace around the payload.
    """
    return _THINK_BLOCK_PATTERN.sub("", raw_output, count=1)


class VLLMProvider:
    """Generate completions through vLLM's OpenAI-compatible API."""

    async def generate(self, system: str, user: str, timeout_seconds: float) -> str:
        """Return raw assistant content from vLLM."""

        settings = get_settings()
        payload: dict[str, Any] = {
            "model": settings.vllm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                headers = {
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json",
                }

                response = await client.post(
                    f"{settings.vllm_base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            msg = "vLLM request timed out"
            raise LLMTimeoutError(msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"vLLM returned HTTP {exc.response.status_code}"
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "vLLM request failed"
            raise LLMError(msg) from exc

        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            msg = "vLLM response did not contain choices"
            raise LLMError(msg)

        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            msg = "vLLM response did not contain assistant content"
            raise LLMError(msg)

        logger.debug(
            "llm_response_received",
            extra={
                "response_length_chars": len(content),
                "model": settings.vllm_model,
            },
        )

        cleaned_content = strip_thinking_block(content)
        return cleaned_content
