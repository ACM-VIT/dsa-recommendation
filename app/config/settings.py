"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings for the AI analysis service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["ollama", "vllm"] = "vllm"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "qwen2.5-coder:7b"
    llm_timeout_seconds: float = Field(default=20, gt=0)
    llm_max_tokens: int = Field(default=600, gt=0)
    log_level: str = "INFO"
    max_source_code_chars: int = Field(default=20000, gt=0)
    max_problem_statement_chars: int = Field(default=4000, gt=0)
    prompt_max_chars: int = Field(default=12000, gt=0)
    rule_engine_enabled: bool = True
    max_concept_gaps: int = Field(default=8, ge=0)
    solution_leak_line_threshold: int = Field(default=3, gt=0)
    vllm_api_key: str | None = Field(default=None, alias="VLLM_API_KEY")
    ai_service_api_key: str = Field(alias="AI_SERVICE_API_KEY")
    llm_max_retries: int = Field(default=2, ge=0)
    llm_retry_backoff_seconds: float = Field(default=0.5, gt=0)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=60, gt=0)
    rate_limit_window_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> "Settings":
        """Fail fast with a clear message if a provider's required key is missing."""
        if self.llm_provider == "vllm" and not self.vllm_api_key:
            msg = (
                "VLLM_API_KEY is required when LLM_PROVIDER=vllm but was not set. "
                "Set the VLLM_API_KEY environment variable."
            )
            raise ValueError(msg)
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
