"""Orchestrator integration tests with mocked LLM clients."""

import pytest

from app.llm.base import LLMTimeoutError
from app.models.request_schemas import AnalyzeRequest
from app.orchestrator import orchestrator
from app.orchestrator.orchestrator import analyze_submission
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


class FakeLLMClient:
    """Fake LLM client used to assert orchestrator behavior."""

    calls = 0
    raw_output = ""
    exception: Exception | None = None

    async def get_structured_completion(self, prompt) -> str:  # noqa: ANN001
        """Return configured fake output or raise a configured exception."""

        self.__class__.calls += 1
        if self.exception is not None:
            raise self.exception
        return self.raw_output


def _request(payload: dict) -> AnalyzeRequest:
    """Validate a request payload."""

    return AnalyzeRequest.model_validate(payload)


def _valid_llm_output() -> str:
    """Return valid mocked LLM JSON."""

    return """
    {
      "feedback_text": "The search interval drops a valid candidate.",
      "hint_text": "Trace the left and right bounds on a single-element input.",
      "error_category": "off_by_one",
      "reasoning_quality": "strong",
      "concept_gaps": ["binary search", "bounds"]
    }
    """


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Patch the orchestrator LLM client for every test."""

    FakeLLMClient.calls = 0
    FakeLLMClient.raw_output = _valid_llm_output()
    FakeLLMClient.exception = None
    monkeypatch.setattr(orchestrator, "LLMClient", FakeLLMClient)
    return FakeLLMClient


@pytest.mark.asyncio
async def test_compilation_error_uses_rule_only_without_llm(fake_llm) -> None:
    """Compilation errors return rule-only feedback without calling the LLM."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "compilation_error",
        "compile_output": "SyntaxError: invalid syntax",
        "test_summary": {
            "total_test_cases": 0,
            "passed_test_cases": 0,
            "failed_test_cases": 0,
        },
        "sample_failed_cases": [],
    }

    response = await analyze_submission(_request(payload))

    assert response.processing_status == "rule_only"
    assert response.error_category == "compilation_error"
    assert response.model_used == "rule_engine_only"
    assert fake_llm.calls == 0


@pytest.mark.asyncio
async def test_wrong_answer_uses_llm_success_path(fake_llm) -> None:
    """Wrong answers call the LLM and return validated output."""

    response = await analyze_submission(_request(VALID_WRONG_ANSWER_PAYLOAD))

    assert response.processing_status == "completed"
    assert response.feedback_text.startswith("The search interval")
    assert response.error_category == "off_by_one"
    assert response.concept_gaps == ["binary search", "bounds"]
    assert fake_llm.calls == 1


@pytest.mark.asyncio
async def test_llm_timeout_returns_timeout_status(fake_llm) -> None:
    """LLM timeouts are converted into fallback responses."""

    fake_llm.exception = LLMTimeoutError("slow")

    response = await analyze_submission(_request(VALID_WRONG_ANSWER_PAYLOAD))

    assert response.processing_status == "timeout"
    assert response.model_used == "none"
    assert fake_llm.calls == 1


@pytest.mark.asyncio
async def test_invalid_llm_output_returns_invalid_status(fake_llm) -> None:
    """Garbage LLM output returns an invalid-output fallback."""

    fake_llm.raw_output = "this is not json"

    response = await analyze_submission(_request(VALID_WRONG_ANSWER_PAYLOAD))

    assert response.processing_status == "llm_output_invalid"
    assert response.model_used != "none"
    assert fake_llm.calls == 1
