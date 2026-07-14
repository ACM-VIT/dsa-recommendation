"""HTTP integration tests for the /analyze endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_analyze_submission
from app.llm.base import LLMTimeoutError
from app.main import app
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse, ErrorResponse
from app.orchestrator import orchestrator
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


class FakeLLMClient:
    """Fake LLM client used to keep HTTP tests away from real providers."""

    calls = 0
    raw_output = ""
    exception: Exception | None = None

    async def get_structured_completion(self, prompt) -> str:  # noqa: ANN001
        """Return configured fake output or raise a configured exception."""

        self.__class__.calls += 1
        if self.exception is not None:
            raise self.exception
        return self.raw_output


def _valid_llm_output() -> str:
    """Return valid mocked LLM JSON."""

    return """
    {
      "feedback_text": "The subtractive Roman numeral cases are not handled.",
      "hint_text": "Add checks for 4, 9, 40, 90, 400, and 900 before greedy symbols.",
      "error_category": "edge_case_missing",
      "reasoning_quality": "partial",
      "concept_gaps": ["edge cases", "greedy algorithms"]
    }
    """


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    """Patch the orchestrator LLM client for every HTTP test."""

    FakeLLMClient.calls = 0
    FakeLLMClient.raw_output = _valid_llm_output()
    FakeLLMClient.exception = None
    monkeypatch.setattr(orchestrator, "LLMClient", FakeLLMClient)
    app.dependency_overrides.clear()
    yield FakeLLMClient
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""

    return TestClient(app, raise_server_exceptions=False)


def _assert_no_stack_trace(response_text: str) -> None:
    """Assert responses do not expose internals."""

    assert "Traceback" not in response_text
    assert "Exception" not in response_text


def test_analyze_valid_payload_returns_completed_response(
    client: TestClient,
    fake_llm,
) -> None:
    """POST /analyze returns a schema-valid completed response."""

    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 200
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.submission_id == VALID_WRONG_ANSWER_PAYLOAD["submission_id"]
    assert parsed.processing_status == "completed"
    assert parsed.error_category == "edge_case_missing"
    assert fake_llm.calls == 1
    _assert_no_stack_trace(response.text)


def test_analyze_compilation_error_is_rule_only_without_llm(
    client: TestClient,
    fake_llm,
) -> None:
    """Compilation errors short-circuit to deterministic rule-only feedback."""

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

    response = client.post("/analyze", json=payload)

    assert response.status_code == 200
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.processing_status == "rule_only"
    assert parsed.error_category == "compilation_error"
    assert parsed.model_used == "rule_engine_only"
    assert fake_llm.calls == 0
    _assert_no_stack_trace(response.text)


def test_analyze_malformed_payload_returns_error_response(client: TestClient) -> None:
    """Malformed request payloads return the stable ErrorResponse envelope."""

    payload = VALID_WRONG_ANSWER_PAYLOAD.copy()
    payload.pop("submission_id")

    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    parsed = ErrorResponse.model_validate(response.json())
    assert parsed.error_code == "validation_error"
    assert parsed.message == "Request validation failed."
    _assert_no_stack_trace(response.text)


def test_analyze_oversized_source_code_returns_error_response(client: TestClient) -> None:
    """Oversized source_code is rejected by request validation."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {"source_code": "x" * 20001}

    response = client.post("/analyze", json=payload)

    assert response.status_code == 422
    parsed = ErrorResponse.model_validate(response.json())
    assert parsed.error_code == "validation_error"
    _assert_no_stack_trace(response.text)


def test_analyze_llm_timeout_returns_200_timeout(
    client: TestClient,
    fake_llm,
) -> None:
    """LLM timeouts are represented as response status, not HTTP 500."""

    fake_llm.exception = LLMTimeoutError("slow")

    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 200
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.processing_status == "timeout"
    assert fake_llm.calls == 1
    _assert_no_stack_trace(response.text)


def test_analyze_invalid_llm_output_returns_200_invalid_status(
    client: TestClient,
    fake_llm,
) -> None:
    """Garbage LLM text is represented as llm_output_invalid."""

    fake_llm.raw_output = "not json"

    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 200
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.processing_status == "llm_output_invalid"
    assert fake_llm.calls == 1
    _assert_no_stack_trace(response.text)


def test_health_returns_ok_from_same_app(client: TestClient) -> None:
    """GET /health remains available after API wiring."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    _assert_no_stack_trace(response.text)


def test_unhandled_route_exception_returns_sanitized_500(client: TestClient) -> None:
    """Unexpected route failures use ErrorResponse without stack traces."""

    async def broken_analyzer(request: AnalyzeRequest) -> AnalyzeResponse:
        raise RuntimeError("secret internal failure")

    app.dependency_overrides[get_analyze_submission] = (
        lambda: broken_analyzer
    )

    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 500
    parsed = ErrorResponse.model_validate(response.json())
    assert parsed.error_code == "internal_error"
    assert parsed.message == "An internal error occurred."
    assert "secret internal failure" not in response.text
    _assert_no_stack_trace(response.text)
