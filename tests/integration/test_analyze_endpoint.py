"""HTTP integration tests for the /analyze endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import rate_limiter
from app.api.deps import get_analyze_submission
from app.config.settings import get_settings
from app.llm.base import LLMTimeoutError
from app.main import app
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse, ErrorResponse
from app.orchestrator import orchestrator
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

AUTH_HEADERS = {"Authorization": f"Bearer {get_settings().ai_service_api_key}"}


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
    rate_limiter.reset()
    yield FakeLLMClient
    app.dependency_overrides.clear()
    rate_limiter.reset()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client authenticated with a valid service API key."""

    return TestClient(app, raise_server_exceptions=False, headers=AUTH_HEADERS)


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


def test_analyze_with_valid_api_key_returns_200(
    client: TestClient,
    fake_llm,
) -> None:
    """A correct Authorization bearer token is accepted and processes normally."""

    response = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 200
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.processing_status == "completed"
    _assert_no_stack_trace(response.text)


def test_analyze_without_api_key_returns_401(fake_llm) -> None:
    """A request with no Authorization header is rejected before processing."""

    unauthenticated_client = TestClient(app, raise_server_exceptions=False)

    response = unauthenticated_client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert fake_llm.calls == 0
    _assert_no_stack_trace(response.text)


def test_analyze_with_invalid_api_key_returns_401(fake_llm) -> None:
    """A request with a wrong bearer token is rejected before processing."""

    unauthenticated_client = TestClient(
        app,
        raise_server_exceptions=False,
        headers={"Authorization": "Bearer wrong-key"},
    )

    response = unauthenticated_client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert fake_llm.calls == 0
    _assert_no_stack_trace(response.text)


def test_health_does_not_require_authentication() -> None:
    """GET /health stays public and does not require an API key."""

    unauthenticated_client = TestClient(app, raise_server_exceptions=False)

    response = unauthenticated_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_429_when_rate_limit_exceeded(
    client: TestClient,
    fake_llm,
    monkeypatch,
) -> None:
    """Requests beyond the configured limit for one identity get HTTP 429."""

    monkeypatch.setattr(get_settings(), "rate_limit_requests", 2)

    first = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)
    second = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)
    third = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "retry-after" in {k.lower() for k in third.headers}
    _assert_no_stack_trace(third.text)


def test_analyze_stream_matches_non_streaming_contract(
    client: TestClient,
    fake_llm,
) -> None:
    """POST /analyze/stream emits chunk events then a complete event with full AnalyzeResponse."""

    response = client.post("/analyze/stream", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_names = [name for name, _ in events]

    assert event_names.count("chunk") > 0
    assert event_names[-1] == "complete"

    complete_payload = json.loads(events[-1][1])
    parsed = AnalyzeResponse.model_validate(complete_payload)
    assert parsed.submission_id == VALID_WRONG_ANSWER_PAYLOAD["submission_id"]
    assert parsed.processing_status == "completed"
    assert parsed.error_category == "edge_case_missing"
    _assert_no_stack_trace(response.text)


def test_analyze_stream_requires_authentication() -> None:
    """POST /analyze/stream enforces the same auth as /analyze."""

    unauthenticated_client = TestClient(app, raise_server_exceptions=False)

    response = unauthenticated_client.post("/analyze/stream", json=VALID_WRONG_ANSWER_PAYLOAD)

    assert response.status_code == 401


def test_analyze_stream_reconstructed_text_matches_non_streaming(
    client: TestClient,
    fake_llm,
) -> None:
    """Chunked feedback/hint text reassembles to the same text /analyze returns."""

    streamed = client.post("/analyze/stream", json=VALID_WRONG_ANSWER_PAYLOAD)
    non_streamed = client.post("/analyze", json=VALID_WRONG_ANSWER_PAYLOAD)

    events = _parse_sse_events(streamed.text)
    feedback_chunks = [
        json.loads(data)["delta"] for name, data in events
        if name == "chunk" and json.loads(data)["field"] == "feedback_text"
    ]
    hint_chunks = [
        json.loads(data)["delta"] for name, data in events
        if name == "chunk" and json.loads(data)["field"] == "hint_text"
    ]

    non_streamed_body = non_streamed.json()
    assert " ".join(feedback_chunks) == non_streamed_body["feedback_text"]
    assert " ".join(hint_chunks) == non_streamed_body["hint_text"]


def _parse_sse_events(raw_text: str) -> list[tuple[str, str]]:
    """Parse `event: X\\ndata: Y\\n\\n` blocks into (event, data) pairs."""

    events: list[tuple[str, str]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_line = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_line = line.removeprefix("data: ")
        if event_name is not None and data_line is not None:
            events.append((event_name, data_line))
    return events
