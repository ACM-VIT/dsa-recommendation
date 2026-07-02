"""Structured logger tests."""

import json
import logging

from app.logging.logger import (
    ContextLoggerAdapter,
    JsonFormatter,
    bind_submission_id,
    reset_submission_id,
)
from main import app


def test_logger_adapter_flattens_extra_fields() -> None:
    """Custom extra fields are promoted to the top-level log payload."""

    token = bind_submission_id("sub_123")
    try:
        adapter = ContextLoggerAdapter(logging.getLogger("test.logger"), {})
        message, kwargs = adapter.process(
            "request completed",
            {"extra": {"request_id": "req_1", "status_code": 200, "duration_ms": 45}},
        )
    finally:
        reset_submission_id(token)

    assert message == "request completed"
    assert kwargs["extra"]["submission_id"] == "sub_123"
    assert kwargs["extra"]["request_id"] == "req_1"
    assert kwargs["extra"]["status_code"] == 200
    assert kwargs["extra"]["duration_ms"] == 45
    assert "extra" not in kwargs["extra"]


def test_logger_adapter_preserves_submission_id_authority() -> None:
    """Context-bound submission_id wins over any value passed in extra."""

    token = bind_submission_id("sub_context")
    try:
        adapter = ContextLoggerAdapter(logging.getLogger("test.logger"), {})
        _, kwargs = adapter.process(
            "request completed",
            {"extra": {"submission_id": "sub_override", "request_id": "req_1"}},
        )
    finally:
        reset_submission_id(token)

    assert kwargs["extra"]["submission_id"] == "sub_context"
    assert kwargs["extra"]["request_id"] == "req_1"


def test_json_formatter_outputs_flattened_fields() -> None:
    """The formatter emits flattened structured fields at the top level."""

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    record.submission_id = "sub_123"
    record.extra = {
        "submission_id": "sub_override",
        "request_id": "req_1",
        "status_code": 200,
        "duration_ms": 45,
    }

    payload = json.loads(formatter.format(record))

    assert payload["submission_id"] == "sub_123"
    assert payload["request_id"] == "req_1"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 45
    assert "extra" not in payload


def test_root_main_exports_fastapi_app() -> None:
    """The compatibility shim continues to expose the FastAPI app."""

    assert app is not None
