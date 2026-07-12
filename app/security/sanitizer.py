"""Input and output safety helpers."""

from app.config.settings import get_settings
from app.logging.logger import get_logger
from app.models.request_schemas import AnalyzeRequest

logger = get_logger(__name__)

SAFE_LLM_OUTPUT_FALLBACK = (
    "Feedback withheld: response may have contained solution code. Please try again."
)


class UnsafeInputError(ValueError):
    """Raised when a request contains unsafe prompt-breaking input."""


def validate_input_safety(request: AnalyzeRequest) -> None:
    """Validate request input for basic prompt safety concerns."""

    settings = get_settings()
    if len(request.source_code) > settings.max_source_code_chars:
        msg = "source_code exceeds configured maximum length"
        raise UnsafeInputError(msg)

    text_fields = {
        "problem_statement": request.problem_statement,
        "source_code": request.source_code,
        "stdout": request.stdout,
        "stderr": request.stderr,
        "compile_output": request.compile_output,
    }
    for index, failed_case in enumerate(request.sample_failed_cases, start=1):
        text_fields[f"sample_failed_cases[{index}].stdin"] = failed_case.stdin
        text_fields[f"sample_failed_cases[{index}].expected_output"] = (
            failed_case.expected_output
        )
        text_fields[f"sample_failed_cases[{index}].actual_output"] = failed_case.actual_output

    for field_name, value in text_fields.items():
        if "\x00" in value:
            msg = f"{field_name} contains a null byte"
            raise UnsafeInputError(msg)


def scrub_llm_output(text: str, source_code: str) -> str:
    """Scrub LLM-derived text before returning it to the backend."""

    settings = get_settings()
    source_lines = _non_trivial_source_lines(source_code)
    normalized_text = _normalize_whitespace(text)
    matching_lines = [
        line for line in source_lines if line and line in normalized_text
    ]
    if len(matching_lines) > settings.solution_leak_line_threshold:
        logger.warning(
            "llm output withheld because it may contain solution code",
            extra={
                "matched_source_lines": len(matching_lines),
                "threshold": settings.solution_leak_line_threshold,
            },
        )
        return SAFE_LLM_OUTPUT_FALLBACK

    return text


def _non_trivial_source_lines(source_code: str) -> list[str]:
    """Return normalized source lines worth checking for verbatim leakage."""

    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in source_code.splitlines():
        line = _normalize_whitespace(raw_line)
        if len(line) < 6 or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for explainable verbatim-ish line matching."""

    return " ".join(text.strip().split())
