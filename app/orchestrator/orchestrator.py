"""End-to-end analysis orchestration."""

from __future__ import annotations

import time

from app.constants import (
    PROCESSING_STATUS_COMPLETED,
    PROCESSING_STATUS_ERROR,
    PROCESSING_STATUS_LLM_OUTPUT_INVALID,
    PROCESSING_STATUS_RULE_ONLY,
    PROCESSING_STATUS_TIMEOUT,
    PROCESSING_STATUS_UNSAFE_INPUT,
    REASONING_QUALITY_UNKNOWN,
)
from app.llm.base import LLMError, LLMTimeoutError
from app.llm.client import LLMClient, get_active_model_name
from app.logging.logger import bind_submission_id, get_logger, reset_submission_id
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse, ErrorCategory, ProcessingStatus
from app.parser.normalizer import normalize
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from app.security.sanitizer import (
    UnsafeInputError,
    scrub_llm_output_pair,
    validate_input_safety,
)
from app.validator.response_validator import validate_llm_output

logger = get_logger(__name__)


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic time in milliseconds."""

    return max(0, int((time.perf_counter() - started_at) * 1000))


def _normalize_error_category(category: str | None) -> ErrorCategory:
    """Return a response-safe error category."""

    allowed_categories = set(ErrorCategory.__args__)
    if category in allowed_categories:
        return category
    return "unknown"


def _fallback_response(
    request: AnalyzeRequest,
    *,
    started_at: float,
    processing_status: ProcessingStatus,
    model_used: str,
    feedback_text: str | None = None,
    hint_text: str | None = None,
    error_category: str | None = None,
) -> AnalyzeResponse:
    """Build a valid fallback AnalyzeResponse."""

    return AnalyzeResponse(
        submission_id=request.submission_id,
        feedback_text=feedback_text
        or "We could not complete the AI analysis for this submission.",
        hint_text=hint_text or "Please try again, or review the judge output first.",
        error_category=_normalize_error_category(error_category),
        reasoning_quality=REASONING_QUALITY_UNKNOWN,
        concept_gaps=[],
        processing_status=processing_status,
        processing_ms=_elapsed_ms(started_at),
        model_used=model_used,
    )


async def analyze_submission(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a submission through safety, rules, LLM, validation, and scrubbing."""
    started_at = time.perf_counter()
    token = bind_submission_id(request.submission_id)
    try:
        try:
            validate_input_safety(request)
        except UnsafeInputError:
            logger.warning("unsafe input rejected")
            return _fallback_response(
                request,
                started_at=started_at,
                processing_status=PROCESSING_STATUS_UNSAFE_INPUT,
                model_used="none",
                feedback_text="Your submission could not be processed due to invalid input.",
                hint_text="Please check your submission for any unusual characters and try again.",
            )

        try:
            sub = normalize(request)
            rule_outcome = run_rules(sub)
            logger.info(
                "rule engine completed",
                extra={
                    "needs_llm": rule_outcome.needs_llm,
                    "error_category": rule_outcome.error_category,
                },
            )

            if not rule_outcome.needs_llm:
                return _fallback_response(
                    request,
                    started_at=started_at,
                    processing_status=PROCESSING_STATUS_RULE_ONLY,
                    model_used="rule_engine_only",
                    feedback_text=rule_outcome.deterministic_feedback,
                    hint_text=rule_outcome.deterministic_hint,
                    error_category=rule_outcome.error_category,
                )

            prompt = build_prompt(sub, rule_outcome)
            try:
                raw_output = await LLMClient().get_structured_completion(prompt)
            except LLMTimeoutError:
                logger.warning("llm request timed out")
                return _fallback_response(
                    request,
                    started_at=started_at,
                    processing_status=PROCESSING_STATUS_TIMEOUT,
                    model_used="none",
                    hint_text=rule_outcome.deterministic_hint,
                    error_category=rule_outcome.error_category,
                )
            except LLMError:
                logger.exception("llm request failed")
                return _fallback_response(
                    request,
                    started_at=started_at,
                    processing_status=PROCESSING_STATUS_ERROR,
                    model_used="none",
                    hint_text=rule_outcome.deterministic_hint,
                    error_category=rule_outcome.error_category,
                )

            parsed = validate_llm_output(raw_output, request.submission_id, sub.source_code)
            if parsed is None:
                return _fallback_response(
                    request,
                    started_at=started_at,
                    processing_status=PROCESSING_STATUS_LLM_OUTPUT_INVALID,
                    model_used=get_active_model_name(),
                    hint_text=rule_outcome.deterministic_hint,
                    error_category=rule_outcome.error_category,
                )
            feedback_text, hint_text = scrub_llm_output_pair(
                parsed.feedback_text, parsed.hint_text, sub.source_code
            )

            return AnalyzeResponse(
                submission_id=request.submission_id,
                feedback_text=feedback_text,
                hint_text=hint_text,
                error_category=parsed.error_category,
                reasoning_quality=parsed.reasoning_quality,
                concept_gaps=parsed.concept_gaps,
                processing_status=PROCESSING_STATUS_COMPLETED,
                processing_ms=_elapsed_ms(started_at),
                model_used=get_active_model_name(),
            )
        except Exception:
            logger.exception("analysis orchestration failed")
            return _fallback_response(
                request,
                started_at=started_at,
                processing_status=PROCESSING_STATUS_ERROR,
                model_used="none",
            )
    finally:
        reset_submission_id(token)
