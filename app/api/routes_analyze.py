"""HTTP routes for submission analysis."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import AnalyzeCallable, get_analyze_submission, verify_api_key
from app.api.rate_limiter import enforce_rate_limit
from app.logging.logger import bind_submission_id, get_logger, reset_submission_id
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse

logger = get_logger(__name__)

router = APIRouter()

# Small fixed chunk size for a perceptible streaming effect. Not provider-driven —
# the streamed text is already the fully validated/scrubbed AnalyzeResponse content,
# so this only controls delivery granularity, not model behavior.
_STREAM_WORDS_PER_CHUNK = 6


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def analyze(
    request: AnalyzeRequest,
    analyzer: Annotated[AnalyzeCallable, Depends(get_analyze_submission)],
) -> AnalyzeResponse:
    """Analyze a completed code submission."""

    token = bind_submission_id(request.submission_id)
    try:
        return await analyzer(request)
    finally:
        reset_submission_id(token)


def _text_chunks(text: str, words_per_chunk: int = _STREAM_WORDS_PER_CHUNK) -> list[str]:
    """Split text into small word-groups for incremental SSE delivery."""

    words = text.split(" ")
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def _sse_event(event: str, data: dict) -> str:
    """Format one Server-Sent Event with a JSON data payload."""

    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_analysis(
    request: AnalyzeRequest,
    analyzer: AnalyzeCallable,
) -> AsyncIterator[str]:
    """Run the existing analysis pipeline unchanged, then stream the validated result.

    The full request is processed through the same analyzer used by POST /analyze
    (safety checks, rule engine, LLM call, output validation, and leak scrubbing all
    run exactly as they do today) before anything is sent to the client. Only the
    already-validated feedback_text/hint_text are streamed as "chunk" events; the
    final "complete" event carries the identical AnalyzeResponse payload that the
    non-streaming endpoint returns.
    """

    token = bind_submission_id(request.submission_id)
    try:
        try:
            response = await analyzer(request)
        except Exception:
            logger.exception("streaming analysis failed")
            yield _sse_event("error", {"message": "Analysis failed."})
            return

        for field_name in ("feedback_text", "hint_text"):
            for chunk in _text_chunks(getattr(response, field_name)):
                yield _sse_event("chunk", {"field": field_name, "delta": chunk})

        yield _sse_event("complete", response.model_dump())
    finally:
        reset_submission_id(token)


@router.post(
    "/analyze/stream",
    dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)],
)
async def analyze_stream(
    request: AnalyzeRequest,
    analyzer: Annotated[AnalyzeCallable, Depends(get_analyze_submission)],
) -> StreamingResponse:
    """Analyze a completed code submission and stream the validated result via SSE."""

    return StreamingResponse(
        _stream_analysis(request, analyzer),
        media_type="text/event-stream",
    )
