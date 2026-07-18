"""FastAPI dependencies for API routes."""

from collections.abc import Awaitable, Callable

from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse
from app.orchestrator.orchestrator import analyze_submission

AnalyzeCallable = Callable[[AnalyzeRequest], Awaitable[AnalyzeResponse]]


def get_analyze_submission() -> AnalyzeCallable:
    """Return the submission analyzer used by the /analyze route."""

    print("Controller reached")
    return analyze_submission
