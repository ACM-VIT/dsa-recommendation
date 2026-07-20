"""FastAPI dependencies for API routes."""

import secrets
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.settings import get_settings
from app.models.request_schemas import AnalyzeRequest
from app.models.response_schemas import AnalyzeResponse
from app.orchestrator.orchestrator import analyze_submission

AnalyzeCallable = Callable[[AnalyzeRequest], Awaitable[AnalyzeResponse]]

_bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Require a valid bearer token matching the configured service API key."""

    if credentials is None or not secrets.compare_digest(
        credentials.credentials, get_settings().ai_service_api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_analyze_submission() -> AnalyzeCallable:
    """Return the submission analyzer used by the /analyze route."""

    return analyze_submission
