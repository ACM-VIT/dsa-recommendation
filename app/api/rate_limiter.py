"""Lightweight in-memory rate limiting for HTTP routes.

Fixed-window counter keyed by client identity (authenticated bearer token when
present, otherwise client IP). Correct for a single process; a multi-instance
deployment would need a shared store (e.g. Redis) for a globally consistent
limit — out of scope here, matching the "lightweight" requirement for this pass.

FastAPI runs synchronous dependencies (like enforce_rate_limit) in a thread
pool, not on the asyncio event loop, so concurrent requests can genuinely
execute this function on different threads at the same time. The read-check-
increment-write sequence below is therefore guarded by a lock so the counter
update is atomic across threads.
"""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, status

from app.config.settings import get_settings

# {identity: (window_start_monotonic, request_count)}
_windows: dict[str, tuple[float, int]] = {}
_lock = threading.Lock()


def _client_identity(request: Request) -> str:
    """Return the authenticated bearer token, falling back to client IP."""

    auth_header = request.headers.get("authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
        return auth_header

    client = request.client
    return client.host if client else "unknown"


def reset() -> None:
    """Clear all rate limit window state. Intended for test isolation."""

    with _lock:
        _windows.clear()


def enforce_rate_limit(request: Request) -> None:
    """Raise HTTP 429 if the caller has exceeded the configured request rate."""

    settings = get_settings()
    if not settings.rate_limit_enabled:
        return

    identity = _client_identity(request)
    now = time.monotonic()

    with _lock:
        window_start, count = _windows.get(identity, (now, 0))

        if now - window_start >= settings.rate_limit_window_seconds:
            window_start, count = now, 0

        count += 1
        _windows[identity] = (window_start, count)

    if count > settings.rate_limit_requests:
        retry_after = max(0.0, settings.rate_limit_window_seconds - (now - window_start))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
