"""Unit tests for the in-memory rate limiter."""

import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from app.api import rate_limiter


def _make_request(
    *, authorization: str | None = None, client_host: str | None = "1.2.3.4"
) -> Request:
    """Build a minimal ASGI Request carrying just the headers/client this module reads."""

    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))

    scope = {
        "type": "http",
        "headers": headers,
        "client": (client_host, 12345) if client_host is not None else None,
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def _reset_state():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def _settings(*, enabled: bool = True, requests: int = 2, window: float = 60.0) -> SimpleNamespace:
    return SimpleNamespace(
        rate_limit_enabled=enabled,
        rate_limit_requests=requests,
        rate_limit_window_seconds=window,
    )


def test_identity_uses_bearer_token_when_present() -> None:
    """The bearer token is the primary rate-limit identity."""

    request = _make_request(authorization="Bearer secret-token", client_host="9.9.9.9")

    assert rate_limiter._client_identity(request) == "secret-token"


def test_identity_falls_back_to_client_ip_when_no_auth_header() -> None:
    """With no Authorization header at all, the client IP is used."""

    request = _make_request(authorization=None, client_host="9.9.9.9")

    assert rate_limiter._client_identity(request) == "9.9.9.9"


def test_identity_falls_back_to_unknown_when_no_client_info() -> None:
    """With neither auth header nor client info, a stable fallback is used."""

    request = _make_request(authorization=None, client_host=None)

    assert rate_limiter._client_identity(request) == "unknown"


def test_enforce_rate_limit_allows_requests_within_limit(monkeypatch) -> None:
    """Requests at or under the configured limit pass without raising."""

    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(requests=2))
    request = _make_request(authorization="Bearer token-a")

    rate_limiter.enforce_rate_limit(request)
    rate_limiter.enforce_rate_limit(request)  # still within limit


def test_enforce_rate_limit_raises_429_when_exceeded(monkeypatch) -> None:
    """The request that exceeds the configured limit raises HTTP 429."""

    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(requests=2))
    request = _make_request(authorization="Bearer token-b")

    rate_limiter.enforce_rate_limit(request)
    rate_limiter.enforce_rate_limit(request)
    with pytest.raises(HTTPException) as exc_info:
        rate_limiter.enforce_rate_limit(request)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


def test_enforce_rate_limit_scopes_by_identity(monkeypatch) -> None:
    """Different bearer tokens get independent buckets."""

    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(requests=1))
    request_a = _make_request(authorization="Bearer token-a")
    request_b = _make_request(authorization="Bearer token-b")

    rate_limiter.enforce_rate_limit(request_a)
    # A different identity's first request is unaffected by token-a's usage.
    rate_limiter.enforce_rate_limit(request_b)

    with pytest.raises(HTTPException):
        rate_limiter.enforce_rate_limit(request_a)


def test_enforce_rate_limit_resets_after_window_elapses(monkeypatch) -> None:
    """A new window resets the counter for that identity."""

    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(requests=1, window=0.01))
    request = _make_request(authorization="Bearer token-c")

    rate_limiter.enforce_rate_limit(request)
    with pytest.raises(HTTPException):
        rate_limiter.enforce_rate_limit(request)

    import time

    time.sleep(0.02)
    rate_limiter.enforce_rate_limit(request)  # new window, allowed again


def test_enforce_rate_limit_disabled_never_raises(monkeypatch) -> None:
    """When rate_limit_enabled is False, no limit is applied."""

    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(enabled=False, requests=1))
    request = _make_request(authorization="Bearer token-d")

    for _ in range(5):
        rate_limiter.enforce_rate_limit(request)


def test_enforce_rate_limit_is_thread_safe_under_concurrent_requests(monkeypatch) -> None:
    """Concurrent requests for the same identity cannot exceed the configured limit.

    FastAPI runs sync dependencies like enforce_rate_limit in a thread pool, so
    real concurrent execution (not just concurrent asyncio tasks on one loop) is
    the scenario that must be safe. This drives many real OS threads through the
    same critical section at once, with the GIL's switch interval lowered to
    maximize the chance of interleaving, and asserts the count admitted never
    exceeds the limit -- which only holds if the read-check-increment-write
    sequence is properly synchronized.
    """

    limit = 20
    thread_count = 100
    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(requests=limit))
    request = _make_request(authorization="Bearer race-token")

    barrier = threading.Barrier(thread_count)
    allowed = 0
    rejected = 0
    counts_lock = threading.Lock()

    def worker() -> None:
        nonlocal allowed, rejected
        barrier.wait()  # maximize the number of threads hitting the critical section at once
        try:
            rate_limiter.enforce_rate_limit(request)
            outcome_allowed = True
        except HTTPException:
            outcome_allowed = False
        with counts_lock:
            if outcome_allowed:
                allowed += 1
            else:
                rejected += 1

    original_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.0001)
    try:
        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(original_switch_interval)

    assert allowed == limit
    assert rejected == thread_count - limit
    assert allowed + rejected == thread_count


def test_enforce_rate_limit_serializes_on_shared_lock(monkeypatch) -> None:
    """A concurrent call blocks while another thread holds the counter's lock.

    This deterministically proves the read-check-increment-write section is
    mutually exclusive, rather than relying on scheduling luck to trigger an
    interleaving (the GIL can make races on such a short critical section rare
    to observe naturally even when the code is unsafe).
    """

    monkeypatch.setattr(rate_limiter, "get_settings", lambda: _settings(requests=100))
    request = _make_request(authorization="Bearer lock-token")

    started = threading.Event()
    finished = threading.Event()

    def worker() -> None:
        started.set()
        rate_limiter.enforce_rate_limit(request)
        finished.set()

    with rate_limiter._lock:
        t = threading.Thread(target=worker)
        t.start()
        assert started.wait(timeout=1), "worker thread never started"
        # The worker must still be blocked waiting for the lock we're holding.
        assert not finished.wait(timeout=0.2), "worker completed while lock was held"

    # Lock released -- the worker should now be able to acquire it and finish.
    t.join(timeout=1)
    assert finished.is_set()
