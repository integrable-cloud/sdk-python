"""The HTTP core: auth, retries, idempotency, timeouts, errors.

Sync and async are the *same* logic, written once. `_prepare` and `_settle`
hold every decision - which headers, whether to retry, how long to wait - and
the two transports differ only in where the ``await`` goes. Writing them
separately is how a retry policy ends up correct in one and subtly wrong in the
other, and nothing fails when it does.

Three behaviours are the reason to use this rather than httpx directly:

**Idempotency keys are automatic.** Every mutating request gets one unless you
supply your own. The key is generated once per logical call and reused across
that call's retries - a key minted inside the retry loop is a different key
each time, which is the mistake the header exists to prevent.

**Retries respect the server.** ``Retry-After`` always beats the backoff curve:
the server knows when the window resets and the client is guessing. Backoff is
jittered, because synchronised retries from many clients are how a recovering
service gets knocked over a second time.

**Only safe things are retried.** A mutation with no idempotency key is never
retried, because "did that land?" is exactly what retrying cannot answer.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

import httpx

from .errors import (
    ConnectionError_,
    IntegrableError,
    TimeoutError_,
    error_from_response,
)

DEFAULT_BASE_URL = "https://api.integrable.cloud"
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_USER_AGENT = "integrable-cloud-python/0.1.0"


class RateLimit:
    """What the rate limiter said on a response."""

    __slots__ = ("limit", "remaining", "reset")

    def __init__(self, limit: int, remaining: int, reset: int) -> None:
        self.limit = limit
        self.remaining = remaining
        self.reset = reset

    def __repr__(self) -> str:
        return f"RateLimit(limit={self.limit}, remaining={self.remaining}, reset={self.reset})"


class ApiResponse:
    """A response, with the metadata worth surfacing alongside the body."""

    __slots__ = (
        "data",
        "status",
        "request_id",
        "api_version",
        "rate_limit",
        "replayed",
        "sunset",
    )

    def __init__(self, response: httpx.Response, data: Any) -> None:
        headers = response.headers
        self.data = data
        self.status = response.status_code
        self.request_id = headers.get("x-request-id")
        self.api_version = headers.get("x-api-version")
        self.rate_limit = _read_rate_limit(headers)
        #: True when the API replayed a stored response for your key.
        self.replayed = headers.get("idempotent-replay") == "true"
        #: Set when this API version has a removal date. Log it.
        self.sunset = headers.get("sunset")


def _read_rate_limit(headers: Mapping[str, str]) -> RateLimit | None:
    limit = headers.get("ratelimit-limit")
    if not limit:
        return None
    try:
        return RateLimit(
            int(limit),
            int(headers.get("ratelimit-remaining", 0)),
            int(headers.get("ratelimit-reset", 0)),
        )
    except ValueError:
        return None


class _Core:
    """Everything that is not the transport."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "An API key is required. Create one at Settings -> API keys, then "
                "pass it as Integrable(api_key=...)."
            )
        if not api_key.startswith("sk_"):
            raise ValueError(
                "That does not look like an Integrable API key - they start with "
                "'sk_live_'. A dashboard session token will not work here."
            )

        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = dict(default_headers or {})
        #: The rate-limit state from the most recent response.
        self.last_rate_limit: RateLimit | None = None

    def _prepare(
        self,
        method: str,
        path: str,
        idempotency_key: str | None,
        headers: Mapping[str, str] | None,
        has_body: bool,
    ) -> tuple[str, dict[str, str]]:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        merged = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "user-agent": _USER_AGENT,
            **self.default_headers,
            **dict(headers or {}),
        }
        if has_body:
            merged["content-type"] = "application/json"
        # Generated once, outside the retry loop, so every attempt of this
        # logical call carries the same key. That is the whole mechanism.
        if method.upper() in _MUTATING:
            merged["idempotency-key"] = idempotency_key or str(uuid.uuid4())
        return url, merged

    def _settle(
        self, response: httpx.Response, attempt: int
    ) -> tuple[ApiResponse | None, IntegrableError | None]:
        rate_limit = _read_rate_limit(response.headers)
        if rate_limit:
            self.last_rate_limit = rate_limit

        if response.is_success:
            data = response.json() if response.content else None
            return ApiResponse(response, data), None

        try:
            body = response.json()
        except ValueError:
            body = None
        return None, error_from_response(
            response.status_code, body, dict(response.headers), attempt
        )

    def _should_retry(self, error: IntegrableError, attempt: int, max_retries: int) -> bool:
        return attempt <= max_retries and error.retryable

    def _backoff(self, attempt: int, error: IntegrableError) -> float:
        """How long to wait before the next attempt.

        ``Retry-After`` wins whenever the server sent one. Otherwise
        exponential with *full* jitter - ``random(0, base * 2**n)`` rather than
        ``base * 2**n`` - because unjittered backoff synchronises every client
        that failed at the same moment into retrying at the same moment.
        """
        raw = error.headers.get("retry-after")
        if raw:
            try:
                return min(float(raw), 60.0)
            except ValueError:
                pass
        ceiling = min(0.5 * float(2 ** (attempt - 1)), 8.0)
        return random.random() * ceiling

    def _translate(self, exc: Exception, timeout: float) -> IntegrableError:
        if isinstance(exc, httpx.TimeoutException):
            return TimeoutError_(f"Request timed out after {timeout}s")
        return ConnectionError_(str(exc) or "Network request failed")
