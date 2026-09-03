"""The error taxonomy, mapped from what the API actually sends.

Every failure body carries two envelopes at once: a nested ``error`` object,
and the RFC 9457 members (``type``, ``title``, ``status``, ``detail``,
``instance``). Both are read, because either can be the more specific one -
a route's own validation error fills ``error`` richly, while a gateway between
you and us may only manage the RFC members.

A distinct class per condition rather than one class carrying a status code,
because the interesting question in an ``except`` is almost never "what number"
but "can I retry this, and if not, whose fault is it". A class hierarchy
answers that; ``if e.status == 429`` makes every caller re-derive it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class IntegrableError(Exception):
    """Base class for everything this SDK raises."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.body = dict(body or {})
        self.headers = dict(headers or {})
        #: Total attempts made, including the one that failed.
        self.attempts = attempts

        error = self.body.get("error") or {}
        #: Machine-readable code, e.g. ``not_found``, ``idempotency_key_reused``.
        self.code: str | None = error.get("code")
        #: Quote this at support - it identifies the exact call.
        self.request_id: str | None = error.get("request_id") or self.headers.get(
            "x-request-id"
        )
        #: Structured detail, where the endpoint provides it.
        self.details: dict[str, Any] | None = error.get("details")

    @property
    def retryable(self) -> bool:
        """Whether retrying this exact request could plausibly succeed.

        The client already retries these; this is for a caller doing its own
        queueing on top.
        """
        return False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status={self.status}, code={self.code!r}, "
            f"request_id={self.request_id!r}, message={self.message!r})"
        )


class ConnectionError_(IntegrableError):
    """The request never reached the API: DNS, TLS, connection reset, offline."""

    @property
    def retryable(self) -> bool:
        return True


class TimeoutError_(IntegrableError):
    """The request exceeded the configured timeout."""

    @property
    def retryable(self) -> bool:
        return True


class AuthenticationError(IntegrableError):
    """401 - the key is missing, malformed, expired or revoked."""


class PermissionError_(IntegrableError):
    """403 - authenticated, but the key's scopes do not permit this."""


class NotFoundError(IntegrableError):
    """404 - no such resource, or it belongs to another workspace."""


class ConflictError(IntegrableError):
    """409 - including an idempotent request whose original is still running."""

    @property
    def retryable(self) -> bool:
        # Only the in-progress case. A name collision will still collide.
        return self.code == "idempotency_in_progress"


class ValidationError(IntegrableError):
    """422 - the request body failed validation. ``details`` names the fields."""


class QuotaExceededError(IntegrableError):
    """402 - a plan limit was reached. ``details`` carries the metric and limit."""


class PreconditionRequiredError(IntegrableError):
    """428 - a precondition is unmet, e.g. an unconfirmed email address."""


class RateLimitError(IntegrableError):
    """429 - rate limited."""

    @property
    def retry_after(self) -> float | None:
        """Seconds to wait, from the response header."""
        raw = self.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @property
    def retryable(self) -> bool:
        return True


class ServerError(IntegrableError):
    """5xx - something failed on our side."""

    @property
    def retryable(self) -> bool:
        return True


_BY_STATUS: dict[int, type[IntegrableError]] = {
    401: AuthenticationError,
    402: QuotaExceededError,
    403: PermissionError_,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    428: PreconditionRequiredError,
    429: RateLimitError,
}


def error_from_response(
    status: int,
    body: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    attempts: int,
) -> IntegrableError:
    """Builds the right class from a response.

    The message prefers the API's own wording over anything invented here: a
    generic "Request failed with status 422" replaces a sentence that said
    exactly which field was wrong, and that sentence is the reason anyone reads
    a traceback.
    """
    cls = _BY_STATUS.get(status) or (ServerError if status >= 500 else IntegrableError)
    payload = dict(body or {})
    message = (
        (payload.get("error") or {}).get("message")
        or payload.get("detail")
        or payload.get("title")
        or f"Request failed with status {status}"
    )
    return cls(message, status=status, body=payload, headers=headers, attempts=attempts)
