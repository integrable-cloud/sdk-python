"""The behaviour that makes this worth using instead of httpx directly.

All offline, against a mocked transport, so the suite is fast and needs no
credentials. The properties being proven are the ones that stay invisible until
they matter: that a retry reuses the *same* idempotency key, that a mutation
without one is never retried, and that ``Retry-After`` beats the backoff curve.

Both clients are exercised. Sync and async share `_Core`, and this is what
holds them to it - a policy correct in one and subtly wrong in the other is the
exact failure that sharing was meant to prevent.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from integrable_cloud import (
    AsyncIntegrable,
    AuthenticationError,
    ConflictError,
    Integrable,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    ValidationError,
)

KEY = "sk_live_test_key"
BASE = "https://api.integrable.cloud"


def client(**kw):
    return Integrable(KEY, max_retries=2, **kw)


# -- construction -------------------------------------------------------------


def test_an_empty_key_is_refused_with_an_actionable_message():
    with pytest.raises(ValueError, match="API key is required"):
        Integrable("")


def test_a_session_token_is_refused():
    """The mistake people actually make: pasting a dashboard JWT."""
    with pytest.raises(ValueError, match="sk_live_"):
        Integrable("eyJhbGciOiJIUzI1NiJ9.abc")


# -- authentication -----------------------------------------------------------


@respx.mock
def test_the_key_is_sent_as_a_bearer_token():
    route = respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    client().bots.list()
    assert route.calls[0].request.headers["authorization"] == f"Bearer {KEY}"


# -- idempotency --------------------------------------------------------------


@respx.mock
def test_every_mutation_carries_a_key():
    route = respx.post(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(201, json={"id": "b1"})
    )
    client().bots.create(name="x")
    assert route.calls[0].request.headers.get("idempotency-key")


@respx.mock
def test_a_read_does_not():
    """A key on a GET means a row written per read - write amplification
    proportional to read traffic, for a method that is already idempotent."""
    route = respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    client().bots.list()
    assert "idempotency-key" not in route.calls[0].request.headers


@respx.mock
def test_retries_of_one_call_reuse_the_same_key():
    """The detail everybody gets wrong.

    A key generated inside the retry loop is a different key each time, so the
    server sees three unrelated requests and creates three bots - precisely
    what the header exists to prevent.
    """
    route = respx.post(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(503, json={"detail": "upstream"}),
            httpx.Response(503, json={"detail": "upstream"}),
            httpx.Response(201, json={"id": "b1"}),
        ]
    )
    client().bots.create(name="x")

    keys = {call.request.headers["idempotency-key"] for call in route.calls}
    assert len(route.calls) == 3
    assert len(keys) == 1


@respx.mock
def test_a_caller_supplied_key_wins():
    route = respx.post(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(201, json={"id": "b1"})
    )
    client().post("/api/bots", json={"name": "x"}, idempotency_key="my-own-key")
    assert route.calls[0].request.headers["idempotency-key"] == "my-own-key"


@respx.mock
def test_a_replay_is_reported_so_a_caller_can_tell_it_from_a_fresh_create():
    respx.post(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(
            201, json={"id": "b1"}, headers={"idempotent-replay": "true"}
        )
    )
    response = client().post("/api/bots", json={"name": "x"})
    assert response.replayed is True


# -- retries ------------------------------------------------------------------


@respx.mock
def test_a_5xx_is_retried_and_succeeds():
    route = respx.get(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(500, json={"detail": "boom"}),
            httpx.Response(200, json={"items": []}),
        ]
    )
    client().bots.list()
    assert len(route.calls) == 2


@respx.mock
def test_a_4xx_that_will_never_succeed_is_not_retried():
    route = respx.get(f"{BASE}/api/bots/nope").mock(
        return_value=httpx.Response(404, json={"detail": "Bot not found"})
    )
    with pytest.raises(NotFoundError):
        client().bots.get("nope")
    assert len(route.calls) == 1


@respx.mock
def test_it_gives_up_after_max_retries_and_reports_the_attempt_count():
    route = respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(503, json={"detail": "still down"})
    )
    with pytest.raises(ServerError) as caught:
        client().bots.list()
    assert len(route.calls) == 3  # 1 + max_retries
    assert caught.value.attempts == 3


def test_retry_after_beats_the_backoff_curve():
    """The server knows when the window resets; the client is guessing.

    Asserted against the computation rather than the wall clock. A timing
    assertion here measures httpx and respx setup as much as it measures the
    sleep, which makes it flaky on a loaded machine and, worse, quietly
    meaningless when it passes for the wrong reason.
    """
    from integrable_cloud._client import _Core

    core = _Core(KEY)

    honoured = RateLimitError("slow", status=429, headers={"retry-after": "0"})
    assert core._backoff(attempt=1, error=honoured) == 0.0

    # Ten seconds is far beyond anything the jittered curve produces at
    # attempt 1 (ceiling 0.5s), so this can only have come from the header.
    patient = RateLimitError("slow", status=429, headers={"retry-after": "10"})
    assert core._backoff(attempt=1, error=patient) == 10.0

    # And a runaway value is clamped rather than parking the process for an
    # hour on a server's say-so.
    absurd = RateLimitError("slow", status=429, headers={"retry-after": "86400"})
    assert core._backoff(attempt=1, error=absurd) == 60.0


def test_backoff_without_a_header_is_bounded_and_jittered():
    """Full jitter - random(0, ceiling) - not a fixed curve.

    Unjittered backoff synchronises every client that failed at the same
    moment into retrying at the same moment, which is how a service that was
    recovering gets knocked over a second time.
    """
    from integrable_cloud._client import _Core

    core = _Core(KEY)
    error = ServerError("boom", status=503)

    samples = [core._backoff(attempt=3, error=error) for _ in range(200)]
    assert all(0.0 <= s <= 2.0 for s in samples)  # ceiling at attempt 3
    assert len(set(samples)) > 1, "not jittered - every wait was identical"


@respx.mock
def test_a_429_is_retried_using_that_header():
    route = respx.get(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(429, json={"detail": "slow"}, headers={"retry-after": "0"}),
            httpx.Response(200, json={"items": []}),
        ]
    )
    client().bots.list()
    assert len(route.calls) == 2


@respx.mock
def test_retries_can_be_turned_off():
    route = respx.get(f"{BASE}/api/bots").mock(return_value=httpx.Response(500, json={}))
    with pytest.raises(ServerError):
        Integrable(KEY, max_retries=0).bots.list()
    assert len(route.calls) == 1


# -- errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthenticationError),
        (402, QuotaExceededError),
        (404, NotFoundError),
        (409, ConflictError),
        (422, ValidationError),
        (429, RateLimitError),
    ],
)
@respx.mock
def test_status_maps_to_a_class_you_can_branch_on(status, expected):
    respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(status, json={"error": {"code": "x"}})
    )
    with pytest.raises(expected):
        Integrable(KEY, max_retries=0).bots.list()


@respx.mock
def test_the_apis_own_message_survives():
    respx.post(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {"code": "validation_error", "message": "name must not be empty"},
                "detail": "name must not be empty",
            },
        )
    )
    with pytest.raises(ValidationError) as caught:
        client().bots.create()
    assert caught.value.message == "name must not be empty"
    assert caught.value.code == "validation_error"


@respx.mock
def test_the_request_id_is_surfaced_for_a_support_ticket():
    respx.get(f"{BASE}/api/bots/x").mock(
        return_value=httpx.Response(
            404, json={"error": {"code": "not_found", "request_id": "req_abc"}}
        )
    )
    with pytest.raises(NotFoundError) as caught:
        client().bots.get("x")
    assert caught.value.request_id == "req_abc"


def test_only_an_in_progress_conflict_is_retryable():
    in_progress = ConflictError(
        "x", status=409, body={"error": {"code": "idempotency_in_progress"}}
    )
    name_taken = ConflictError("x", status=409, body={"error": {"code": "conflict"}})
    assert in_progress.retryable is True
    assert name_taken.retryable is False


@respx.mock
def test_rate_limit_error_exposes_retry_after():
    respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(429, json={}, headers={"retry-after": "12"})
    )
    with pytest.raises(RateLimitError) as caught:
        Integrable(KEY, max_retries=0).bots.list()
    assert caught.value.retry_after == 12.0


# -- rate limit ---------------------------------------------------------------


@respx.mock
def test_the_budget_from_the_last_response_is_readable():
    respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(
            200,
            json={"items": []},
            headers={
                "ratelimit-limit": "120",
                "ratelimit-remaining": "7",
                "ratelimit-reset": "42",
            },
        )
    )
    c = client()
    c.bots.list()
    assert (c.rate_limit.limit, c.rate_limit.remaining, c.rate_limit.reset) == (120, 7, 42)


# -- pagination ---------------------------------------------------------------


@respx.mock
def test_the_cursor_is_followed_to_the_end():
    respx.get(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "items": [{"id": "1"}, {"id": "2"}],
                    "next_cursor": "c1",
                    "has_more": True,
                },
            ),
            httpx.Response(
                200, json={"items": [{"id": "3"}], "next_cursor": None, "has_more": False}
            ),
        ]
    )
    assert [b["id"] for b in client().bots.walk()] == ["1", "2", "3"]


@respx.mock
def test_a_repeated_cursor_stops_rather_than_looping_forever():
    """A server bug that would otherwise hammer the API indefinitely."""
    respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(
            200, json={"items": [{"id": "1"}], "next_cursor": "same", "has_more": True}
        )
    )
    assert len(list(client().bots.walk())) == 2


@respx.mock
def test_a_none_query_value_is_dropped_not_sent_empty():
    """`?cursor=` and no cursor mean different things to a paginated endpoint;
    sending the first when you meant the second returns page one forever."""
    route = respx.get(f"{BASE}/api/bots").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    list(client().bots.walk())
    assert "cursor" not in route.calls[0].request.url.params


# -- async mirror -------------------------------------------------------------


@respx.mock
async def test_the_async_client_shares_the_retry_policy():
    route = respx.get(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(503, json={"detail": "upstream"}),
            httpx.Response(200, json={"items": []}),
        ]
    )
    async with AsyncIntegrable(KEY, max_retries=2) as c:
        await c.bots.list()
    assert len(route.calls) == 2


@respx.mock
async def test_the_async_client_shares_the_idempotency_rule():
    route = respx.post(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(201, json={"id": "b1"}),
        ]
    )
    async with AsyncIntegrable(KEY, max_retries=2) as c:
        await c.bots.create(name="x")

    keys = {call.request.headers["idempotency-key"] for call in route.calls}
    assert len(keys) == 1, "the async retry minted a fresh key"


@respx.mock
async def test_the_async_client_paginates():
    respx.get(f"{BASE}/api/bots").mock(
        side_effect=[
            httpx.Response(
                200, json={"items": [{"id": "1"}], "next_cursor": "c1", "has_more": True}
            ),
            httpx.Response(200, json={"items": [{"id": "2"}], "has_more": False}),
        ]
    )
    async with AsyncIntegrable(KEY) as c:
        seen = [b["id"] async for b in c.bots.walk()]
    assert seen == ["1", "2"]
