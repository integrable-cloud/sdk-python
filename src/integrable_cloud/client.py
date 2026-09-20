"""The sync and async clients, and the typed resources hanging off them.

    from integrable_cloud import Integrable

    client = Integrable(api_key=os.environ["INTEGRABLE_API_KEY"])
    for agent in client.agents.walk():
        print(agent["id"], agent["name"])

The async client is the same surface with ``await`` and ``async for``:

    from integrable_cloud import AsyncIntegrable

    async with AsyncIntegrable(api_key=...) as client:
        page = await client.agents.list()

Both share `_Core`, so the retry policy, the idempotency rule and the error
mapping are written once and cannot diverge between them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, cast

import httpx

from ._client import DEFAULT_BASE_URL, ApiResponse, RateLimit, _Core
from .errors import IntegrableError

__all__ = ["Integrable", "AsyncIntegrable", "ApiResponse", "RateLimit", "DEFAULT_BASE_URL"]

Query = Mapping[str, Any]


def _clean(query: Query | None) -> dict[str, Any]:
    """Drops ``None`` rather than sending it as an empty string.

    ``?cursor=`` and no cursor at all mean different things to a paginated
    endpoint, and sending the first when you meant the second silently returns
    page one forever.
    """
    return {k: v for k, v in (query or {}).items() if v is not None}


class Integrable:
    """Synchronous client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._core = _Core(
            api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._http = http_client or httpx.Client(timeout=timeout)
        self._owns_http = http_client is None

        self.agents = _Agents(self)
        self.conversations = _Conversations(self)
        self.knowledge = _Knowledge(self)
        self.analytics = _Analytics(self)
        self.webhooks = _Webhooks(self)

    @property
    def rate_limit(self) -> RateLimit | None:
        """What the rate limiter said on the most recent response."""
        return self._core.last_rate_limit

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        query: Query | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ApiResponse:
        url, hdrs = self._core._prepare(
            method, path, idempotency_key, headers, json is not None
        )
        retries = self._core.max_retries if max_retries is None else max_retries
        wait = self._core.timeout if timeout is None else timeout
        last: IntegrableError | None = None

        for attempt in range(1, retries + 2):
            try:
                response = self._http.request(
                    method, url, headers=hdrs, json=json, params=_clean(query), timeout=wait
                )
                ok, err = self._core._settle(response, attempt)
                if ok is not None:
                    return ok
                last = err
            except Exception as exc:  # noqa: BLE001
                last = self._core._translate(exc, wait)

            assert last is not None
            if not self._core._should_retry(last, attempt, retries):
                break
            time.sleep(self._core._backoff(attempt, last))

        raise last if last else IntegrableError("Request failed for an unknown reason")

    def get(self, path: str, **kw: Any) -> ApiResponse:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> ApiResponse:
        return self.request("POST", path, **kw)

    def patch(self, path: str, **kw: Any) -> ApiResponse:
        return self.request("PATCH", path, **kw)

    def delete(self, path: str, **kw: Any) -> ApiResponse:
        return self.request("DELETE", path, **kw)

    def paginate(self, path: str, query: Query | None = None) -> Iterator[dict[str, Any]]:
        """Walks every page of a list endpoint, lazily.

        Follow the cursor; never increment a page number against this API.
        Offset pagination makes Postgres read and discard every skipped row, so
        page 40 costs forty times page 1 and eventually times out.
        """
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = self.get(path, query={**_clean(query), "cursor": cursor}).data or {}
            yield from page.get("items") or []

            nxt = page.get("next_cursor")
            # Guards a server that returns the same cursor with has_more
            # forever - otherwise an infinite loop hammering the API.
            if not page.get("has_more") or not nxt or nxt in seen:
                return
            seen.add(nxt)
            cursor = nxt

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Integrable:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncIntegrable:
    """Asynchronous client. Same surface, same policy, ``await``-ed."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._core = _Core(
            api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._http = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_http = http_client is None

        self.agents = _AsyncAgents(self)
        self.conversations = _AsyncConversations(self)
        self.knowledge = _AsyncKnowledge(self)
        self.analytics = _AsyncAnalytics(self)
        self.webhooks = _AsyncWebhooks(self)

    @property
    def rate_limit(self) -> RateLimit | None:
        return self._core.last_rate_limit

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        query: Query | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> ApiResponse:
        url, hdrs = self._core._prepare(
            method, path, idempotency_key, headers, json is not None
        )
        retries = self._core.max_retries if max_retries is None else max_retries
        wait = self._core.timeout if timeout is None else timeout
        last: IntegrableError | None = None

        for attempt in range(1, retries + 2):
            try:
                response = await self._http.request(
                    method, url, headers=hdrs, json=json, params=_clean(query), timeout=wait
                )
                ok, err = self._core._settle(response, attempt)
                if ok is not None:
                    return ok
                last = err
            except Exception as exc:  # noqa: BLE001
                last = self._core._translate(exc, wait)

            assert last is not None
            if not self._core._should_retry(last, attempt, retries):
                break
            await asyncio.sleep(self._core._backoff(attempt, last))

        raise last if last else IntegrableError("Request failed for an unknown reason")

    async def get(self, path: str, **kw: Any) -> ApiResponse:
        return await self.request("GET", path, **kw)

    async def post(self, path: str, **kw: Any) -> ApiResponse:
        return await self.request("POST", path, **kw)

    async def patch(self, path: str, **kw: Any) -> ApiResponse:
        return await self.request("PATCH", path, **kw)

    async def delete(self, path: str, **kw: Any) -> ApiResponse:
        return await self.request("DELETE", path, **kw)

    async def paginate(
        self, path: str, query: Query | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            response = await self.get(path, query={**_clean(query), "cursor": cursor})
            page = response.data or {}
            for item in page.get("items") or []:
                yield item

            nxt = page.get("next_cursor")
            if not page.get("has_more") or not nxt or nxt in seen:
                return
            seen.add(nxt)
            cursor = nxt

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncIntegrable:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# -- resources ----------------------------------------------------------------
#
# Thin by design. The interesting logic is all in the client above; these exist
# so an editor can complete `client.knowledge.add_text(` instead of leaving
# everyone to remember a URL template and a discriminator field.


class _Agents:
    def __init__(self, client: Integrable) -> None:
        self._c = client

    def list(self, **query: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", self._c.get("/api/agents", query=query).data)

    def walk(self, **query: Any) -> Iterator[dict[str, Any]]:
        return self._c.paginate("/api/agents", query)

    def get(self, agent_id: str) -> dict[str, Any]:
        return cast("dict[str, Any]", self._c.get(f"/api/agents/{agent_id}").data)

    def create(self, **body: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", self._c.post("/api/agents", json=body).data)

    def update(self, agent_id: str, **body: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", self._c.patch(f"/api/agents/{agent_id}", json=body).data)

    def delete(self, agent_id: str) -> None:
        self._c.delete(f"/api/agents/{agent_id}")

    def embed(self, agent_id: str) -> dict[str, Any]:
        """The embed snippet to paste into a site, with its integrity hash."""
        return cast("dict[str, Any]", self._c.get(f"/api/agents/{agent_id}/embed").data)


class _Conversations:
    def __init__(self, client: Integrable) -> None:
        self._c = client

    def list(self, agent_id: str, **query: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", self._c.get(f"/api/agents/{agent_id}/conversations", query=query).data
        )

    def walk(self, agent_id: str, **query: Any) -> Iterator[dict[str, Any]]:
        """Every matching conversation. The one to use for an export or a sync."""
        return self._c.paginate(f"/api/agents/{agent_id}/conversations", query)

    def get(self, agent_id: str, conversation_id: str) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._c.get(f"/api/agents/{agent_id}/conversations/{conversation_id}").data,
        )


class _Knowledge:
    def __init__(self, client: Integrable) -> None:
        self._c = client

    def list(self, agent_id: str, **query: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", self._c.get(f"/api/agents/{agent_id}/knowledge", query=query).data
        )

    def walk(self, agent_id: str, **query: Any) -> Iterator[dict[str, Any]]:
        return self._c.paginate(f"/api/agents/{agent_id}/knowledge", query)

    def create(self, agent_id: str, **body: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", self._c.post(f"/api/agents/{agent_id}/knowledge", json=body).data
        )

    def add_text(self, agent_id: str, title: str, text: str) -> dict[str, Any]:
        """Teach an agent something, without assembling the discriminator.

        Indexing is asynchronous - poll :meth:`status` until it reports
        ``ready``. Adding identical text twice is a no-op: the existing
        document comes back rather than being embedded again.
        """
        return self.create(agent_id, source_type="raw_text", title=title, raw_text=text)

    def status(self, agent_id: str, document_id: str) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._c.get(f"/api/agents/{agent_id}/knowledge/{document_id}/status").data,
        )

    def delete(self, agent_id: str, document_id: str) -> None:
        self._c.delete(f"/api/agents/{agent_id}/knowledge/{document_id}")


class _Analytics:
    def __init__(self, client: Integrable) -> None:
        self._c = client

    def for_agent(self, agent_id: str, days: int = 30) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._c.get(f"/api/agents/{agent_id}/analytics", query={"days": days}).data,
        )

    def gaps(self, agent_id: str) -> dict[str, Any]:
        """Questions the agent could not answer well."""
        return cast("dict[str, Any]", self._c.get(f"/api/agents/{agent_id}/analytics/gaps").data)


class _Webhooks:
    def __init__(self, client: Integrable) -> None:
        self._c = client

    def events(self) -> Any:
        """The event catalogue, each with a sample payload."""
        return cast("dict[str, Any]", self._c.get("/api/webhooks/events").data)

    def list(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self._c.get("/api/webhooks").data)

    def create(self, url: str, events: Sequence[str], **body: Any) -> dict[str, Any]:
        """Registers an endpoint. The signing secret is returned exactly once."""
        return cast(
            "dict[str, Any]",
            self._c.post("/api/webhooks", json={"url": url, "events": events, **body}).data,
        )

    def delete(self, webhook_id: str) -> None:
        self._c.delete(f"/api/webhooks/{webhook_id}")

    def test(self, webhook_id: str) -> dict[str, Any]:
        """Fires a sample payload down the real delivery path, signing included."""
        return cast("dict[str, Any]", self._c.post(f"/api/webhooks/{webhook_id}/test").data)

    def rotate_secret(self, webhook_id: str) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", self._c.post(f"/api/webhooks/{webhook_id}/rotate-secret").data
        )

    def deliveries(self, **query: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", self._c.get("/api/webhooks/deliveries", query=query).data)

    def replay(self, delivery_id: str) -> dict[str, Any]:
        """Writes a new delivery record rather than overwriting the failure."""
        return cast(
            "dict[str, Any]",
            self._c.post(f"/api/webhooks/deliveries/{delivery_id}/replay").data,
        )


# The async mirrors. Same paths, same argument names; only the awaits differ.


class _AsyncAgents:
    def __init__(self, client: AsyncIntegrable) -> None:
        self._c = client

    async def list(self, **query: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", (await self._c.get("/api/agents", query=query)).data)

    def walk(self, **query: Any) -> AsyncIterator[dict[str, Any]]:
        return self._c.paginate("/api/agents", query)

    async def get(self, agent_id: str) -> dict[str, Any]:
        return cast("dict[str, Any]", (await self._c.get(f"/api/agents/{agent_id}")).data)

    async def create(self, **body: Any) -> dict[str, Any]:
        return cast("dict[str, Any]", (await self._c.post("/api/agents", json=body)).data)

    async def update(self, agent_id: str, **body: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", (await self._c.patch(f"/api/agents/{agent_id}", json=body)).data
        )

    async def delete(self, agent_id: str) -> None:
        await self._c.delete(f"/api/agents/{agent_id}")


class _AsyncConversations:
    def __init__(self, client: AsyncIntegrable) -> None:
        self._c = client

    async def list(self, agent_id: str, **query: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            (await self._c.get(f"/api/agents/{agent_id}/conversations", query=query)).data,
        )

    def walk(self, agent_id: str, **query: Any) -> AsyncIterator[dict[str, Any]]:
        return self._c.paginate(f"/api/agents/{agent_id}/conversations", query)


class _AsyncKnowledge:
    def __init__(self, client: AsyncIntegrable) -> None:
        self._c = client

    async def list(self, agent_id: str, **query: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            (await self._c.get(f"/api/agents/{agent_id}/knowledge", query=query)).data,
        )

    def walk(self, agent_id: str, **query: Any) -> AsyncIterator[dict[str, Any]]:
        return self._c.paginate(f"/api/agents/{agent_id}/knowledge", query)

    async def create(self, agent_id: str, **body: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            (await self._c.post(f"/api/agents/{agent_id}/knowledge", json=body)).data,
        )

    async def add_text(self, agent_id: str, title: str, text: str) -> dict[str, Any]:
        return await self.create(agent_id, source_type="raw_text", title=title, raw_text=text)


class _AsyncAnalytics:
    def __init__(self, client: AsyncIntegrable) -> None:
        self._c = client

    async def for_agent(self, agent_id: str, days: int = 30) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            (await self._c.get(f"/api/agents/{agent_id}/analytics", query={"days": days})).data,
        )


class _AsyncWebhooks:
    def __init__(self, client: AsyncIntegrable) -> None:
        self._c = client

    async def list(self) -> dict[str, Any]:
        return cast("dict[str, Any]", (await self._c.get("/api/webhooks")).data)

    async def create(self, url: str, events: Sequence[str], **body: Any) -> dict[str, Any]:
        response = await self._c.post(
            "/api/webhooks", json={"url": url, "events": events, **body}
        )
        return cast("dict[str, Any]", response.data)
