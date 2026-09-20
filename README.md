# integrable-cloud

Official Python SDK for the [Integrable Cloud](https://integrable.cloud) API — website chat agents, conversations, knowledge bases and analytics.

```bash
pip install integrable-cloud
```

Python 3.10+. Sync and async clients, one dependency (`httpx`).

## Quick start

```python
import os
from integrable_cloud import Integrable

client = Integrable(api_key=os.environ["INTEGRABLE_API_KEY"])

for agent in client.agents.walk():
    print(agent["id"], agent["name"], agent["status"])
```

Create a key in the dashboard under **Settings → API keys**. It starts with `sk_live_` and is shown once.

## Async

The same surface, awaited:

```python
from integrable_cloud import AsyncIntegrable

async with AsyncIntegrable(api_key=...) as client:
    page = await client.agents.list()

    async for conversation in client.conversations.walk(agent_id, days=30):
        await sync_to_crm(conversation)
```

Both clients share one implementation of the retry policy, the idempotency rule and the error mapping — so they cannot drift apart, which is the usual failure when a library maintains two of everything.

## What it does that `httpx` does not

**Retries safely.** Transient failures — 429, 5xx, connection resets — are retried with jittered exponential backoff. A `Retry-After` header always wins over the backoff curve, because the server knows when the window resets and the client is guessing.

**Idempotency keys, automatically.** Every mutating request carries one, and a retry reuses the *same* key — so a create that timed out and was retried produces one agent rather than three:

```python
client.agents.create(name="Support", system_prompt="...")   # key generated for you
client.post("/api/agents", json=body, idempotency_key=my_stable_id)  # or supply one
```

A mutating request without a key is never retried: "did that land?" is exactly the question retrying cannot answer safely.

**Pagination that stays fast.**

```python
for conversation in client.conversations.walk(agent_id, days=30):
    process(conversation)
```

Follows the cursor lazily. Never increment a page number against this API — offset pagination makes the database read and discard every skipped row, so page 40 costs forty times page 1.

**Errors you can branch on.**

```python
from integrable_cloud import QuotaExceededError, RateLimitError, ValidationError

try:
    client.knowledge.add_text(agent_id, "Hours", "Open 9-5, Mon-Fri.")
except QuotaExceededError as e:
    print("Plan limit reached:", e.details)
except ValidationError as e:
    print("Bad request:", e.message, e.details)
except RateLimitError as e:
    print(f"Retry in {e.retry_after}s")
```

Every error carries `.request_id` — quote it at support and the exact call can be found. Every error also has `.retryable`, so a caller running its own queue can ask directly instead of re-deriving it from a status code.

**Rate-limit visibility.**

```python
client.agents.list()
print(client.rate_limit)   # RateLimit(limit=120, remaining=118, reset=41)
```

## Configuration

```python
client = Integrable(
    api_key=os.environ["INTEGRABLE_API_KEY"],
    base_url="https://api.integrable.cloud",  # override for staging
    timeout=60.0,                              # seconds, per request
    max_retries=2,                             # 0 disables
    default_headers={"x-app": "my-service"},
    http_client=my_instrumented_httpx_client,  # for tests or a proxy
)
```

## Reaching an endpoint with no wrapper

The typed resources cover the common calls. All 149 endpoints are reachable directly:

```python
response = client.get(f"/api/agents/{agent_id}/contacts", query={"limit": 50})
print(response.data["items"])
print(response.api_version, response.request_id)
```

## Also available over MCP

If you want to *ask* about your workspace rather than write code against it, the same API key connects it to Claude, ChatGPT or your editor over the Model Context Protocol — see [the MCP guide](https://integrable.cloud/docs/mcp).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -q
```

Tests are entirely offline, against a mocked transport — no credentials, no network.

## Releasing

Bump `version` in `pyproject.toml`, commit, then tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

The release workflow verifies the tag matches the package version, runs the full gate, and publishes.

## Links

- [Documentation](https://integrable.cloud/docs)
- [API reference](https://integrable.cloud/docs/api)
- [TypeScript SDK](https://github.com/integrable-cloud/sdk-typescript)

MIT © Integrable Cloud
