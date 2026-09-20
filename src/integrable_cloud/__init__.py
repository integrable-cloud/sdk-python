"""Official Python SDK for the Integrable Cloud API.

    from integrable_cloud import Integrable

    client = Integrable(api_key=os.environ["INTEGRABLE_API_KEY"])
    for agent in client.agents.walk():
        print(agent["id"], agent["name"])

Sync and async clients share one implementation of the retry policy, the
idempotency rule and the error mapping, so the two cannot drift.
"""

from ._client import DEFAULT_BASE_URL, ApiResponse, RateLimit
from .client import AsyncIntegrable, Integrable
from .errors import (
    AuthenticationError,
    ConflictError,
    ConnectionError_,
    IntegrableError,
    NotFoundError,
    PermissionError_,
    PreconditionRequiredError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    TimeoutError_,
    ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "Integrable",
    "AsyncIntegrable",
    "ApiResponse",
    "RateLimit",
    "DEFAULT_BASE_URL",
    "IntegrableError",
    "AuthenticationError",
    "PermissionError_",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "QuotaExceededError",
    "PreconditionRequiredError",
    "RateLimitError",
    "ServerError",
    "ConnectionError_",
    "TimeoutError_",
    "__version__",
]
