"""Utilities for creating HTTP clients with default timeouts."""

from __future__ import annotations

import logging
import os
import asyncio
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from collections import defaultdict
from typing import AsyncGenerator, Generator, TYPE_CHECKING, Any, Dict, Tuple
import random

# Use system-level randomness for jitter to avoid predictable retry delays
_RNG = random.SystemRandom()

import httpx
if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    import requests  # type: ignore[import-untyped]

DEFAULT_TIMEOUT_STR = os.getenv("MODEL_DOWNLOAD_TIMEOUT", "30")
try:
    DEFAULT_TIMEOUT = float(DEFAULT_TIMEOUT_STR)
except ValueError:
    logging.warning(
        "Invalid MODEL_DOWNLOAD_TIMEOUT '%s'; using default timeout 30s",
        DEFAULT_TIMEOUT_STR,
    )
    DEFAULT_TIMEOUT = 30.0


@contextmanager
def get_requests_session(
    timeout: float = DEFAULT_TIMEOUT,
) -> Generator["requests.Session", None, None]:
    """Return a :class:`requests.Session` with a default timeout.

    The import is deferred so the module can be used without the optional
    ``requests`` dependency installed.
    """
    import requests  # type: ignore[import-untyped]

    session = requests.Session()
    # Avoid respecting proxy-related environment variables which can cause
    # local connections (e.g. to the mock GPT server in CI) to be routed through
    # a non-existent proxy and hang until a network timeout occurs.
    session.trust_env = False
    session.proxies = {}
    original = session.request

    @wraps(original)
    def request(method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original(method, url, **kwargs)

    session.request = request  # type: ignore[assignment]
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_httpx_client(
    timeout: float = DEFAULT_TIMEOUT, **kwargs
) -> Generator[httpx.Client, None, None]:
    """Return an :class:`httpx.Client` with a default timeout."""
    kwargs.setdefault("timeout", timeout)
    # For consistency with the asynchronous helpers, avoid inheriting proxy
    # settings from the environment unless explicitly requested.  This mirrors
    # the behaviour of :func:`get_async_http_client` and prevents surprising
    # proxy usage in environments where variables like ``HTTP_PROXY`` are set.
    kwargs.setdefault("trust_env", False)
    client = httpx.Client(**kwargs)
    try:
        yield client
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Asynchronous HTTPX client management
# ---------------------------------------------------------------------------

_ASYNC_CLIENT: httpx.AsyncClient | None = None
_ASYNC_CLIENT_LOCK = asyncio.Lock()

# In-memory caches and metrics for HTTP requests
REFERENCE_CACHE: Dict[str, Tuple[int, httpx.Headers, bytes]] = {}
RETRY_METRICS: defaultdict[str, int] = defaultdict(int)


async def get_async_http_client(
    timeout: float = DEFAULT_TIMEOUT, **kwargs
) -> httpx.AsyncClient:
    """Return a shared :class:`httpx.AsyncClient` instance."""
    global _ASYNC_CLIENT
    async with _ASYNC_CLIENT_LOCK:
        if _ASYNC_CLIENT is None:
            kwargs.setdefault("timeout", timeout)
            kwargs.setdefault("trust_env", False)
            try:
                _ASYNC_CLIENT = httpx.AsyncClient(**kwargs)
            except TypeError:  # pragma: no cover - stubbed client
                _ASYNC_CLIENT = httpx.AsyncClient()
    return _ASYNC_CLIENT


async def close_async_http_client() -> None:
    """Close the shared asynchronous HTTP client if it exists."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None:
        close = getattr(_ASYNC_CLIENT, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:
                logging.exception("Failed to close async HTTP client")
        _ASYNC_CLIENT = None


@asynccontextmanager
async def async_http_client(
    timeout: float = DEFAULT_TIMEOUT, **kwargs
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Context manager providing a temporary :class:`httpx.AsyncClient`."""
    kwargs.setdefault("timeout", timeout)
    kwargs.setdefault("trust_env", False)
    try:
        client = httpx.AsyncClient(**kwargs)
    except TypeError:  # pragma: no cover - stubbed client
        client = httpx.AsyncClient()
    try:
        yield client
    finally:
        close = getattr(client, "aclose", None)
        if callable(close):
            await close()


async def request_with_retry(
    method: str,
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    max_attempts: int = 5,
    backoff_base: float = 0.5,
    jitter: float = 0.1,
    **kwargs: Any,
) -> httpx.Response:
    """Perform an HTTP request with retries and caching for reference data."""

    cacheable = any(key in url for key in ("limits", "symbols"))
    if cacheable and url in REFERENCE_CACHE:
        status, headers, content = REFERENCE_CACHE[url]
        request = httpx.Request(method, url)
        return httpx.Response(status, headers=headers, content=content, request=request)

    if client is None:
        client = await get_async_http_client()

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = await client.request(method, url, **kwargs)
            status_code = resp.status_code
            if status_code not in (429,) and status_code < 500:
                if cacheable:
                    content = await resp.aread()
                    REFERENCE_CACHE[url] = (status_code, resp.headers, content)
                    resp._content = content  # type: ignore[attr-defined]
                return resp
        except httpx.HTTPError as exc:
            last_exc = exc
            resp = None

        if attempt == max_attempts - 1:
            if last_exc is not None:
                raise last_exc
            if resp is not None:
                return resp
            raise RuntimeError("HTTP request failed without response")

        RETRY_METRICS[url] += 1
        delay = backoff_base * (2 ** attempt) + _RNG.uniform(0, jitter)
        await asyncio.sleep(delay)

