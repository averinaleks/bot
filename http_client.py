"""Utilities for creating HTTP clients with default timeouts."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, Tuple

from bot.utils import retry
from services.logging_utils import sanitize_log_value
from services.stubs import create_httpx_stub, is_offline_env

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from httpx import AsyncClient as HTTPXAsyncClient
    from httpx import Client as HTTPXClient
    from httpx import Headers as HTTPXHeaders
    from httpx import Request as HTTPXRequest
    from httpx import Response as HTTPXResponse
else:  # pragma: no cover - runtime fallback when httpx missing
    HTTPXAsyncClient = HTTPXClient = HTTPXHeaders = HTTPXRequest = HTTPXResponse = Any


_OFFLINE_ENV = is_offline_env()

httpx: Any

try:  # pragma: no cover - exercised when httpx is available
    if _OFFLINE_ENV:
        raise ImportError("offline mode uses httpx stub")
    import httpx as _httpx  # type: ignore
except Exception:  # noqa: BLE001 - guarantee stub availability
    httpx = create_httpx_stub()
else:
    httpx = _httpx

# Use system-level randomness for jitter to avoid predictable retry delays
_RNG = random.SystemRandom()
if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    import requests  # type: ignore[import-untyped]

DEFAULT_TIMEOUT_STR = os.getenv("MODEL_DOWNLOAD_TIMEOUT", "10")
try:
    DEFAULT_TIMEOUT = float(DEFAULT_TIMEOUT_STR)
except ValueError:
    logging.warning(
        "Invalid MODEL_DOWNLOAD_TIMEOUT '%s'; using default timeout 10s",
        sanitize_log_value(DEFAULT_TIMEOUT_STR),
    )
    DEFAULT_TIMEOUT = 10.0


@contextmanager
def get_requests_session(
    timeout: float = DEFAULT_TIMEOUT,
    *,
    verify: bool | None = True,
) -> Generator["requests.Session", None, None]:
    """Return a :class:`requests.Session` with a default timeout.

    The import is deferred so the module can be used without the optional
    ``requests`` dependency installed.

    Parameters
    ----------
    timeout:
        Default timeout in seconds applied to requests made via the session.
    verify:
        Optional SSL verification flag mirroring :mod:`requests` semantics.
        ``None`` leaves the library default untouched, whereas ``True`` and
        ``False`` explicitly enable or disable certificate checks.
    """
    import requests  # type: ignore[import-untyped]

    session = requests.Session()
    # Avoid respecting proxy-related environment variables which can cause
    # local connections (e.g. to the mock GPT server in CI) to be routed through
    # a non-existent proxy and hang until a network timeout occurs.
    session.trust_env = False
    session.proxies = {}
    if verify is not None:
        session.verify = verify
    original = session.request

    @wraps(original)
    def request(method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("allow_redirects", False)
        return original(method, url, **kwargs)

    session.request = request  # type: ignore[assignment]
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_httpx_client(
    timeout: float = DEFAULT_TIMEOUT, **kwargs
) -> Generator[HTTPXClient, None, None]:
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

_ASYNC_CLIENT: HTTPXAsyncClient | None = None
_ASYNC_CLIENT_LOCK = asyncio.Lock()

# In-memory caches and metrics for HTTP requests
REFERENCE_CACHE: Dict[str, Tuple[int, HTTPXHeaders, bytes]] = {}
RETRY_METRICS: defaultdict[str, int] = defaultdict(int)


async def get_async_http_client(
    timeout: float = 10.0, **kwargs
) -> HTTPXAsyncClient:
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
    timeout: float = 10.0, **kwargs
) -> AsyncGenerator[HTTPXAsyncClient, None]:
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


@retry(5, lambda base: min(base, 8) + _RNG.uniform(0, 1))
async def _send_request(
    method: str, url: str, *, client: HTTPXAsyncClient, **kwargs: Any
) -> HTTPXResponse:
    try:
        resp = await client.request(method, url, **kwargs)
        if resp.status_code in (429,) or resp.status_code >= 500:
            resp.raise_for_status()
        return resp
    except Exception:
        RETRY_METRICS[url] += 1
        logging.exception("Async HTTP request failed")
        raise


async def request_with_retry(
    method: str,
    url: str,
    *,
    client: HTTPXAsyncClient | None = None,
    **kwargs: Any,
) -> HTTPXResponse:
    """Perform an HTTP request with retries and caching for reference data."""

    cacheable = any(key in url for key in ("limits", "symbols"))
    if cacheable and url in REFERENCE_CACHE:
        status, headers, content = REFERENCE_CACHE[url]
        request = httpx.Request(method, url)
        return httpx.Response(status, headers=headers, content=content, request=request)

    if client is None:
        client = await get_async_http_client()

    resp = await _send_request(method, url, client=client, **kwargs)

    if cacheable:
        content = await resp.aread()
        REFERENCE_CACHE[url] = (resp.status_code, resp.headers, content)
        resp._content = content  # type: ignore[attr-defined]
    return resp

