"""Utilities for interacting with the GPT OSS API.

The API endpoint is configured via the ``GPT_OSS_API`` environment variable.
For security reasons, the URL must either use HTTPS or resolve to a local
or private address (HTTP is allowed only for such addresses).
"""

import logging
import os
import json
import socket
from enum import Enum
from urllib.parse import urlparse, urljoin
from ipaddress import ip_address
import asyncio
import threading
from typing import Any, Coroutine

# NOTE: httpx is imported for exception types only.
import httpx

from pydantic import BaseModel, Field, ValidationError

from bot.utils import retry
# Absolute import ensures the project's own configuration module is used
# instead of any unrelated ``config`` module on the import path.
from bot.config import OFFLINE_MODE
if OFFLINE_MODE:
    from services.offline import OfflineGPT

logger = logging.getLogger("TradingBot")

# Maximum allowed prompt size in bytes
MAX_PROMPT_BYTES = 10000
# Maximum allowed response size in bytes
MAX_RESPONSE_BYTES = 10000
# Maximum number of retries for network requests
MAX_RETRIES = 3


def _allow_insecure_url() -> bool:
    """Return True if insecure GPT_OSS_API URLs are explicitly allowed."""
    return os.getenv("ALLOW_INSECURE_GPT_URL") == "1"

# Backward compatibility: expose module-level flag for tests/legacy code.
ALLOW_INSECURE_GPT_URL = _allow_insecure_url()


def _insecure_allowed() -> bool:
    """Return True if insecure GPT URLs are explicitly permitted."""

    return ALLOW_INSECURE_GPT_URL or os.getenv("ALLOW_INSECURE_GPT_URL") == "1"

class GPTClientError(Exception):
    """Base exception for GPT client errors."""


class GPTClientNetworkError(GPTClientError):
    """Raised when the GPT OSS API cannot be reached."""


class GPTClientJSONError(GPTClientError):
    """Raised when the GPT OSS API returns invalid JSON."""


class GPTClientResponseError(GPTClientError):
    """Raised when the GPT OSS API returns an unexpected structure."""


class SignalAction(str, Enum):
    """Possible trading actions."""

    buy = "buy"
    sell = "sell"
    hold = "hold"


class Signal(BaseModel):
    """Parsed trading signal from GPT output."""

    signal: SignalAction = SignalAction.hold
    tp_mult: float | None = Field(default=None, ge=0, le=10)
    sl_mult: float | None = Field(default=None, ge=0, le=10)


def _truncate_prompt(prompt: str) -> str:
    """Trim *prompt* to :data:`MAX_PROMPT_BYTES` bytes if necessary.

    A warning is logged when truncation happens.
    """

    prompt_bytes = prompt.encode("utf-8")
    if len(prompt_bytes) > MAX_PROMPT_BYTES:
        logger.warning(
            "Prompt exceeds maximum length of %s bytes; truncating",
            MAX_PROMPT_BYTES,
        )
        prompt = prompt_bytes[:MAX_PROMPT_BYTES].decode("utf-8", errors="ignore")
    return prompt


def _validate_api_url(api_url: str) -> tuple[str, set[str]]:
    parsed = urlparse(api_url)
    if not parsed.scheme:
        raise GPTClientError(
            "GPT_OSS_API URL must include a scheme (http or https)"
        )
    if not parsed.hostname:
        raise GPTClientError("GPT_OSS_API URL must include a hostname")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise GPTClientError(
            f"GPT_OSS_API URL scheme {scheme!r} is not supported; use http or https"
        )

    try:
        addr_info = socket.getaddrinfo(
            parsed.hostname, None, family=socket.AF_UNSPEC
        )
        resolved_ips = {info[4][0] for info in addr_info}
    except socket.gaierror as exc:
        if os.getenv("TEST_MODE") == "1":
            resolved_ips = {"127.0.0.1"}
        else:
            logger.error(
                "Failed to resolve GPT_OSS_API host %s: %s", parsed.hostname, exc
            )
            raise GPTClientError(
                f"GPT_OSS_API host {parsed.hostname!r} cannot be resolved"
            ) from exc


    if scheme == "http" and parsed.hostname != "localhost":
        private_ips = {
            ip
            for ip in resolved_ips
            if ip_address(ip).is_private
            or ip_address(ip).is_loopback
            or ip_address(ip).is_link_local
        }
        public_ips = resolved_ips - private_ips
        if private_ips and public_ips:
            raise GPTClientError(
                "GPT_OSS_API host resolves to both private and public addresses"
            )
        if public_ips:
            if _insecure_allowed():
                logger.warning("Using insecure GPT_OSS_API URL %s", api_url)
            else:
                raise GPTClientError(
                    "Insecure GPT_OSS_API URL; only private addresses are allowed"
                )
    return parsed.hostname, resolved_ips


async def _fetch_response(
    client: httpx.Client | httpx.AsyncClient,
    prompt: str,
    url: str,
    hostname: str,
    allowed_ips: set[str],
) -> bytes:
    """Resolve hostname, verify IP and return response bytes."""
    try:
        loop = asyncio.get_running_loop()
        current_ips = {
            info[4][0]
            for info in await loop.getaddrinfo(
                hostname, None, family=socket.AF_UNSPEC
            )
        }
    except socket.gaierror as exc:
        if os.getenv("TEST_MODE") == "1":
            current_ips = allowed_ips
        else:
            logger.error(
                "Failed to resolve GPT_OSS_API host %s before request: %s",
                hostname,
                exc,
            )
            raise GPTClientNetworkError("Failed to resolve GPT_OSS_API host") from exc

    if not current_ips & allowed_ips:
        logger.error(
            "GPT_OSS_API host IP mismatch: %s resolved to %s, expected %s",
            hostname,
            current_ips,
            allowed_ips,
        )
        raise GPTClientNetworkError("GPT_OSS_API host resolution mismatch")

    if isinstance(client, httpx.AsyncClient):
        async with client.stream("POST", url, json={"prompt": prompt}) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("application/json"):
                raise GPTClientResponseError("Unexpected Content-Type from GPT OSS API")
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > MAX_RESPONSE_BYTES:
                    raise GPTClientError("Response exceeds maximum length")
            return bytes(content)
    else:
        with client.stream("POST", url, json={"prompt": prompt}) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("application/json"):
                raise GPTClientResponseError("Unexpected Content-Type from GPT OSS API")
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > MAX_RESPONSE_BYTES:
                    raise GPTClientError("Response exceeds maximum length")
            return bytes(content)


def _get_api_url_timeout() -> tuple[str, float, str, set[str]]:
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        message = (
            "GPT_OSS_API environment variable is not set. "
            "Set GPT_OSS_API to the base URL of the GPT OSS service, e.g. http://localhost:8003."
        )
        logger.error(message)
        raise GPTClientNetworkError(message)

    hostname, allowed_ips = _validate_api_url(api_url)

    timeout_env = os.getenv("GPT_OSS_TIMEOUT", "5")
    try:
        timeout = float(timeout_env)
        if timeout <= 0:
            logger.warning(
                "Non-positive GPT_OSS_TIMEOUT value %r; defaulting to 5.0",
                timeout_env,
            )
            timeout = 5.0
    except ValueError:
        logger.warning(
            "Invalid GPT_OSS_TIMEOUT value %r; defaulting to 5.0", timeout_env
        )
        timeout = 5.0

    api_url = api_url.rstrip("/")
    url = urljoin(api_url + "/", "v1/completions")
    return url, timeout, hostname, allowed_ips


def _parse_gpt_response(content: bytes) -> str:
    """Parse GPT OSS API JSON *content* and return the first completion text."""
    try:
        data = json.loads(content)
    except ValueError as exc:
        logger.exception("Invalid JSON response from GPT OSS API: %s", exc)
        raise GPTClientJSONError("Invalid JSON response from GPT OSS API") from exc
    try:
        return data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "Unexpected response structure from GPT OSS API: %s | data: %r",
            exc,
            data,
        )
        raise GPTClientResponseError(
            "Unexpected response structure from GPT OSS API"
        ) from exc


def query_gpt(prompt: str) -> str:
    """Send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is read from the ``GPT_OSS_API`` environment variable. If
    it is not set a :class:`GPTClientNetworkError` is raised. Request timeout is
    read from ``GPT_OSS_TIMEOUT`` (seconds, default ``5``). Network errors are
    retried up to MAX_RETRIES times with exponential backoff between one and ten
    seconds before giving up. Prompts longer than :data:`MAX_PROMPT_BYTES` are
    truncated with a warning.
    """
    if OFFLINE_MODE:
        return OfflineGPT.query(prompt)

    prompt = _truncate_prompt(prompt)
    url, timeout, hostname, allowed_ips = _get_api_url_timeout()

    # Maximum time to wait for the asynchronous task considering retries and backoff
    backoff_total = sum(min(2**i, 10) for i in range(MAX_RETRIES - 1))
    max_wait_time = timeout * MAX_RETRIES + backoff_total

    def _run_coro_in_thread(coro: Coroutine[Any, Any, bytes]) -> bytes:
        result: dict[str, Any] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - unexpected
                result["error"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(max_wait_time)
        if thread.is_alive():
            raise GPTClientError(
                "Timed out waiting for async task after maximum retries"
            )
        if "error" in result:
            raise result["error"]
        return result["value"]

    @retry(
        MAX_RETRIES,
        lambda attempt: min(2 ** (attempt - 1), 10),
    )
    def _post() -> bytes:
        async def _async_post() -> bytes:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                return await _fetch_response(
                    client, prompt, url, hostname, allowed_ips
                )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_async_post())
        else:
            return _run_coro_in_thread(_async_post())

    try:
        content = _post()
    except httpx.TimeoutException as exc:  # pragma: no cover - request timeout
        logger.exception("Timed out querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError(
            "GPT OSS API request timed out after maximum retries"
        ) from exc
    except GPTClientError:
        raise
    except httpx.HTTPError as exc:  # pragma: no cover - other network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    return _parse_gpt_response(content)


async def query_gpt_async(prompt: str) -> str:
    """Asynchronously send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is taken from the ``GPT_OSS_API`` environment variable. If it
    is not set a :class:`GPTClientNetworkError` is raised. Request timeout is read
    from ``GPT_OSS_TIMEOUT`` (seconds, default ``5``). Network errors are retried
    up to MAX_RETRIES times with exponential backoff between one and ten seconds
    before giving up.

    Uses :class:`httpx.AsyncClient` for the HTTP request but mirrors the behaviour of
    :func:`query_gpt` including error handling and environment configuration.
    Prompts longer than :data:`MAX_PROMPT_BYTES` are truncated with a warning.
    """
    if OFFLINE_MODE:
        return await OfflineGPT.query_async(prompt)

    prompt = _truncate_prompt(prompt)
    url, timeout, hostname, allowed_ips = await asyncio.to_thread(
        _get_api_url_timeout
    )

    @retry(
        MAX_RETRIES,
        lambda attempt: min(2 ** (attempt - 1), 10),
    )
    async def _post() -> bytes:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            return await _fetch_response(client, prompt, url, hostname, allowed_ips)

    try:
        content = await _post()
    except httpx.TimeoutException as exc:  # pragma: no cover - request timeout
        logger.exception("Timed out querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError(
            "GPT OSS API request timed out after maximum retries"
        ) from exc
    except GPTClientError:
        raise
    except httpx.HTTPError as exc:  # pragma: no cover - other network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error querying GPT OSS API: %s", exc)
        raise GPTClientError("Unexpected error querying GPT OSS API") from exc
    return _parse_gpt_response(content)


async def query_gpt_json_async(prompt: str) -> dict:
    """Return JSON parsed from :func:`query_gpt_async` text output."""

    if OFFLINE_MODE:
        return await OfflineGPT.query_json_async(prompt)

    text = await query_gpt_async(prompt)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON from GPT OSS API: %s", exc)
        raise GPTClientJSONError("Invalid JSON response from GPT OSS API") from exc
    if not isinstance(data, dict):
        raise GPTClientResponseError("Unexpected response structure from GPT OSS API")
    required = {"signal", "tp_mult", "sl_mult"}
    if not required.issubset(data):
        missing = sorted(required - set(data))
        logger.error("Missing fields in GPT response: %s", missing)
        return {"signal": "hold"}
    return data


def parse_signal(payload: str) -> Signal:
    """Parse *payload* JSON into a :class:`Signal` instance.

    On any parsing or validation error a default ``hold`` signal is returned and
    a warning is logged.
    """

    try:
        data = json.loads(payload)
        return Signal.model_validate(data)
    except (ValueError, ValidationError, TypeError) as exc:  # pragma: no cover - invalid input
        logger.warning("Failed to parse signal: %s", exc)
        return Signal()
