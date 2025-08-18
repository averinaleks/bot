import logging
import os
import json
from urllib.parse import urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("TradingBot")


class GPTClientError(Exception):
    """Base exception for GPT client errors."""


class GPTClientNetworkError(GPTClientError):
    """Raised when the GPT OSS API cannot be reached."""


class GPTClientJSONError(GPTClientError):
    """Raised when the GPT OSS API returns invalid JSON."""


class GPTClientResponseError(GPTClientError):
    """Raised when the GPT OSS API returns an unexpected structure."""


def _validate_api_url(api_url: str) -> None:
    parsed = urlparse(api_url)
    if not parsed.scheme or not parsed.hostname:
        raise GPTClientError("Invalid GPT_OSS_API URL")
    if parsed.scheme != "https" and parsed.hostname != "localhost":
        logger.critical("Insecure GPT_OSS_API URL: %s", api_url)
        raise GPTClientError("GPT_OSS_API must use HTTPS or be localhost")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    reraise=True,
)
def _post_with_retry(url: str, prompt: str) -> httpx.Response:
    """POST helper that retries on network failures."""
    with httpx.Client(trust_env=False, timeout=5) as client:
        response = client.post(url, json={"prompt": prompt})
        response.raise_for_status()
        return response


def query_gpt(prompt: str) -> str:
    """Send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is read from the ``GPT_OSS_API`` environment variable. If
    it is not set a :class:`GPTClientNetworkError` is raised. Network errors are
    retried up to three times with exponential backoff between one and ten
    seconds before giving up.
    """
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        logger.error("Environment variable GPT_OSS_API is not set")
        raise GPTClientNetworkError("GPT_OSS_API environment variable not set")

    _validate_api_url(api_url)
    url = api_url.rstrip("/") + "/v1/completions"
    try:
        response = _post_with_retry(url, prompt)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    try:
        data = response.json()
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
        raise GPTClientResponseError("Unexpected response structure from GPT OSS API") from exc


async def query_gpt_async(prompt: str) -> str:
    """Asynchronously send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is taken from the ``GPT_OSS_API`` environment variable. If it
    is not set a :class:`GPTClientNetworkError` is raised. Request timeout is read
    from ``GPT_OSS_TIMEOUT`` (seconds, default ``5``). Network errors are retried
    up to three times with exponential backoff between one and ten seconds before
    giving up.

    Uses :class:`httpx.AsyncClient` for the HTTP request but mirrors the behaviour of
    :func:`query_gpt` including error handling and environment configuration.
    """
    api_url = os.getenv("GPT_OSS_API")
    if not api_url:
        logger.error("Environment variable GPT_OSS_API is not set")
        raise GPTClientNetworkError("GPT_OSS_API environment variable not set")

    _validate_api_url(api_url)
    timeout_env = os.getenv("GPT_OSS_TIMEOUT", "5")
    try:
        timeout = float(timeout_env)
    except ValueError:
        logger.warning(
            "Invalid GPT_OSS_TIMEOUT value %r; defaulting to 5.0", timeout_env
        )
        timeout = 5.0
    url = api_url.rstrip("/") + "/v1/completions"
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=10),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
                    response = await client.post(url, json={"prompt": prompt})
                    response.raise_for_status()
                    try:
                        data = response.json()
                    except ValueError as exc:
                        logger.exception("Invalid JSON response from GPT OSS API: %s", exc)
                        raise GPTClientJSONError(
                            "Invalid JSON response from GPT OSS API"
                        ) from exc
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    except GPTClientError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error querying GPT OSS API: %s", exc)
        raise GPTClientError("Unexpected error querying GPT OSS API") from exc
    try:
        return data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "Unexpected response structure from GPT OSS API: %s | data: %r",
            exc,
            data,
        )
        raise GPTClientResponseError("Unexpected response structure from GPT OSS API") from exc


async def query_gpt_json_async(prompt: str) -> dict:
    """Return JSON parsed from :func:`query_gpt_async` text output."""

    text = await query_gpt_async(prompt)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON from GPT OSS API: %s", exc)
        raise GPTClientJSONError("Invalid JSON response from GPT OSS API") from exc
    if not isinstance(data, dict):
        raise GPTClientResponseError("Unexpected response structure from GPT OSS API")
    return data
