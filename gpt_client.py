import logging
import os
import httpx
import requests

logger = logging.getLogger("TradingBot")


class GPTClientError(Exception):
    """Base exception for GPT client errors."""


class GPTClientNetworkError(GPTClientError):
    """Raised when the GPT OSS API cannot be reached."""


class GPTClientJSONError(GPTClientError):
    """Raised when the GPT OSS API returns invalid JSON."""


class GPTClientResponseError(GPTClientError):
    """Raised when the GPT OSS API returns an unexpected structure."""


def query_gpt(prompt: str) -> str:
    """Send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is read from the ``GPT_OSS_API`` environment variable.
    If it is not set, ``http://localhost:8003`` is used.
    """
    api_url = os.getenv("GPT_OSS_API", "http://localhost:8003")
    url = api_url.rstrip("/") + "/v1/completions"
    try:
        response = requests.post(url, json={"prompt": prompt}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network errors
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

    Uses :class:`httpx.AsyncClient` for the HTTP request but mirrors the behaviour of
    :func:`query_gpt` including error handling and environment configuration.
    """
    api_url = os.getenv("GPT_OSS_API", "http://localhost:8003")
    url = api_url.rstrip("/") + "/v1/completions"
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(url, json={"prompt": prompt}, timeout=5)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                logger.exception("Invalid JSON response from GPT OSS API: %s", exc)
                raise GPTClientJSONError("Invalid JSON response from GPT OSS API") from exc
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.exception("Error querying GPT OSS API: %s", exc)
        raise GPTClientNetworkError("Failed to query GPT OSS API") from exc
    try:
        return data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "Unexpected response structure from GPT OSS API: %s | data: %r",
            exc,
            data,
        )
        raise GPTClientResponseError("Unexpected response structure from GPT OSS API") from exc
