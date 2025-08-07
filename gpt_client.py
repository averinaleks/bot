import logging
import os
import requests

logger = logging.getLogger("TradingBot")


def query_gpt(prompt: str) -> str:
    """Send *prompt* to the GPT OSS API and return the first completion text.

    The API endpoint is read from the ``GPT_OSS_API`` environment variable.
    If it is not set, ``http://localhost:8003`` is used.
    """
    api_url = os.getenv("GPT_OSS_API", "http://localhost:8003")
    url = api_url.rstrip("/") + "/completions"
    try:
        response = requests.post(url, json={"prompt": prompt}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Error querying GPT OSS API: %s", exc)
        return ""
    try:
        data = response.json()
    except ValueError as exc:
        logger.exception("Invalid JSON response from GPT OSS API: %s", exc)
        return ""
    try:
        return data["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "Unexpected response structure from GPT OSS API: %s | data: %r",
            exc,
            data,
        )
        return ""
