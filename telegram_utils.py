import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from bot import config as bot_config
from http_client import get_async_http_client
from services.logging_utils import sanitize_log_value
from telegram_logger import TelegramLogger, resolve_unsent_path
from utils import safe_float, safe_int

logger = logging.getLogger(__name__)

CFG = bot_config.BotConfig()

if CFG.save_unsent_telegram:
    try:
        _UNSENT_FALLBACK_PATH: Path | None = resolve_unsent_path(
            CFG.log_dir, CFG.unsent_telegram_path
        )
    except ValueError as exc:
        logger.warning(
            "Ignoring unsafe unsent_telegram_path %s: %s",
            sanitize_log_value(CFG.unsent_telegram_path),
            sanitize_log_value(str(exc)),
        )
        _UNSENT_FALLBACK_PATH = None
else:
    _UNSENT_FALLBACK_PATH = None


async def send_telegram_alert(
    message: str,
    *,
    get_http_client: Callable[[], Awaitable[httpx.AsyncClient]] | None = None,
) -> None:
    """Send a Telegram notification if credentials are configured."""

    log_message = sanitize_log_value(message)

    if bot_config.OFFLINE_MODE:
        logger.debug("Offline mode enabled, Telegram alert suppressed: %s", log_message)
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram inactive, message not sent: %s", log_message)
        return
    redaction = "***"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe_url = sanitize_log_value(url.replace(token, redaction))
    get_client = get_http_client or get_async_http_client
    client = await get_client()
    max_attempts = safe_int(
        os.getenv("TELEGRAM_ALERT_RETRIES"),
        3,
        env_var="TELEGRAM_ALERT_RETRIES",
    )
    delay = safe_float(
        os.getenv("TELEGRAM_ALERT_DELAY"),
        1.0,
        env_var="TELEGRAM_ALERT_DELAY",
    )
    payload = {"chat_id": chat_id, "text": message}
    for attempt in range(1, max_attempts + 1):
        try:
            try:
                resp = await client.post(url, json=payload, timeout=10)
            except TypeError:
                resp = await client.post(url, data=payload, timeout=10)
            raise_for_status = getattr(resp, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            reason = f"network error ({exc.__class__.__name__})"
        except httpx.HTTPError as exc:  # pragma: no cover - other network errors
            reason = sanitize_log_value(str(exc).replace(token, redaction))
        if attempt == max_attempts:
            logger.error(
                "Failed to send Telegram alert after %s attempts (%s via %s): %s",
                max_attempts,
                reason,
                safe_url,
                log_message,
            )
            if CFG.save_unsent_telegram and _UNSENT_FALLBACK_PATH is not None:
                _logger = type("_TL", (), {"unsent_path": _UNSENT_FALLBACK_PATH})()
                TelegramLogger._save_unsent(_logger, chat_id, message)
            return
        logger.warning(
            "Failed to send Telegram alert (attempt %s/%s): %s (%s)",
            attempt,
            max_attempts,
            safe_url,
            reason,
        )
        if delay > 0:
            await asyncio.sleep(delay)
        delay *= 2
