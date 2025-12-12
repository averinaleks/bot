"""Logging configuration helpers."""
from __future__ import annotations

import logging
import os

from bot.utils import configure_logging as _base_configure_logging
from bot.telegram_logger import TelegramLogger


def configure_logging() -> None:
    """Configure standard handlers and optional Telegram logging."""
    _base_configure_logging()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        from telegram import Bot  # type: ignore[attr-defined]

        bot = Bot(token)
        handler = TelegramLogger(bot, chat_id, level=logging.ERROR)
        root = logging.getLogger()
        root.addHandler(handler)
    except Exception as exc:  # pragma: no cover - best effort
        logging.getLogger("TradingBot").exception(
            "Failed to initialize Telegram logger: %s", exc
        )
