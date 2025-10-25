"""Factory helpers for runtime services."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import config as bot_config
from services.logging_utils import sanitize_log_value
from services.stubs import is_offline_env

logger = logging.getLogger("TradingBot")

ConfigArg = Any


def _select_config(positional: ConfigArg | None, keyword: ConfigArg | None) -> ConfigArg | None:
    """Return the configuration object passed to a factory."""

    return keyword if keyword is not None else positional


def _should_use_offline(config: ConfigArg | None) -> bool:
    """Return ``True`` when offline stubs should be used."""

    if config is not None:
        for attr in ("offline", "offline_mode", "use_offline_services", "OFFLINE_MODE"):
            value = getattr(config, attr, None)
            if value is not None:
                return bool(value)
    if getattr(bot_config, "OFFLINE_MODE", False):
        return True
    return is_offline_env()


def build_exchange(
    cfg: ConfigArg | None = None,
    *,
    config: ConfigArg | None = None,
) -> Any:
    """Create a Bybit exchange client using ccxt or fall back to offline stub."""

    resolved_cfg = _select_config(cfg, config)
    if _should_use_offline(resolved_cfg):
        from services.offline import OfflineBybit

        return OfflineBybit()

    try:  # pragma: no cover - import exercised in integration tests
        import ccxt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - handled for minimal installs
        logger.warning(
            "`ccxt` не найден: используется OfflineBybit. Установите `pip install ccxt` для онлайн-режима (%s)",
            sanitize_log_value(str(exc)),
        )
        from services.offline import OfflineBybit

        return OfflineBybit()

    exchange_name = str(getattr(resolved_cfg, "exchange", "bybit") or "bybit").lower()
    if exchange_name != "bybit":
        logger.warning(
            "Поддерживается только биржа Bybit; используется значение %s",
            sanitize_log_value(exchange_name),
        )

    params = {
        "apiKey": os.getenv("BYBIT_API_KEY", ""),
        "secret": os.getenv("BYBIT_API_SECRET", ""),
        "enableRateLimit": True,
    }
    try:
        exchange = ccxt.bybit(params)
    except Exception as exc:  # pragma: no cover - unexpected ccxt failures
        logger.error(
            "Не удалось создать клиент ccxt.bybit: %s",
            sanitize_log_value(str(exc)),
        )
        from services.offline import OfflineBybit

        return OfflineBybit()
    return exchange


def build_telegram_logger(
    cfg: ConfigArg | None = None,
    *,
    config: ConfigArg | None = None,
) -> Callable[..., Any]:
    """Return a Telegram logger factory suited for the current environment."""

    resolved_cfg = _select_config(cfg, config)
    if _should_use_offline(resolved_cfg):
        from services.offline import OfflineTelegram

        return OfflineTelegram

    from telegram_logger import TelegramLogger

    return TelegramLogger


def build_gpt_client(
    cfg: ConfigArg | None = None,
    *,
    config: ConfigArg | None = None,
) -> Any:
    """Return a GPT client adapter depending on the environment."""

    resolved_cfg = _select_config(cfg, config)
    if _should_use_offline(resolved_cfg):
        from services.offline import OfflineGPT

        return OfflineGPT

    from bot import gpt_client as bot_gpt_client

    return bot_gpt_client


__all__ = [
    "build_exchange",
    "build_telegram_logger",
    "build_gpt_client",
]
