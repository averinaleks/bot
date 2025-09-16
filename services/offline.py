"""Offline stubs for external services."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from types import SimpleNamespace

from bot.config import OFFLINE_MODE

logger = logging.getLogger("TradingBot")

_OFFLINE_ENV_DEFAULTS: dict[str, str] = {
    "BYBIT_API_KEY": "offline-bybit-key",
    "BYBIT_API_SECRET": "offline-bybit-secret",
    "TRADE_MANAGER_TOKEN": "offline-trade-token",
}


def ensure_offline_env(defaults: Mapping[str, str] | None = None) -> list[str]:
    """Ensure API credentials have placeholder values in offline mode.

    Parameters
    ----------
    defaults:
        Optional mapping of environment variables to fallback values. When a
        variable from ``defaults`` is absent and :data:`OFFLINE_MODE` is
        enabled, the value is injected into :mod:`os.environ` and a warning is
        logged. Already defined variables are never overridden.

    Returns
    -------
    list[str]
        Names of variables that received placeholder values. The list is empty
        when nothing was changed or :data:`OFFLINE_MODE` is disabled.
    """

    if not OFFLINE_MODE:
        return []

    applied: list[str] = []
    mapping = defaults or _OFFLINE_ENV_DEFAULTS
    for key, value in mapping.items():
        if os.getenv(key):
            continue
        os.environ[key] = value
        applied.append(key)
        logger.warning(
            "OFFLINE_MODE=1: переменная %s не задана; используется фиктивное значение",
            key,
        )
    return applied


if OFFLINE_MODE:
    ensure_offline_env()


class OfflineBybit:
    """Dummy Bybit client returning canned responses."""

    def __init__(self, *args, **kwargs) -> None:
        self.orders: list[dict] = []

    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float | None = None, *args, **kwargs) -> dict:
        order = {
            "id": len(self.orders) + 1,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
        }
        self.orders.append(order)
        return order

    def create_order_with_trailing_stop(self, *args, **kwargs) -> dict:
        return self.create_order(*args, **kwargs)

    def create_order_with_take_profit_and_stop_loss(self, *args, **kwargs) -> dict:
        return self.create_order(*args, **kwargs)

    def fetch_positions(self) -> list[dict]:
        return self.orders


class OfflineTelegram(logging.Handler):
    """Stub Telegram logger that logs messages locally."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.chat_id = kwargs.get("chat_id")

    async def send_telegram_message(self, message: str, urgent: bool = False) -> None:
        logger.info("[OFFLINE TELEGRAM] %s", message)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging
        pass

    @staticmethod
    async def shutdown() -> None:
        return None


class OfflineGPT:
    """Return predefined responses for GPT queries."""

    @staticmethod
    def query(prompt: str) -> str:
        return "offline response"

    @staticmethod
    async def query_async(prompt: str) -> str:
        return OfflineGPT.query(prompt)

    @staticmethod
    async def query_json_async(prompt: str) -> dict:
        return {"signal": "hold"}
