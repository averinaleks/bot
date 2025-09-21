"""Offline stubs for external services."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable, Mapping

from bot.config import OFFLINE_MODE
from services.logging_utils import sanitize_log_value

logger = logging.getLogger("TradingBot")

_PlaceholderValue = str | Callable[[], str]


def generate_placeholder_credential(name: str, *, entropy_bytes: int = 32) -> str:
    """Return a high-entropy placeholder credential for offline usage.

    The helper avoids embedding hard-coded secrets in the repository by
    generating a random token every time it is invoked.  The ``name`` is
    incorporated purely for debugging so operators can distinguish the purpose
    of the generated value when inspecting environment variables.
    """

    entropy = max(16, entropy_bytes)
    token = secrets.token_urlsafe(entropy)
    suffix = name.replace(" ", "-")
    return f"offline-{suffix}-{token}"


_OFFLINE_ENV_DEFAULTS: dict[str, _PlaceholderValue] = {
    "BYBIT_API_KEY": lambda: generate_placeholder_credential("bybit-key"),
    "BYBIT_API_SECRET": lambda: generate_placeholder_credential("bybit-secret"),
    "TRADE_MANAGER_TOKEN": lambda: generate_placeholder_credential("trade-token"),
}


def _resolve_placeholder(value: _PlaceholderValue) -> str:
    result = value() if callable(value) else value
    if not isinstance(result, str):
        raise TypeError("placeholder generator must return a string")
    if not result:
        raise ValueError("placeholder value must be non-empty")
    return result


def ensure_offline_env(
    defaults: Mapping[str, _PlaceholderValue] | None = None,
) -> list[str]:
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
    for key, raw_value in mapping.items():
        if os.getenv(key):
            continue
        try:
            resolved = _resolve_placeholder(raw_value)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.error(
                "OFFLINE_MODE=1: не удалось сгенерировать значение для %s: %s",
                sanitize_log_value(key),
                exc,
            )
            continue
        os.environ[key] = resolved
        applied.append(key)
        logger.warning(
            "OFFLINE_MODE=1: переменная %s не задана; используется фиктивное значение",
            sanitize_log_value(key),
        )
    return applied


if OFFLINE_MODE:
    ensure_offline_env()


class OfflineBybit:
    """Dummy Bybit client returning canned responses."""

    def __init__(self, *args, **kwargs) -> None:
        self.orders: list[dict] = []

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        *args,
        **kwargs,
    ) -> dict:
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
        logger.info("[OFFLINE TELEGRAM] %s", sanitize_log_value(message))

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
