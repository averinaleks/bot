"""Offline stubs for external services."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

logger = logging.getLogger("TradingBot")


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
