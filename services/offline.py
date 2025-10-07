"""Offline stubs for external services."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping
from typing import SupportsFloat, SupportsIndex, SupportsInt, cast

from bot import config as bot_config
from services.logging_utils import sanitize_log_value

logger = logging.getLogger("TradingBot")

# ``OFFLINE_MODE`` mirrors :data:`bot.config.OFFLINE_MODE` so that tests can
# monkeypatch the flag directly on this module without poking at the config
# package.  The constant must therefore be defined at import time.
OFFLINE_MODE: bool = bool(bot_config.OFFLINE_MODE)

_PlaceholderValue = str | Callable[[], str]
_IntLike = SupportsInt | SupportsIndex | str | bytes | bytearray
_FloatLike = SupportsFloat | SupportsIndex | str | bytes | bytearray

# Personalisation string for deterministic price derivation in ``OfflineBybit``.
# The value must be at most 16 bytes as required by ``hashlib.blake2s``.
_OFFLINE_PRICE_PERSONALISATION = b"offline-price"

# Mirror the configuration flag so tests can override it via monkeypatch.
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
    "TELEGRAM_BOT_TOKEN": lambda: generate_placeholder_credential("telegram-token"),
    "TELEGRAM_CHAT_ID": lambda: generate_placeholder_credential("telegram-chat"),
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
        self._prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Override cached price for ``symbol`` in tests or simulations."""

        if price <= 0:
            raise ValueError("offline price must be positive")
        self._prices[symbol.upper()] = float(price)

    def _resolve_price(self, symbol: str) -> float:
        cached = self._prices.get(symbol.upper())
        if cached and cached > 0:
            return cached
        # Derive a deterministic pseudo price from the symbol name using a
        # collision-resistant hash.  ``hash`` is intentionally avoided because
        # its salted randomisation would yield non-reproducible values across
        # interpreter restarts, which Semgrep rightfully flags as
        # ``python.lang.correctness.use-hash``.  Using ``blake2s`` keeps the
        # offline fixtures stable without relying on insecure randomness.
        normalised = symbol.upper().encode("utf-8", "surrogatepass")
        digest = hashlib.blake2s(
            normalised,
            digest_size=16,
            person=_OFFLINE_PRICE_PERSONALISATION,
        ).digest()
        base = int.from_bytes(digest, "big") % 10_000
        price = 10_000 + base
        resolved = float(price)
        self._prices[symbol.upper()] = resolved
        return resolved

    def fetch_ticker(self, symbol: str) -> dict:
        """Return a deterministic pseudo ticker for ``symbol``."""

        return {"last": self._resolve_price(symbol)}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 200,
    ) -> list[list[float]]:
        """Generate a deterministic OHLCV series for ``symbol``."""

        limit = max(1, int(limit or 0))
        base_price = self._resolve_price(symbol)
        candles: list[list[float]] = []
        now_ms = int(time.time() * 1000)
        step = 60_000
        for index in range(limit):
            ts = now_ms - (limit - index) * step
            delta = math.sin(index / 3.0) * 0.01 * base_price
            open_price = base_price + delta
            close_price = base_price - delta
            high_price = max(open_price, close_price) * 1.01
            low_price = min(open_price, close_price) * 0.99
            volume = 1_000 + index * 10
            candles.append(
                [
                    float(ts),
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    float(volume),
                ]
            )
        return candles

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


class OfflineTradeManager:
    """Lightweight stand-in for :class:`bot.trade_manager.core.TradeManager`."""

    def __init__(
        self,
        config,
        data_handler,
        model_builder,
        telegram_bot,
        chat_id,
        rl_agent=None,
        *,
        telegram_logger_factory=None,
        gpt_client_factory=None,
    ) -> None:
        ensure_offline_env()

        self.config = config
        self.data_handler = data_handler
        self.model_builder = model_builder
        self.telegram_bot = telegram_bot
        self.chat_id = chat_id
        self.rl_agent = rl_agent
        self.telegram_logger_factory = telegram_logger_factory
        self.gpt_client_factory = gpt_client_factory
        self.exchange = getattr(data_handler, "exchange", None)

        self._stop_event = asyncio.Event()
        self._finished = asyncio.Event()

        self.telegram_logger = self._build_telegram_logger(
            telegram_logger_factory,
            telegram_bot,
            chat_id,
        )

        self._max_iterations = self._resolve_iterations(config)
        self._iteration_delay = self._resolve_delay(config)

    @staticmethod
    def _resolve_iterations(config) -> int:
        raw = getattr(config, "offline_iterations", None)
        if raw is None:
            return 1

        try:
            value = int(cast(_IntLike, raw))
        except (TypeError, ValueError):
            return 1
        return max(1, value)

    @staticmethod
    def _resolve_delay(config) -> float:
        raw = getattr(config, "offline_iteration_delay", None)
        if raw is None:
            return 0.0
        try:
            value = float(cast(_FloatLike, raw))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, value)

    def _build_telegram_logger(self, factory, telegram_bot, chat_id):
        if factory is None:
            return None

        constructors = (
            lambda: factory(self.config),
            lambda: factory(config=self.config),
            lambda: factory(telegram_bot, chat_id),
            lambda: factory(),
        )

        for creator in constructors:
            try:
                return creator()
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to instantiate offline telegram logger: %s", exc)
                break
        return None

    async def run(self) -> None:
        """Execute a short, deterministic offline trading cycle."""

        logger.info("Starting offline trading cycle")
        try:
            for iteration in range(self._max_iterations):
                if self._stop_event.is_set():
                    break

                await self._maybe_refresh_data()
                await self._maybe_update_model()

                await asyncio.sleep(self._iteration_delay)
                logger.debug("Offline iteration %d completed", iteration + 1)
        finally:
            self._finished.set()
            logger.info("Offline trading cycle finished")

    async def stop(self) -> None:
        """Signal the offline loop to stop and release resources."""

        self._stop_event.set()
        if not self._finished.is_set():
            await self._finished.wait()

        logger.info("Offline TradeManager stopped")

        logger_instance = getattr(self.telegram_logger, "shutdown", None)
        if callable(logger_instance):
            with contextlib.suppress(Exception):
                result = logger_instance()
                if asyncio.iscoroutine(result):
                    await result

    async def _maybe_refresh_data(self) -> None:
        refresh = getattr(self.data_handler, "refresh", None)
        if not callable(refresh):
            return

        try:
            result = refresh()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Offline data refresh failed: %s", exc)

    async def _maybe_update_model(self) -> None:
        builder = getattr(self.model_builder, "update_models", None)
        if not callable(builder):
            return

        try:
            result = builder()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Offline model update failed: %s", exc)


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


OFFLINE_SERVICE_FACTORIES: dict[str, str] = {
    "exchange": "services.offline:OfflineBybit",
    "telegram_logger": "services.offline:OfflineTelegram",
    "gpt_client": "services.offline:OfflineGPT",
}
