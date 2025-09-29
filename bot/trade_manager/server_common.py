"""Shared helpers for TradeManager services and package entrypoints."""

from __future__ import annotations

import hmac
import logging
import os
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from bot.config import load_config
from bot.dotenv_utils import load_dotenv
from services.exchange_provider import ExchangeProvider

__all__ = [
    "load_environment",
    "get_api_token",
    "extract_request_token",
    "validate_token",
    "load_trade_manager_config",
    "ExchangeRuntime",
    "allow_unauthenticated_requests",
]

logger = logging.getLogger(__name__)


def load_environment() -> None:
    """Load environment variables from ``.env`` when available."""

    load_dotenv()


def get_api_token() -> str | None:
    """Return the configured TradeManager API token if present."""

    token = os.getenv("TRADE_MANAGER_TOKEN")
    if token is None:
        return None
    token = token.strip()
    return token or None


def extract_request_token(headers: Mapping[str, str]) -> str:
    """Extract the bearer token value from common header names."""

    token = headers.get("Authorization", "")
    if token.lower().startswith("bearer "):
        token = token[7:]
    else:
        token = headers.get("X-API-KEY", token)
    return token.strip()


def validate_token(headers: Mapping[str, str], expected: str | None) -> str | None:
    """Return a rejection reason when a request token is invalid."""

    if not expected:
        return None
    token = extract_request_token(headers)
    if not token:
        return "missing token"
    if not hmac.compare_digest(token, expected):
        return "token mismatch"
    return None


def load_trade_manager_config(path: str = "config.json") -> Any:
    """Load the TradeManager configuration using the shared loader."""

    return load_config(path)


def _allow_offline_mode() -> bool:
    return (
        os.getenv("OFFLINE_MODE") == "1"
        or os.getenv("TEST_MODE") == "1"
        or os.getenv("TRADE_MANAGER_USE_STUB") == "1"
    )


def allow_unauthenticated_requests() -> bool:
    """Return ``True`` when TradeManager may operate without API authentication."""

    # Offline integration tests and stubbed runtime scenarios intentionally skip
    # token validation so the service can be exercised without credentials.
    return os.getenv("OFFLINE_MODE") == "1" or os.getenv("TRADE_MANAGER_USE_STUB") == "1"


def _close_exchange_instance(instance: Any) -> None:
    close_method = getattr(instance, "close", None)
    if callable(close_method):
        close_method()


class ExchangeRuntime:
    """Manage initialization and reuse of a single Bybit exchange instance."""

    def __init__(
        self,
        *,
        service_name: str,
        context_name: str,
        after_create: Callable[[Any], None] | None = None,
    ) -> None:
        self._logger = logging.getLogger(service_name)
        self._context: ContextVar[Any | None] = ContextVar(context_name, default=None)
        self._after_create = after_create
        self._ccxt = self._ensure_ccxt(service_name)
        self.ccxt_base_error = getattr(self._ccxt, "BaseError", Exception)
        self.ccxt_network_error = getattr(
            self._ccxt, "NetworkError", self.ccxt_base_error
        )
        self.ccxt_bad_request = getattr(
            self._ccxt, "BadRequest", self.ccxt_base_error
        )
        self.provider = ExchangeProvider(self._create_exchange, close=_close_exchange_instance)

    @property
    def ccxt(self) -> Any:
        """Expose the loaded ``ccxt`` module for consumers and tests."""

        return self._ccxt

    def _ensure_ccxt(self, service_name: str):
        try:  # optional dependency
            import ccxt  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional in offline mode
            if _allow_offline_mode():
                self._logger.warning(
                    "`ccxt` не найден: %s использует OfflineBybit. "
                    "Для работы с реальной биржей установите `pip install ccxt`.",
                    service_name,
                )
                from services.offline import OfflineBybit

                return SimpleNamespace(  # type: ignore[return-value]
                    bybit=OfflineBybit,
                    BaseError=Exception,
                    NetworkError=Exception,
                    BadRequest=Exception,
                )
            self._logger.critical(
                "Библиотека `ccxt` обязательна для %s. Установите её через "
                "`pip install ccxt` или активируйте OFFLINE_MODE=1.",
                service_name,
            )
            raise ImportError(
                "TradeManager не может работать без зависимости `ccxt`."
            ) from exc
        return ccxt

    def _create_exchange(self) -> Any:
        exchange = self._ccxt.bybit(
            {
                "apiKey": os.getenv("BYBIT_API_KEY", ""),
                "secret": os.getenv("BYBIT_API_SECRET", ""),
            }
        )
        if self._after_create is not None:
            self._after_create(exchange)
        return exchange

    def current(self) -> Any | None:
        exchange = self._context.get()
        if exchange is not None:
            return exchange
        cached = self.provider.peek()
        if cached is not None:
            self._context.set(cached)
        return cached

    def bind(self) -> Any:
        exchange = self.provider.get()
        self._context.set(exchange)
        return exchange

    def init(self) -> Any:
        try:
            exchange = self.provider.get()
        except Exception as exc:  # pragma: no cover - config errors
            logging.exception("Failed to initialize Bybit client: %s", exc)
            raise RuntimeError("Invalid Bybit configuration") from exc
        self._context.set(exchange)
        return exchange

    def reset_context(self) -> None:
        self._context.set(None)
