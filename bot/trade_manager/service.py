# pylint: disable=unused-import
"""Flask routes for the TradeManager service."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import math
import os
import sys
import threading
from typing import Any, Awaitable, Mapping, TypeVar, cast

import httpx
from bot.ray_compat import ray
from flask import Flask, Response, jsonify, request

from services.logging_utils import sanitize_log_value

from .core import (
    IS_TEST_MODE as CORE_TEST_MODE,
    TradeManager,
    _HOSTNAME_RE,
    _register_cleanup_handlers,
    InvalidHostError,
    configure_logging,
    logger,
    setup_multiprocessing,
)
from . import server_common




__all__ = [
    "api_app",
    "asgi_app",
    "create_trade_manager",
    "trade_manager",
    "_ready_event",
    "get_trade_manager",
    "_resolve_host",
    "InvalidHostError",
    "main",
]

# ----------------------------------------------------------------------
# REST API for minimal integration testing
# ----------------------------------------------------------------------

api_app = Flask(__name__)

# Expose an ASGI-compatible application so Gunicorn's UvicornWorker can run
# this Flask app without raising "Flask.__call__() missing start_response".
try:  # Flask 2.2+ provides ``asgi_app`` for native ASGI support
    asgi_app = api_app.asgi_app  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - older Flask versions
    try:
        from a2wsgi import WSGIMiddleware as A2WSGIMiddleware  # type: ignore
    except ImportError as exc:  # pragma: no cover - fallback if a2wsgi isn't installed
        logger.exception("Не удалось импортировать a2wsgi (%s): %s", type(exc).__name__, exc)
        from uvicorn.middleware.wsgi import WSGIMiddleware as UvicornWSGIMiddleware

        asgi_app = UvicornWSGIMiddleware(api_app)  # type: ignore[arg-type]
    else:
        asgi_app = A2WSGIMiddleware(api_app)  # type: ignore[arg-type]

# Track when the TradeManager initialization finishes
_ready_event = threading.Event()


_T = TypeVar("_T")


class TradeManagerFactory:
    """Manage lifecycle of a singleton :class:`TradeManager` instance."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._instance: TradeManager | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> TradeManager | None:
        with self._lock:
            return self._instance

    def get_loop(self) -> asyncio.AbstractEventLoop | None:
        with self._lock:
            return self._loop

    def set(self, instance: TradeManager | None, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        with self._lock:
            self._instance = instance
            if loop is not None or instance is None:
                self._loop = loop

    def reset(self) -> None:
        self.set(None, loop=None)


trade_manager_factory = TradeManagerFactory()


def get_trade_manager() -> TradeManager | None:
    """Return the current :class:`TradeManager` instance if available."""

    return trade_manager_factory.get()


def trade_manager() -> TradeManager | None:
    """Backwards compatible alias returning the active manager."""

    return trade_manager_factory.get()


def _manager_with_loop() -> tuple[TradeManager | None, asyncio.AbstractEventLoop | None]:
    manager = trade_manager_factory.get()
    loop = trade_manager_factory.get_loop()
    if loop is None and manager is not None:
        loop = getattr(manager, "loop", None)
    return manager, loop


def _await_manager_result(
    loop: asyncio.AbstractEventLoop, awaitable: Awaitable[_T], timeout: float | None = 10.0
) -> _T:
    """Wait for an awaitable to finish on the TradeManager loop."""

    future = asyncio.run_coroutine_threadsafe(awaitable, loop)
    try:
        return future.result(timeout)
    except Exception:
        future.cancel()
        raise

# Determine test mode at import time, considering both environment and core flag
IS_TEST_MODE = os.getenv("TEST_MODE") == "1" or CORE_TEST_MODE

TRADE_MANAGER_TOKEN: str | None = server_common.get_api_token()


def _load_env() -> None:
    """Load `.env` and validate required environment variables."""
    global TRADE_MANAGER_TOKEN
    server_common.load_environment()
    TRADE_MANAGER_TOKEN = server_common.get_api_token()
    if not TRADE_MANAGER_TOKEN:
        logger.warning(
            "TRADE_MANAGER_TOKEN пуст, все торговые запросы будут отвергнуты"
        )


def _require_token() -> tuple[Any, int] | None:
    """Validate Authorization header unless running in test mode."""
    if IS_TEST_MODE:
        return None
    if not TRADE_MANAGER_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    header_mapping: Mapping[str, str] = cast(
        Mapping[str, str], {key: value for key, value in request.headers.items()}
    )
    reason = server_common.validate_token(header_mapping, TRADE_MANAGER_TOKEN)
    if reason is None:
        return None
    return jsonify({"error": "unauthorized"}), 401


async def create_trade_manager() -> TradeManager | None:
    """Instantiate the TradeManager using config.json."""
    existing = trade_manager_factory.get()
    if existing is not None:
        return existing
    logger.info("Загрузка конфигурации из config.json")
    try:
        cfg = server_common.load_trade_manager_config("config.json")
        logger.info("Конфигурация успешно загружена")
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception(
            "Failed to load configuration (%s): %s", type(exc).__name__, exc
        )
        raise
    if not ray.is_initialized():
        from security import apply_ray_security_defaults

        logger.info(
            "Инициализация Ray: num_cpus=%s, num_gpus=1", cfg["ray_num_cpus"]
        )
        try:
            ray.init(
                **apply_ray_security_defaults(
                    {
                        "num_cpus": cfg["ray_num_cpus"],
                        "num_gpus": 1,
                        "ignore_reinit_error": True,
                    }
                )
            )
            logger.info("Ray успешно инициализирован")
        except RuntimeError as exc:
            logger.exception(
                "Ray initialization failed (%s): %s", type(exc).__name__, exc
            )
            raise
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    telegram_bot = None
    if token:
        try:
            from telegram import Bot

            telegram_bot = Bot(token)
            try:
                await telegram_bot.delete_webhook(drop_pending_updates=True)
                logger.info("Удалён существующий Telegram webhook")
            except httpx.HTTPError as exc:  # pragma: no cover - delete_webhook errors
                logger.exception(
                    "Failed to delete Telegram webhook (%s): %s",
                    type(exc).__name__,
                    exc,
                )
        except (RuntimeError, httpx.HTTPError) as exc:  # pragma: no cover - import/runtime errors
            logger.exception(
                "Не удалось создать Telegram Bot (%s): %s",
                type(exc).__name__,
                exc,
            )
            raise
    from bot.data_handler import DataHandler
    from bot.model_builder import ModelBuilder

    logger.info("Создание DataHandler")
    try:
        dh = DataHandler(cfg, telegram_bot, chat_id)
        logger.info("DataHandler успешно создан")
    except RuntimeError as exc:
        logger.exception(
            "Не удалось создать DataHandler (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise

    logger.info("Создание ModelBuilder")
    try:
        mb = ModelBuilder(cfg, dh, None)
        dh.feature_callback = mb.precompute_features
        logger.info("ModelBuilder успешно создан")
        asyncio.create_task(mb.train())
        asyncio.create_task(mb.backtest_loop())
        await dh.load_initial()
        asyncio.create_task(dh.subscribe_to_klines(dh.usdt_pairs))
    except RuntimeError as exc:
        logger.error("Не удалось загрузить исходные данные: %s", exc)
        await dh.stop()
        return None
    except (ValueError, ImportError) as exc:
        logger.exception(
            "Не удалось создать ModelBuilder (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise

    manager = TradeManager(cfg, dh, mb, telegram_bot, chat_id)
    logger.info("Экземпляр TradeManager создан")
    if telegram_bot:
        from bot.utils_loader import require_utils

        TelegramUpdateListener = require_utils("TelegramUpdateListener").TelegramUpdateListener

        listener = TelegramUpdateListener(telegram_bot)

        async def handle_command(update):
            msg = getattr(update, "message", None)
            if not msg or not msg.text:
                return
            text = msg.text.lower()
            tb = manager
            if text.startswith("/start"):
                await tb.set_trading_enabled(True)
                try:
                    await telegram_bot.send_message(
                        chat_id=msg.chat_id, text="Trading enabled"
                    )
                except Exception as exc:
                    logger.error("Не удалось отправить сообщение в Telegram: %s", exc)
            elif text.startswith("/stop"):
                await tb.set_trading_enabled(False)
                try:
                    await telegram_bot.send_message(
                        chat_id=msg.chat_id, text="Trading disabled"
                    )
                except Exception as exc:
                    logger.error("Не удалось отправить сообщение в Telegram: %s", exc)
            elif text.startswith("/status"):
                status = "enabled" if await tb.get_trading_enabled() else "disabled"
                positions = []
                current = trade_manager_factory.get()
                if current is not None:
                    try:
                        res = current.get_open_positions()
                        positions = (
                            await res if inspect.isawaitable(res) else res
                        ) or []
                    except Exception as exc:  # pragma: no cover - log and ignore
                        logger.error("Не удалось получить открытые позиции: %s", exc)
                message = f"Trading {status}"
                if positions:
                    message += "\n" + "\n".join(str(p) for p in positions)
                try:
                    await telegram_bot.send_message(chat_id=msg.chat_id, text=message)
                except Exception as exc:
                    logger.error("Не удалось отправить сообщение в Telegram: %s", exc)

        threading.Thread(
            target=lambda: asyncio.run(listener.listen(handle_command)),
            daemon=True,
        ).start()
        setattr(manager, "_listener", listener)
    if not IS_TEST_MODE:
        _register_cleanup_handlers(manager)
    trade_manager_factory.set(manager)
    return manager
def _initialize_trade_manager() -> None:
    """Background initialization for the TradeManager."""

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        manager = loop.run_until_complete(create_trade_manager())
        if manager is not None:
            trade_manager_factory.set(manager, loop=loop)
            loop.create_task(manager.run())
            _ready_event.set()
            loop.run_forever()
        else:
            _ready_event.set()
    except (RuntimeError, ValueError) as exc:
        trade_manager_factory.reset()
        logger.exception(
            "TradeManager initialization failed (%s): %s",
            type(exc).__name__,
            exc,
        )
        _ready_event.set()
        raise


# Set ready event immediately in test mode
if IS_TEST_MODE:
    _ready_event.set()


_startup_launched = False


@api_app.before_request
def _start_trade_manager() -> None:
    """Launch trade manager initialization in a background thread."""
    global _startup_launched
    if _startup_launched or IS_TEST_MODE:
        return
    _startup_launched = True
    threading.Thread(target=_initialize_trade_manager, daemon=True).start()





class ValidationError(ValueError):
    """Raised when request payload validation fails."""


def _validate_payload_is_object(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("payload must be a JSON object")
    return data


def _validate_symbol(info: Mapping[str, Any]) -> str:
    symbol = info.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValidationError("symbol is required")
    return symbol.strip()


def _validate_side(info: Mapping[str, Any]) -> str:
    side = info.get("side")
    if not isinstance(side, str) or side.lower() not in {"buy", "sell"}:
        raise ValidationError("side must be either 'buy' or 'sell'")
    return side.lower()


def _validate_price(info: Mapping[str, Any]) -> float:
    raw_price = info.get("price")
    if raw_price is None:
        raise ValidationError("price is required")
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        raise ValidationError("price must be a number") from None
    if not math.isfinite(price) or price <= 0:
        raise ValidationError("price must be a positive finite number")
    return price


def _validate_open_position(info: Any) -> tuple[str, str, float, dict[str, Any]]:
    data = _validate_payload_is_object(info)
    symbol = _validate_symbol(data)
    side = _validate_side(data)
    price = _validate_price(data)
    return symbol, side, price, data


def _validate_close_position(info: Any) -> tuple[str, float, dict[str, Any]]:
    data = _validate_payload_is_object(info)
    symbol = _validate_symbol(data)
    price = _validate_price(data)
    return symbol, price, data


def _validation_error_response(exc: ValidationError) -> tuple[Response, int]:
    logger.info(
        "Некорректный запрос к TradeManager: %s",
        sanitize_log_value(str(exc)),
    )
    return jsonify({"error": "invalid request"}), 400


@api_app.route("/open_position", methods=["POST"])
def open_position_route():
    """Open a new trade position."""
    err = _require_token()
    if err:
        return err
    manager, loop = _manager_with_loop()
    if not _ready_event.is_set() or manager is None or loop is None:
        return jsonify({"error": "not ready"}), 503
    try:
        symbol, side, price, info = _validate_open_position(
            request.get_json(force=True, silent=True)
        )
    except ValidationError as exc:
        return _validation_error_response(exc)
    loop.call_soon_threadsafe(
        asyncio.create_task,
        manager.open_position(symbol, side, price, info),
    )
    return jsonify({"status": "ok"})


@api_app.route("/close_position", methods=["POST"])
def close_position_route():
    err = _require_token()
    if err:
        return err
    manager, loop = _manager_with_loop()
    if not _ready_event.is_set() or manager is None or loop is None:
        return jsonify({"error": "not ready"}), 503
    try:
        symbol, price, info = _validate_close_position(
            request.get_json(force=True, silent=True)
        )
    except ValidationError as exc:
        return _validation_error_response(exc)
    reason = info.get("reason", "")
    loop.call_soon_threadsafe(
        asyncio.create_task,
        manager.close_position(symbol, price, reason),
    )
    return jsonify({"status": "ok"})


@api_app.route("/positions")
def positions_route():
    err = _require_token()
    if err:
        return err
    manager, loop = _manager_with_loop()
    if not _ready_event.is_set() or manager is None or loop is None:
        return jsonify({"error": "not ready"}), 503
    try:
        positions = _await_manager_result(loop, manager.get_positions_snapshot())
    except Exception:
        logger.exception("Не удалось получить позиции")
        return jsonify({"error": "internal error"}), 500
    return jsonify({"positions": positions})


@api_app.route("/stats")
def stats_route():
    manager, _ = _manager_with_loop()
    if not _ready_event.is_set() or manager is None:
        return jsonify({"error": "not ready"}), 503
    stats = manager.get_stats()
    return jsonify({"stats": stats})


@api_app.route("/start")
def start_route():
    manager, loop = _manager_with_loop()
    if not _ready_event.is_set() or manager is None or loop is None:
        return jsonify({"error": "not ready"}), 503
    loop.call_soon_threadsafe(asyncio.create_task, manager.run())
    return jsonify({"status": "started"})


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@api_app.route("/ready")
def ready() -> Response | tuple[Response, int]:
    """Return 200 once the TradeManager is initialized."""
    manager, _ = _manager_with_loop()
    if _ready_event.is_set() and manager is not None:
        return jsonify({"status": "ok"})
    return jsonify({"status": "initializing"}), 503




def _resolve_host() -> str:
    """Получить безопасный адрес привязки сервиса.

    Если переменная окружения ``HOST`` не задана, используется ``127.0.0.1`` и
    выводится предупреждение. Запуск на всех интерфейсах (``0.0.0.0`` или ``::``)
    запрещён без явной конфигурации.
    """

    host_env = os.getenv("HOST")
    if not host_env:
        logger.warning(
            "HOST не установлен, используется 127.0.0.1. Укажите HOST для внешнего доступа",
        )
        return "127.0.0.1"

    host_env = host_env.strip()
    if host_env.lower() == "localhost":
        return "127.0.0.1"

    try:
        ip = ipaddress.ip_address(host_env)
    except ValueError:
        if not _HOSTNAME_RE.fullmatch(host_env):
            raise InvalidHostError(f"Некорректное значение HOST {host_env}")
        raise InvalidHostError(f"Недопустимый хост {host_env}: разрешены только локальные адреса")

    if ip.is_unspecified or ip.compressed != "127.0.0.1":
        raise InvalidHostError(
            f"Недопустимый адрес {host_env}: разрешён только 127.0.0.1"
        )
    return "127.0.0.1"

def main() -> None:
    """Entry point for running the service as a script."""
    configure_logging()
    setup_multiprocessing()
    _load_env()
    try:
        host = _resolve_host()
    except InvalidHostError as exc:  # pragma: no cover - configuration errors
        logger.error("Ошибка конфигурации HOST: %s", exc)
        sys.exit(1)
    port = int(os.getenv("PORT", "8002"))
    logger.info("Запуск сервиса TradeManager на %s:%s", host, port)
    api_app.run(host=host, port=port)


if __name__ == "__main__":
    main()
