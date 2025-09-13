# pylint: disable=unused-import
"""Flask routes for the TradeManager service."""

from __future__ import annotations

import asyncio
import atexit
import inspect
import ipaddress
import json
import os
import signal
import sys
import threading
import types
from typing import Any

import httpx
import ray
from dotenv import load_dotenv
from flask import Flask, jsonify, request

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
from bot.config import load_config


__all__ = [
    "api_app",
    "asgi_app",
    "create_trade_manager",
    "_ready_event",
    "trade_manager",
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
    asgi_app = api_app.asgi_app
except AttributeError:  # pragma: no cover - older Flask versions
    try:
        from a2wsgi import WSGIMiddleware  # type: ignore
    except ImportError as exc:  # pragma: no cover - fallback if a2wsgi isn't installed
        logger.exception("Не удалось импортировать a2wsgi (%s): %s", type(exc).__name__, exc)
        from uvicorn.middleware.wsgi import WSGIMiddleware

    asgi_app = WSGIMiddleware(api_app)

# Track when the TradeManager initialization finishes
_ready_event = threading.Event()

# For simple logging/testing of received orders
POSITIONS = []

trade_manager: TradeManager | None = None

# Determine test mode at import time, considering both environment and core flag
IS_TEST_MODE = os.getenv("TEST_MODE") == "1" or CORE_TEST_MODE

TRADE_MANAGER_TOKEN = os.getenv("TRADE_MANAGER_TOKEN")
if not TRADE_MANAGER_TOKEN:
    logger.warning("TRADE_MANAGER_TOKEN не установлен")


def _require_token() -> tuple[Any, int] | None:
    """Validate Authorization header unless running in test mode."""
    if IS_TEST_MODE:
        return None
    if not TRADE_MANAGER_TOKEN:
        return jsonify({"error": "unauthorized"}), 401
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {TRADE_MANAGER_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401
    return None


async def create_trade_manager() -> TradeManager | None:
    """Instantiate the TradeManager using config.json."""
    global trade_manager
    if trade_manager is None:
        logger.info("Загрузка конфигурации из config.json")
        try:
            cfg = load_config("config.json")
            logger.info("Конфигурация успешно загружена")
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception(
                "Failed to load configuration (%s): %s", type(exc).__name__, exc
            )
            raise
        if not ray.is_initialized():
            logger.info(
                "Инициализация Ray: num_cpus=%s, num_gpus=1", cfg["ray_num_cpus"]
            )
            try:
                ray.init(
                    num_cpus=cfg["ray_num_cpus"],
                    num_gpus=1,
                    ignore_reinit_error=True,
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

        trade_manager = TradeManager(cfg, dh, mb, telegram_bot, chat_id)
        logger.info("Экземпляр TradeManager создан")
        if telegram_bot:
            from bot.utils import TelegramUpdateListener

            listener = TelegramUpdateListener(telegram_bot)

            async def handle_command(update):
                msg = getattr(update, "message", None)
                if not msg or not msg.text:
                    return
                text = msg.text.strip().lower()
                import trading_bot as tb

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
                    if trade_manager is not None:
                        try:
                            res = trade_manager.get_open_positions()
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
            trade_manager._listener = listener
        if not IS_TEST_MODE:
            _register_cleanup_handlers(trade_manager)
    return trade_manager

def _initialize_trade_manager() -> None:
    """Background initialization for the TradeManager."""
    global trade_manager
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trade_manager = loop.run_until_complete(create_trade_manager())
        if trade_manager is not None:
            loop.create_task(trade_manager.run())
            _ready_event.set()
            loop.run_forever()
        else:
            _ready_event.set()
    except (RuntimeError, ValueError) as exc:
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





@api_app.route("/open_position", methods=["POST"])
def open_position_route():
    """Open a new trade position."""
    err = _require_token()
    if err:
        return err
    if not _ready_event.is_set() or trade_manager is None:
        return jsonify({"error": "not ready"}), 503
    info = request.get_json(force=True)
    POSITIONS.append(info)
    symbol = info.get("symbol")
    side = info.get("side")
    price = float(info.get("price", 0))
    if getattr(trade_manager, "loop", None):
        trade_manager.loop.call_soon_threadsafe(
            asyncio.create_task,
            trade_manager.open_position(symbol, side, price, info),
        )
    else:
        return jsonify({"error": "loop not running"}), 503
    return jsonify({"status": "ok"})


@api_app.route("/close_position", methods=["POST"])
def close_position_route():
    err = _require_token()
    if err:
        return err
    if not _ready_event.is_set() or trade_manager is None:
        return jsonify({"error": "not ready"}), 503
    info = request.get_json(force=True)
    symbol = info.get("symbol")
    price = float(info.get("price", 0))
    reason = info.get("reason", "")
    if getattr(trade_manager, "loop", None):
        trade_manager.loop.call_soon_threadsafe(
            asyncio.create_task,
            trade_manager.close_position(symbol, price, reason),
        )
    else:
        return jsonify({"error": "loop not running"}), 503
    return jsonify({"status": "ok"})


@api_app.route("/positions")
def positions_route():
    err = _require_token()
    if err:
        return err
    return jsonify({"positions": POSITIONS})


@api_app.route("/stats")
def stats_route():
    if not _ready_event.is_set() or trade_manager is None:
        return jsonify({"error": "not ready"}), 503
    stats = trade_manager.get_stats()
    return jsonify({"stats": stats})


@api_app.route("/start")
def start_route():
    if not _ready_event.is_set() or trade_manager is None:
        return jsonify({"error": "not ready"}), 503
    if getattr(trade_manager, "loop", None):
        trade_manager.loop.call_soon_threadsafe(
            asyncio.create_task,
            trade_manager.run(),
        )
        return jsonify({"status": "started"})
    return jsonify({"error": "loop not running"}), 503


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@api_app.route("/ready")
def ready() -> tuple:
    """Return 200 once the TradeManager is initialized."""
    if _ready_event.is_set() and trade_manager is not None:
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
    load_dotenv()
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
