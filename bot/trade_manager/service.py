"""Flask routes for the TradeManager service."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import sys
import threading
from typing import Any

from flask import Flask, Response, jsonify, request

from . import server_common
from .core import (
    _HOSTNAME_RE,
    InvalidHostError,
    logger,
    setup_multiprocessing,
)

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

# For simple logging/testing of received orders
POSITIONS: list[dict[str, Any]] = []

# Local mirror of the shared TradeManager instance for backward compatibility
trade_manager: Any | None = None

# Determine test mode via shared module
IS_TEST_MODE = server_common.IS_TEST_MODE


def _sync_trade_manager(value: Any | None) -> None:
    """Synchronize local and shared trade manager references."""

    global trade_manager
    trade_manager = value
    server_common.set_trade_manager(value)


def _load_env() -> Any:
    """Load environment variables via shared helper."""

    return server_common.load_environment()


def _require_token() -> tuple[Any, int] | None:
    """Validate Authorization header unless running in test mode."""

    error = server_common.require_token(request.headers)
    if error is None:
        return None
    return jsonify({"error": "unauthorized"}), error.status_code


async def create_trade_manager(config_path: str = "config.json") -> Any:
    """Instantiate the TradeManager using the shared implementation."""

    tm = await server_common.create_trade_manager(config_path)
    if tm is not None:
        _sync_trade_manager(tm)
    return tm


async def _initialize_from_loop(loop: asyncio.AbstractEventLoop) -> None:
    tm = await create_trade_manager()
    if tm is not None:
        loop.create_task(tm.run())
    _ready_event.set()


def _initialize_trade_manager() -> None:
    """Background initialization for the TradeManager."""

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_initialize_from_loop(loop))
        tm = server_common.get_trade_manager()
        if tm is not None:
            loop.run_forever()
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
    if not _ready_event.is_set():
        return jsonify({"error": "not ready"}), 503
    tm = server_common.get_trade_manager()
    if tm is None:
        return jsonify({"error": "not ready"}), 503
    info = request.get_json(force=True)
    POSITIONS.append(info)
    symbol = info.get("symbol")
    side = info.get("side")
    price = float(info.get("price", 0))
    if getattr(tm, "loop", None):
        tm.loop.call_soon_threadsafe(
            asyncio.create_task,
            tm.open_position(symbol, side, price, info),
        )
    else:
        return jsonify({"error": "loop not running"}), 503
    return jsonify({"status": "ok"})


@api_app.route("/close_position", methods=["POST"])
def close_position_route():
    err = _require_token()
    if err:
        return err
    if not _ready_event.is_set():
        return jsonify({"error": "not ready"}), 503
    tm = server_common.get_trade_manager()
    if tm is None:
        return jsonify({"error": "not ready"}), 503
    info = request.get_json(force=True)
    symbol = info.get("symbol")
    price = float(info.get("price", 0))
    reason = info.get("reason", "")
    if getattr(tm, "loop", None):
        tm.loop.call_soon_threadsafe(
            asyncio.create_task,
            tm.close_position(symbol, price, reason),
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
    tm = server_common.get_trade_manager()
    if not _ready_event.is_set() or tm is None:
        return jsonify({"error": "not ready"}), 503
    stats = tm.get_stats()
    return jsonify({"stats": stats})


@api_app.route("/start")
def start_route():
    tm = server_common.get_trade_manager()
    if not _ready_event.is_set() or tm is None:
        return jsonify({"error": "not ready"}), 503
    if getattr(tm, "loop", None):
        tm.loop.call_soon_threadsafe(
            asyncio.create_task,
            tm.run(),
        )
        return jsonify({"status": "started"})
    return jsonify({"error": "loop not running"}), 503


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@api_app.route("/ready")
def ready() -> Response | tuple[Response, int]:
    """Return 200 once the TradeManager is initialized."""

    tm = server_common.get_trade_manager()
    if _ready_event.is_set() and tm is not None:
        return jsonify({"status": "ok"})
    return jsonify({"status": "initializing"}), 503


def _resolve_host() -> str:
    """Получить безопасный адрес привязки сервиса."""

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

    server_common.configure_service_environment()
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


def __getattr__(name: str) -> Any:
    if name == "trade_manager":
        return server_common.get_trade_manager()
    raise AttributeError(name)


def __setattr__(name: str, value: Any) -> None:
    if name == "trade_manager":
        server_common.set_trade_manager(value)
    else:
        globals()[name] = value


if __name__ == "__main__":
    main()
