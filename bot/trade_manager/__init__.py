"""Публичный интерфейс пакета :mod:`bot.trade_manager`.

Модуль экспортирует согласованный набор объектов как для полноценного режима,
так и для офлайн-запуска. Благодаря этому потребителям достаточно написать
``from bot.trade_manager import TradeManager`` без дополнительных манипуляций с
``sys.path`` или ``noqa``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

from bot import config as bot_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import TradeManager as TradeManagerType
    from telegram_logger import TelegramLogger as TelegramLoggerType
else:
    TradeManagerType = Any
    TelegramLoggerType = Any

if bot_config.OFFLINE_MODE:
    from services.offline import OfflineTradeManager, OfflineTelegram

    TradeManager = cast(type[TradeManagerType], OfflineTradeManager)
    TelegramLogger = cast(type[TelegramLoggerType], OfflineTelegram)

    __all__ = ["TradeManager", "TelegramLogger", "service"]
else:  # pragma: no cover - реальная инициализация
    from bot.http_client import close_async_http_client, get_async_http_client
    from bot.utils_loader import require_utils

    _utils = require_utils("TelegramLogger")
    TelegramLogger = cast(type[TelegramLoggerType], _utils.TelegramLogger)

    from .core import TradeManager as _TradeManager
    from .service import (
        InvalidHostError,
        api_app,
        asgi_app,
        create_trade_manager,
        trade_manager,
        main,
        _resolve_host,
        _ready_event,
    )

    TradeManager = cast(type[TradeManagerType], _TradeManager)

    # Псевдонимы синхронных помощников оставлены для обратной совместимости
    get_http_client = get_async_http_client
    close_http_client = close_async_http_client

    __all__ = [
        "TradeManager",
        "TelegramLogger",
        "api_app",
        "asgi_app",
        "create_trade_manager",
        "trade_manager",
        "main",
        "InvalidHostError",
        "_resolve_host",
        "_ready_event",
        "get_http_client",
        "close_http_client",
        "service",
    ]

_LAZY_SUBMODULES = {
    "service": "bot.trade_manager.service",
    "order_utils": "bot.trade_manager.order_utils",
    "server_common": "bot.trade_manager.server_common",
    "errors": "bot.trade_manager.errors",
    "core": "bot.trade_manager.core",
}


def _load_submodule(name: str) -> Any:
    """Import *name* as a submodule of :mod:`bot.trade_manager`.

    The helper mirrors the behaviour of packages that expose selected
    submodules at the top level.  Keeping the logic centralised makes it easy
    to reuse from ``__getattr__`` and allows tests to patch ``sys.modules`` to
    simulate missing dependencies without fighting the import cache.
    """

    module = importlib.import_module(_LAZY_SUBMODULES[name])
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - упражнение инфраструктуры импорта
    if name in _LAZY_SUBMODULES:
        return _load_submodule(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - минимальная обвязка для introspection
    return sorted(set(__all__) | set(_LAZY_SUBMODULES))
