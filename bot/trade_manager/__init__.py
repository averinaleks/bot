"""Публичный интерфейс пакета :mod:`bot.trade_manager`.

Модуль экспортирует согласованный набор объектов как для полноценного режима,
так и для офлайн-запуска. Благодаря этому потребителям достаточно написать
``from bot.trade_manager import TradeManager`` без дополнительных манипуляций с
``sys.path`` или ``noqa``.
"""

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

    __all__ = ["TradeManager", "TelegramLogger", "service", "order_utils", "server_common"]
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

    # Удаляем подмодуль из пространства имён пакета, чтобы доступ к нему
    # проходил через ``__getattr__``. Это гарантирует, что повторный импорт
    # после манипуляций с ``sys.modules`` в тестах всегда создаёт корректную
    # привязку и не вызывает ``ImportError`` при ``importlib.reload``.
    globals().pop("service", None)

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
        "order_utils",
        "server_common",
    ]


_LAZY_SUBMODULES: dict[str, str] = {
    "service": "service",
    "order_utils": "order_utils",
    "server_common": "server_common",
}


def __getattr__(name: str) -> Any:
    """Ленивая загрузка вспомогательных подмодулей пакета."""

    target = _LAZY_SUBMODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.{target}")
    return module
