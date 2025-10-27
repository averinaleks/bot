"""Публичный интерфейс пакета :mod:`bot.trade_manager`.

Модуль экспортирует согласованный набор объектов как для полноценного режима,
так и для офлайн-запуска. Благодаря этому потребителям достаточно написать
``from bot.trade_manager import TradeManager`` без дополнительных манипуляций с
``sys.path`` или ``noqa``.
"""

import importlib
from typing import TYPE_CHECKING, Any, List, cast

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
    TelegramLogger: type[TelegramLoggerType] = _utils.TelegramLogger

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

    TradeManager: type[TradeManagerType] = _TradeManager

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

def __getattr__(name: str) -> Any:
    """Ленивая загрузка дополнительных атрибутов пакета."""

    if name == "service":
        module = importlib.import_module(".service", package=__name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - стабильно, зависит от __all__
    return sorted(__all__)
