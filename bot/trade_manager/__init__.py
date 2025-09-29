"""Публичный интерфейс пакета :mod:`bot.trade_manager`.

Модуль экспортирует согласованный набор объектов как для полноценного режима,
так и для офлайн-запуска. Благодаря этому потребителям достаточно написать
``from bot.trade_manager import TradeManager`` без дополнительных манипуляций с
``sys.path`` или ``noqa``.
"""

from typing import TYPE_CHECKING, Any, cast

from bot.config import OFFLINE_MODE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import TradeManager as TradeManagerType, TradeManagerTaskError as TradeManagerTaskErrorType
    from telegram_logger import TelegramLogger as TelegramLoggerType
else:
    TradeManagerType = Any
    TelegramLoggerType = Any
    TradeManagerTaskErrorType = Any

if OFFLINE_MODE:
    from services.offline import OfflineBybit, OfflineTelegram

    TradeManager = cast(type[TradeManagerType], OfflineBybit)
    TelegramLogger = cast(type[TelegramLoggerType], OfflineTelegram)
    TradeManagerTaskError = RuntimeError

    __all__ = ["TradeManager", "TelegramLogger", "TradeManagerTaskError"]
else:  # pragma: no cover - реальная инициализация
    from bot.http_client import close_async_http_client, get_async_http_client
    from bot.utils_loader import require_utils

    _utils = require_utils("TelegramLogger")
    TelegramLogger = cast(type[TelegramLoggerType], _utils.TelegramLogger)

    from .core import TradeManager as _TradeManager, TradeManagerTaskError as _TradeManagerTaskError
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
    TradeManagerTaskError = cast(type[TradeManagerTaskErrorType], _TradeManagerTaskError)

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
        "TradeManagerTaskError",
    ]
