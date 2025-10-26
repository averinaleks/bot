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

    __all__ = [
        "TradeManager",
        "TelegramLogger",
        "service",
        "order_utils",
        "server_common",
        "errors",
    ]
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
        "order_utils",
        "server_common",
        "errors",
    ]


_LEGACY_EXPORTS: dict[str, str] = {
    name: f"{__name__}.{name}"
    for name in ("service", "order_utils", "server_common", "errors")
}


def __getattr__(name: str) -> Any:
    """Ленивая подгрузка вспомогательных модулей.

    Ранее потребители обращались к ``bot.trade_manager`` как к пространству
    имён, содержащему вложенные модули вроде ``order_utils``.  Чтобы сохранить
    обратную совместимость и не тянуть тяжёлые зависимости при импорте,
    модули загружаются по требованию.
    """

    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LEGACY_EXPORTS))
