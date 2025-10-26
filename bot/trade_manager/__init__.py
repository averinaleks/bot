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
def __getattr__(name: str) -> Any:  # pragma: no cover - thin forwarding logic
    """Ленивая загрузка подмодулей ``bot.trade_manager``.

    Функция используется для совместимости с существующим кодом, который
    обращается к ``bot.trade_manager.service`` и другим подмодулям напрямую из
    пакета.  В прежней версии файла определение функции оказалось случайно
    удалено, из-за чего интерпретатор доходил до тела функции без заголовка и
    выбрасывал :class:`IndentationError` ещё на этапе импорта.  Это ломало
    запуск тестов и любое импортирование пакета.  Возвращаем корректное
    определение и оставляем минимальную реализацию: сначала пытаемся найти
    атрибут среди уже загруженных объектов, а при отсутствии — импортируем
    одноимённый подмодуль.
    """

    if name in globals():
        return globals()[name]

    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - исключительная ветка
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    globals()[name] = module
    return module
