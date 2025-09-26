"""Публичный интерфейс пакета :mod:`bot.trade_manager`.

Модуль экспортирует согласованный набор объектов как для полноценного режима,
так и для офлайн-запуска. Благодаря этому потребителям достаточно написать
``from bot.trade_manager import TradeManager`` без дополнительных манипуляций с
``sys.path`` или ``noqa``.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from bot.config import OFFLINE_MODE

if OFFLINE_MODE:
    from services.offline import OfflineBybit as TradeManager
    from services.offline import OfflineTelegram as _OfflineTelegram

    __all__ = ["TradeManager", "TelegramLogger"]

    def __getattr__(name: str) -> Any:  # pragma: no cover - simple passthrough
        if name == "TelegramLogger":
            return _OfflineTelegram
        raise AttributeError(name)
else:  # pragma: no cover - реальная инициализация
    from bot.http_client import close_async_http_client, get_async_http_client

    from .core import TradeManager
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
    ]

    def __getattr__(name: str) -> Any:
        if name == "TelegramLogger":
            utils = sys.modules.get("utils")
            if utils is None:
                utils = importlib.import_module("utils")
            return utils.TelegramLogger
        raise AttributeError(name)
