"""Public interface for :mod:`bot.trade_manager`.

This module re-exports the main entry points of the trade manager package while
mapping the asynchronous HTTP client helpers to simple ``get``/``close``
aliases.  The previous implementation imported ``get_http_client`` and
``close_http_client`` from ``core`` and then redefined them with the async
variants, which triggered a redefinition warning from the linter (F811).  To
avoid that, we only import :class:`TradeManager` from ``core`` and explicitly
alias the asynchronous helpers.
"""

from .core import TradeManager
from .service import api_app, asgi_app, create_trade_manager
from bot.http_client import get_async_http_client, close_async_http_client

# Provide the expected public names for HTTP helpers
get_http_client = get_async_http_client
close_http_client = close_async_http_client

__all__ = [
    "TradeManager",
    "api_app",
    "asgi_app",
    "create_trade_manager",
    "get_http_client",
    "close_http_client",
]
