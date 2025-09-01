"""Data handler package exposing main interfaces."""

from .core import DataHandler
from .api import api_app
from .storage import DEFAULT_PRICE
from .utils import atr_fast
from bot.http_client import (
    get_async_http_client as get_http_client,
    close_async_http_client as close_http_client,
)

__all__ = [
    "DataHandler",
    "api_app",
    "DEFAULT_PRICE",
    "atr_fast",
    "get_http_client",
    "close_http_client",
]
