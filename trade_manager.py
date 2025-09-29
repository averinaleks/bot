"""Compatibility wrapper exposing :mod:`bot.trade_manager` under legacy import."""

from __future__ import annotations

import importlib
from typing import Any

from bot.utils_loader import require_utils


def _load_trade_manager_module():
    return importlib.import_module("bot.trade_manager")


def __getattr__(name: str) -> Any:
    if name == "TelegramLogger":
        return require_utils("TelegramLogger").TelegramLogger
    module = _load_trade_manager_module()
    return getattr(module, name)


def __dir__() -> list[str]:  # pragma: no cover - convenience only
    module = _load_trade_manager_module()
    return sorted(set(__all__) | set(dir(module)))


__all__ = ["TradeManager", "TelegramLogger"]
