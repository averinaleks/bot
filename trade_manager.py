"""Compatibility shim exposing :mod:`bot.trade_manager` as a top-level module."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

from bot.utils_loader import require_utils

_target: ModuleType = importlib.import_module("bot.trade_manager")

try:
    _utils = require_utils("TelegramLogger")
except Exception:  # pragma: no cover - fallback when utils unavailable
    _utils = None
else:
    setattr(_target, "TelegramLogger", _utils.TelegramLogger)
    sys.modules.setdefault("utils", _utils)
    sys.modules.setdefault("bot.utils", _utils)

__all__ = getattr(_target, "__all__", [])

for _name in __all__:
    globals()[_name] = getattr(_target, _name)


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    return getattr(_target, name)


def __dir__() -> list[str]:  # pragma: no cover - keep interactive help useful
    return sorted(set(__all__) | set(dir(_target)))
