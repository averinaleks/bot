"""Обёртка совместимости для устаревшего импорта ``import trade_manager``.

Новый код должен использовать ``from bot.trade_manager import TradeManager``.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType
from typing import Iterable

MODULE_NAME = "bot.trade_manager"


def _load_trade_manager_module() -> ModuleType:
    if MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[MODULE_NAME])
    return importlib.import_module(MODULE_NAME)


def _exported_names(module: ModuleType) -> Iterable[str]:
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return tuple(exported)
    return tuple(name for name in vars(module) if not name.startswith("_"))


warnings.warn(
    "`import trade_manager` устарел, используйте `from bot.trade_manager import TradeManager`",
    DeprecationWarning,
    stacklevel=2,
)

_trade_manager_module = _load_trade_manager_module()
__all__ = list(_exported_names(_trade_manager_module))
globals().update({name: getattr(_trade_manager_module, name) for name in __all__})
