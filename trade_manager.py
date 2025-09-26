"""Обёртка совместимости для устаревшего импорта ``import trade_manager``.

Новый код должен использовать ``from bot.trade_manager import TradeManager``.
"""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "`import trade_manager` устарел, используйте `from bot.trade_manager import TradeManager`",
    DeprecationWarning,
    stacklevel=2,
)

_tm = importlib.import_module("bot.trade_manager")
# Reload the package so that dynamic stubs injected into ``bot.utils`` during
# tests are respected even when ``bot.trade_manager`` was previously imported.
_tm = importlib.reload(_tm)

from bot.trade_manager import *  # noqa: F401,F403
