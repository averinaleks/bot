"""Обёртка совместимости для устаревшего импорта ``import trade_manager``.

Новый код должен использовать ``from bot.trade_manager import TradeManager``.
"""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "`import trade_manager` устарел, используйте `from bot.trade_manager import TradeManager`",
    DeprecationWarning,
    stacklevel=2,
)

if "bot.trade_manager" in sys.modules:
    importlib.reload(sys.modules["bot.trade_manager"])

from bot.trade_manager import *  # noqa: F401,F403
