"""Обёртка совместимости для устаревшего импорта ``import trade_manager``.

Новый код должен использовать ``from bot.trade_manager import TradeManager``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`import trade_manager` устарел, используйте `from bot.trade_manager import TradeManager`",
    DeprecationWarning,
    stacklevel=2,
)

from bot.trade_manager import *  # noqa: F401,F403
