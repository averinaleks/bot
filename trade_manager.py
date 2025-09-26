"""Обёртка совместимости для устаревшего импорта ``import trade_manager``.

Новый код должен использовать ``from bot.trade_manager import TradeManager``.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings

warnings.warn(
    "`import trade_manager` устарел, используйте `from bot.trade_manager import TradeManager`",
    DeprecationWarning,
    stacklevel=2,
)

if os.getenv("TEST_MODE") == "1":
    sys.modules.pop("bot.trade_manager", None)

importlib.import_module("bot.trade_manager")
from bot.trade_manager import *  # noqa: F401,F403
