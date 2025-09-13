"""Compatibility wrapper for the TradeManager service."""

from bot.utils import TelegramLogger  # re-export for test injection
from bot.trade_manager.service import *  # noqa: F401,F403
from bot.trade_manager.core import TradeManager

if __name__ == "__main__":  # pragma: no cover - manual execution
    from bot.trade_manager.service import main

    main()
