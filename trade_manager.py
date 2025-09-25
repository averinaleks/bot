"""Compatibility wrapper for the TradeManager service."""

# Import configuration explicitly from the local package to avoid accidentally
# pulling in unrelated modules named ``config``.
from bot.config import OFFLINE_MODE

if OFFLINE_MODE:
    from services.offline import OfflineBybit as TradeManager, OfflineTelegram as TelegramLogger
else:  # pragma: no cover - real implementation
    from bot.utils import TelegramLogger  # noqa: F401  # re-export for test injection
    from bot.trade_manager import (  # noqa: F401
        TradeManager,
        api_app,
        asgi_app,
        create_trade_manager,
    )
    from bot.trade_manager.service import (  # noqa: F401
        InvalidHostError,
        POSITIONS,
        _ready_event,
        _resolve_host,
        main,
        trade_manager,
    )

if __name__ == "__main__":  # pragma: no cover - manual execution
    if OFFLINE_MODE:
        print("Offline mode: trade manager not started")
    else:
        from bot.trade_manager.service import main

        main()
