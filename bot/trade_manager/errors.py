"""Shared exception types for the trade manager package."""

class TradeManagerTaskError(RuntimeError):
    """Raised when one of the TradeManager background tasks fails."""

__all__ = ["TradeManagerTaskError"]
