"""Shared exceptions used by the trade manager package."""

from __future__ import annotations


class TradeManagerTaskError(RuntimeError):
    """Raised when one of the TradeManager background tasks fails."""

    pass


__all__ = ["TradeManagerTaskError"]

