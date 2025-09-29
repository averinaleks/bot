"""Shared error types for the TradeManager package."""

from __future__ import annotations

__all__ = ["TradeManagerTaskError", "InvalidHostError"]


class TradeManagerTaskError(RuntimeError):
    """Raised when one of the TradeManager background tasks fails."""


class InvalidHostError(ValueError):
    """Raised when the configured host is unsafe for binding."""
