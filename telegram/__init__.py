"""Заглушка для пакета :mod:`telegram`.

Нужна только для подавления импортов в журналировании TelegramLogger.
"""
from __future__ import annotations

from .error import BadRequest, Forbidden, RetryAfter, TelegramError

__all__ = ["TelegramError", "BadRequest", "Forbidden", "RetryAfter"]
