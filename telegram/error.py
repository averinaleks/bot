"""Минимальный набор исключений для совместимости с telegram_logger."""
from __future__ import annotations


class TelegramError(Exception):
    """Базовое исключение заглушки Telegram."""


class BadRequest(TelegramError):
    """Ошибочный запрос."""


class Forbidden(TelegramError):
    """Доступ запрещён."""


class RetryAfter(TelegramError):
    """Сигнализирует, что нужно повторить попытку позже."""

    def __init__(self, message: str | None = None, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


__all__ = ["TelegramError", "BadRequest", "Forbidden", "RetryAfter"]
