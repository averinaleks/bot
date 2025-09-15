"""Compatibility helpers for optional python-dotenv dependency."""
from __future__ import annotations

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly in tests
    from dotenv import load_dotenv as _load_dotenv
    from dotenv import dotenv_values as _dotenv_values
except Exception as exc:  # pragma: no cover - fallback when dependency missing
    DOTENV_AVAILABLE = False
    DOTENV_ERROR: Exception | None = exc

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Gracefully skip loading .env files when python-dotenv is absent."""
        logger.debug("python-dotenv is not installed: %s", DOTENV_ERROR)
        return False

    def dotenv_values(*args: Any, **kwargs: Any) -> Dict[str, str | None]:
        """Return an empty mapping when python-dotenv is not installed."""
        logger.debug("python-dotenv is not installed: %s", DOTENV_ERROR)
        return {}
else:  # pragma: no cover - exercised indirectly in tests
    DOTENV_AVAILABLE = True
    DOTENV_ERROR = None

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Proxy to :func:`dotenv.load_dotenv`."""
        return bool(_load_dotenv(*args, **kwargs))

    def dotenv_values(*args: Any, **kwargs: Any) -> Dict[str, str | None]:
        """Proxy to :func:`dotenv.dotenv_values`."""
        return dict(_dotenv_values(*args, **kwargs))

__all__ = ["load_dotenv", "dotenv_values", "DOTENV_AVAILABLE", "DOTENV_ERROR"]
