from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from typing import Any

import numpy as np

from .core import DataHandler
try:  # pragma: no cover - optional dependency
    from .api import api_app
except Exception:  # pragma: no cover - Flask not installed
    api_app = None  # type: ignore[assignment]
from .storage import DEFAULT_PRICE


@dataclass(frozen=True, slots=True)
class DataHandlerSettings:
    """Settings returned by :func:`get_settings`."""

    symbols: list[str]
    config: Any | None = None


def _load_bot_config() -> Any | None:
    """Best-effort loading of :class:`bot.config.BotConfig`."""

    try:  # Deferred import keeps data_handler usable without full config
        from bot import config as bot_config  # noqa: WPS433 - local import is intentional
    except Exception:  # pragma: no cover - configuration import errors
        return None

    try:
        return bot_config.load_config()
    except Exception:  # pragma: no cover - fall back to direct instantiation
        try:
            return bot_config.BotConfig()
        except Exception:  # pragma: no cover - give up and use defaults
            return None


def _parse_symbols(raw: str | None, fallback: list[str]) -> list[str]:
    """Parse *raw* into a list of trading symbols."""

    if raw is None:
        return list(fallback)

    raw = raw.strip()
    if not raw:
        return list(fallback)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        candidates = [item.strip() for item in raw.split(",")]
    else:
        if isinstance(parsed, str):
            candidates = [parsed.strip()]
        elif isinstance(parsed, list):
            candidates = [str(item).strip() for item in parsed]
        else:
            candidates = []

    symbols = [candidate for candidate in candidates if candidate]
    return symbols or list(fallback)


def get_settings() -> DataHandlerSettings:
    """Load data handler settings from environment/configuration."""

    config = _load_bot_config()
    default_symbols = ["BTCUSDT"]
    if config is not None:
        configured = getattr(config, "symbols", None)
        if isinstance(configured, list) and configured:
            default_symbols = [str(symbol).strip() for symbol in configured if str(symbol).strip()]

    symbols = _parse_symbols(os.getenv("SYMBOLS"), default_symbols)
    return DataHandlerSettings(symbols=list(symbols), config=config)


async def get_http_client():
    """Expose the shared async HTTP client used across the project."""

    from bot import http_client as _http_client  # Local import avoids import side-effects

    return await _http_client.get_async_http_client()


async def close_http_client() -> None:
    """Close the shared async HTTP client if it exists."""

    from bot import http_client as _http_client  # Imported on demand to prevent env validation on import

    await _http_client.close_async_http_client()


def atr_fast(
    high: Iterable[float],
    low: Iterable[float],
    close: Iterable[float],
    period: int,
) -> np.ndarray:
    """Compute a simple moving-average based ATR.

    This implementation is intentionally lightweight and only supports the
    features required by the tests.  ``high``, ``low`` and ``close`` should be
    index-aligned iterables of equal length.
    """

    high_array = np.asarray(list(high), dtype=float)
    low_array = np.asarray(list(low), dtype=float)
    close_array = np.asarray(list(close), dtype=float)
    true_range = np.empty_like(close_array)
    true_range[0] = high_array[0] - low_array[0]
    prev_close = close_array[:-1]
    true_range[1:] = np.maximum(high_array[1:], prev_close) - np.minimum(
        low_array[1:], prev_close
    )
    atr = np.empty_like(true_range)
    for index in range(len(true_range)):
        start = max(0, index - period + 1)
        atr[index] = true_range[start : index + 1].mean()
    return atr


__all__ = [
    "DataHandler",
    "api_app",
    "DEFAULT_PRICE",
    "DataHandlerSettings",
    "get_settings",
]
