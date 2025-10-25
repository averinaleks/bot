
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from typing import Any

from .storage import DEFAULT_PRICE

try:  # Optional dependency
    import numpy as _NUMPY  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment without numpy
    _NUMPY = None


def _is_offline_mode() -> bool:
    """Best-effort detection of OFFLINE_MODE without importing heavy deps."""

    try:  # Local import avoids module-level side effects during tests
        from bot import config as bot_config
    except Exception:  # pragma: no cover - configuration import errors
        return False
    return bool(getattr(bot_config, "OFFLINE_MODE", False))


_OFFLINE_REQUESTED = _is_offline_mode()
_CORE_DATA_HANDLER: type[Any] | None = None
if _NUMPY is not None and not _OFFLINE_REQUESTED:
    try:
        from .core import DataHandler as _CORE_DATA_HANDLER  # type: ignore[assignment]
    except Exception:  # pragma: no cover - pandas or other deps missing
        _CORE_DATA_HANDLER = None

if _CORE_DATA_HANDLER is None:
    from .offline import OfflineDataHandler as DataHandler  # noqa: F401
    np = None  # type: ignore[assignment]
else:
    DataHandler = _CORE_DATA_HANDLER
    np = _NUMPY

try:  # pragma: no cover - optional dependency
    from .api import api_app
except Exception:  # pragma: no cover - Flask not installed
    api_app = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class DataHandlerSettings:
    """Settings returned by :func:`get_settings`."""

    symbols: list[str]
    config: Any | None = None


def _load_bot_config() -> Any | None:
    """Best-effort loading of :class:`bot.config.BotConfig`."""

    try:  # Deferred import keeps data_handler usable without full config
        from bot import config as bot_config
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
) -> Any:
    """Compute a simple moving-average based ATR without requiring numpy."""

    highs = [float(value) for value in high]
    lows = [float(value) for value in low]
    closes = [float(value) for value in close]
    length = min(len(highs), len(lows), len(closes))
    if length == 0:
        return [] if np is None else np.asarray([], dtype=float)

    true_range: list[float] = []
    for index in range(length):
        if index == 0:
            true_range.append(highs[0] - lows[0])
            continue
        upper = max(highs[index], closes[index - 1])
        lower = min(lows[index], closes[index - 1])
        true_range.append(upper - lower)

    atr_values: list[float] = []
    window = max(1, int(period))
    for index in range(len(true_range)):
        start = max(0, index - window + 1)
        segment = true_range[start : index + 1]
        atr_values.append(sum(segment) / len(segment))

    if np is not None:
        return np.asarray(atr_values, dtype=float)
    return atr_values


__all__ = [
    "DataHandler",
    "api_app",
    "DEFAULT_PRICE",
    "DataHandlerSettings",
    "get_settings",
]
