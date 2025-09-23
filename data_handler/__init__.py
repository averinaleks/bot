from collections.abc import Iterable

import numpy as np

from .core import DataHandler
try:  # pragma: no cover - optional dependency
    from .api import api_app
except Exception:  # pragma: no cover - Flask not installed
    api_app = None  # type: ignore[assignment]
from .storage import DEFAULT_PRICE


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
]
