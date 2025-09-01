"""Data handler package exposing main interfaces.

The project historically exposed a couple of utility functions at the package
level.  Some tests import ``bot.data_handler`` directly and expect to find
helpers like ``get_http_client`` or fast indicator implementations there.  The
library's internals were refactored and these exports disappeared which caused
runtime errors in the test-suite.  To maintain backwards compatibility we now
re-export the HTTP client helpers and provide a small, pure Python
implementation of ``atr_fast`` used in unit tests.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from bot import test_stubs

test_stubs.apply()

from .core import DataHandler
from .api import api_app
from .storage import DEFAULT_PRICE
from bot import http_client as _http_client


async def get_http_client():
    """Expose the shared async HTTP client used across the project."""

    return await _http_client.get_async_http_client()


async def close_http_client() -> None:
    """Close the shared async HTTP client if it exists."""

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

    h = np.asarray(list(high), dtype=float)
    l = np.asarray(list(low), dtype=float)
    c = np.asarray(list(close), dtype=float)
    tr = np.empty_like(c)
    tr[0] = h[0] - l[0]
    prev_close = c[:-1]
    tr[1:] = np.maximum(h[1:], prev_close) - np.minimum(l[1:], prev_close)
    atr = np.empty_like(tr)
    for i in range(len(tr)):
        start = max(0, i - period + 1)
        atr[i] = tr[start : i + 1].mean()
    return atr


__all__ = [
    "DataHandler",
    "api_app",
    "DEFAULT_PRICE",
    "get_http_client",
    "close_http_client",
    "atr_fast",
]
