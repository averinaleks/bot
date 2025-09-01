"""Utility helpers for data handler."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_ws_rate(timeframe: str) -> int:
    """Return minimal websocket processing rate for timeframe."""
    seconds = pd.Timedelta(timeframe).total_seconds()
    return max(1, int(1800 / seconds))


def atr_fast(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Compute Average True Range using a simple moving average.

    Parameters
    ----------
    high, low, close:
        Arrays of high, low and close prices.
    period:
        Window size for the ATR calculation.

    Returns
    -------
    numpy.ndarray
        Array of ATR values matching the input length.
    """

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    tr = np.empty_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        prev_close = close[i - 1]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )

    atr = np.empty_like(tr)
    for i in range(len(tr)):
        start = max(0, i - period + 1)
        atr[i] = tr[start : i + 1].mean()
    return atr
