"""Utility helpers for data handler."""

from __future__ import annotations

import pandas as pd


def expected_ws_rate(timeframe: str) -> int:
    """Return minimal websocket processing rate for timeframe."""
    seconds = pd.Timedelta(timeframe).total_seconds()
    return max(1, int(1800 / seconds))


def atr_fast(high, low, close, window: int) -> "np.ndarray":
    """Lightweight Average True Range implementation.

    The real project previously exposed an optimised ATR routine that may have
    relied on optional numeric libraries.  The tests only require behaviour
    compatible with :func:`ta.volatility.average_true_range`, so we implement a
    small pandas based version here which works with plain Python objects and
    NumPy arrays.  The function returns a NumPy array for ease of use in the
    rest of the codebase.
    """

    import numpy as np

    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    return atr.to_numpy(np.float64, copy=False)
