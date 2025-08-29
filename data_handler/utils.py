"""Utility helpers for data handler."""
import pandas as pd


def expected_ws_rate(timeframe: str) -> int:
    """Return minimal websocket processing rate for timeframe."""
    seconds = pd.Timedelta(timeframe).total_seconds()
    return max(1, int(1800 / seconds))
