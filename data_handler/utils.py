"""Utility helpers for data handler."""
import pandas as pd


def expected_ws_rate(timeframe: str) -> int:
    """Return minimal websocket processing rate for timeframe."""
    seconds = pd.Timedelta(timeframe).total_seconds()
    return max(1, int(1800 / seconds))


def ensure_utc(ts: pd.Series | pd.Index) -> pd.Series | pd.Index:
    """Convert timestamp series or index to UTC timezone.

    Вход может быть ``pd.Series`` или ``pd.Index``. Если временная зона
    отсутствует, применяется цепочка ``tz_localize('UTC').tz_convert('UTC')``.
    При наличии временной зоны выполняется только ``tz_convert('UTC')``.
    """
    if isinstance(ts, pd.Series):
        dt = pd.to_datetime(ts)
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC").dt.tz_convert("UTC")
        else:  # pragma: no branch
            dt = dt.dt.tz_convert("UTC")
        return dt
    dt = pd.to_datetime(ts)
    if dt.tz is None:
        dt = dt.tz_localize("UTC").tz_convert("UTC")
    else:  # pragma: no branch
        dt = dt.tz_convert("UTC")
    return dt

