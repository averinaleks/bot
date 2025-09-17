from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .core import DataHandler
try:  # pragma: no cover - optional dependency kept for backward compatibility
    from .api import api_app
except Exception:  # pragma: no cover - Flask not installed
    api_app = None  # type: ignore[assignment]
from .storage import DEFAULT_PRICE
from bot import http_client as _http_client

_TA_SPEC = importlib.util.find_spec("ta")
ta = importlib.import_module("ta") if _TA_SPEC is not None else None


async def get_http_client():
    """Expose the shared async HTTP client used across the project."""

    return await _http_client.get_async_http_client()


async def close_http_client() -> None:
    """Close the shared async HTTP client if it exists."""

    await _http_client.close_async_http_client()


def ema_fast(values: Iterable[float], period: int) -> np.ndarray:
    """Vectorised exponential moving average used in tests."""

    if period <= 0:
        raise ValueError("period must be positive")
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return data.astype(float)
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, data.size):
        ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def atr_fast(
    high: Iterable[float],
    low: Iterable[float],
    close: Iterable[float],
    period: int,
) -> np.ndarray:
    """Compute a simple moving-average based ATR for compatibility tests."""

    if period <= 0:
        raise ValueError("period must be positive")
    h = np.asarray(list(high), dtype=float)
    l = np.asarray(list(low), dtype=float)
    c = np.asarray(list(close), dtype=float)
    if h.size == 0:
        return np.asarray([], dtype=float)
    tr = _true_range(h, l, c)
    if period <= 2:
        atr = np.empty_like(tr)
        atr[0] = tr[0]
        for i in range(1, tr.size):
            start = max(0, i - period + 1)
            atr[i] = tr[start : i + 1].mean()
        return atr
    atr = np.zeros_like(tr)
    if tr.size < period:
        cumulative = 0.0
        for i, value in enumerate(tr):
            cumulative += value
            atr[i] = cumulative / (i + 1)
        return atr
    atr[: period - 1] = 0.0
    atr[period - 1] = tr[:period].mean()
    for i in range(period, tr.size):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.empty_like(close, dtype=float)
    if close.size == 0:
        return tr
    tr[0] = high[0] - low[0]
    if close.size > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum(high[1:], prev_close) - np.minimum(low[1:], prev_close)
    return tr


def _wilder_smooth(values: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(values.size, dtype=float)
    if values.size == 0:
        return result
    result[0] = values[0]
    for i in range(1, values.size):
        if i < window:
            result[i] = (result[i - 1] * i + values[i]) / (i + 1)
        else:
            result[i] = (result[i - 1] * (window - 1) + values[i]) / window
    return result


class IndicatorsCache:
    """Lightweight cache for frequently used indicators in tests."""

    def __init__(self, df: pd.DataFrame, cfg: Any, min_volume: float) -> None:
        self.cfg = cfg
        self.min_volume = float(min_volume)
        self.volume_profile_update_interval = int(
            getattr(cfg, "volume_profile_update_interval", 0) or 0
        )
        self.ema30_period = max(1, int(getattr(cfg, "ema30_period", 30)))
        self.ema100_period = max(1, int(getattr(cfg, "ema100_period", 100)))
        self.ema200_period = max(1, int(getattr(cfg, "ema200_period", 200)))
        self.atr_period = max(1, int(getattr(cfg, "atr_period_default", 14)))
        self.rsi_window = max(1, int(getattr(cfg, "rsi_window", 14)))
        self.adx_window = max(1, int(getattr(cfg, "adx_window", 14)))
        self._rsi_avg_gain: float | None = None
        self._rsi_avg_loss: float | None = None
        self._dm_plus: float = 0.0
        self._dm_minus: float = 0.0
        self.last_adx: float = float("nan")
        self.last_rsi: float = 0.0
        self.last_atr: float = 0.0
        self.last_ema30: float = 0.0
        self.last_ema100: float = 0.0
        self.last_ema200: float = 0.0
        self.last_close: float = float("nan")
        self.last_high: float = float("nan")
        self.last_low: float = float("nan")
        self.volume_profile: pd.Series | None = None
        self.rsi = pd.Series(dtype=float)
        self.adx = pd.Series(dtype=float)
        self.atr = pd.Series(dtype=float)
        self.df = df.copy()
        self.df = self.df.sort_index()
        self.df = self.df.astype(float, copy=False)
        self._recompute_indicators()

    def update(self, new_rows: pd.DataFrame) -> None:
        if new_rows.empty:
            return
        combined = pd.concat([self.df, new_rows])
        combined = combined[~combined.index.duplicated(keep="last")]
        self.df = combined.sort_index()
        self.df = self.df.astype(float, copy=False)
        self._recompute_indicators()

    def _recompute_indicators(self) -> None:
        if self.df.empty:
            self.rsi = pd.Series(dtype=float)
            self.adx = pd.Series(dtype=float)
            self.atr = pd.Series(dtype=float)
            self.volume_profile = None
            self.last_adx = float("nan")
            self.last_rsi = 0.0
            self.last_atr = 0.0
            self.last_ema30 = 0.0
            self.last_ema100 = 0.0
            self.last_ema200 = 0.0
            self.last_close = float("nan")
            self.last_high = float("nan")
            self.last_low = float("nan")
            self._rsi_avg_gain = None
            self._rsi_avg_loss = None
            self._dm_plus = 0.0
            self._dm_minus = 0.0
            return

        close = self.df["close"].astype(float)
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        volume = self.df.get("volume", pd.Series(0.0, index=self.df.index, dtype=float))
        volume = volume.astype(float)

        ema30 = close.ewm(span=self.ema30_period, adjust=False).mean()
        ema100 = close.ewm(span=self.ema100_period, adjust=False).mean()
        ema200 = close.ewm(span=self.ema200_period, adjust=False).mean()
        self.df["ema30"] = ema30
        self.df["ema100"] = ema100
        self.df["ema200"] = ema200
        self.last_ema30 = float(ema30.iloc[-1])
        self.last_ema100 = float(ema100.iloc[-1])
        self.last_ema200 = float(ema200.iloc[-1])

        tr = _true_range(high.to_numpy(), low.to_numpy(), close.to_numpy())
        atr_values = _wilder_smooth(tr, self.atr_period)
        self.atr = pd.Series(atr_values, index=self.df.index)
        self.df["atr"] = self.atr
        self.last_atr = float(self.atr.iloc[-1])

        diff = close.diff().fillna(0.0).to_numpy()
        gains = np.maximum(diff, 0.0)
        losses = np.maximum(-diff, 0.0)

        fallback = False
        if ta is not None:
            try:
                ta.momentum.rsi(close, window=self.rsi_window, fillna=True)
            except Exception:
                fallback = True

        if fallback:
            rsi_values = np.zeros(close.size, dtype=float)
            self._rsi_avg_gain = None
            self._rsi_avg_loss = None
        else:
            avg_gain = _wilder_smooth(gains, self.rsi_window)
            avg_loss = _wilder_smooth(losses, self.rsi_window)
            self._rsi_avg_gain = float(avg_gain[-1]) if avg_gain.size else None
            self._rsi_avg_loss = float(avg_loss[-1]) if avg_loss.size else None
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = avg_gain / avg_loss
            rs = np.where(
                (avg_loss == 0.0) & (avg_gain == 0.0), 0.0, rs
            )
            rs = np.where((avg_loss == 0.0) & (avg_gain > 0.0), np.inf, rs)
            rsi_values = 100.0 - 100.0 / (1.0 + rs)
            rsi_values = np.nan_to_num(
                rsi_values, nan=0.0, posinf=100.0, neginf=0.0
            )
        self.rsi = pd.Series(rsi_values, index=self.df.index)
        self.df["rsi"] = self.rsi
        self.last_rsi = float(self.rsi.iloc[-1])

        plus_dm = np.zeros(close.size, dtype=float)
        minus_dm = np.zeros(close.size, dtype=float)
        for i in range(1, close.size):
            up_move = high.iloc[i] - high.iloc[i - 1]
            down_move = low.iloc[i - 1] - low.iloc[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        dm_plus_series = _wilder_smooth(plus_dm, self.adx_window)
        dm_minus_series = _wilder_smooth(minus_dm, self.adx_window)
        self._dm_plus = float(dm_plus_series[-1]) if dm_plus_series.size else 0.0
        self._dm_minus = float(dm_minus_series[-1]) if dm_minus_series.size else 0.0

        tr_sum = self.atr.to_numpy() * self.adx_window
        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = np.divide(
                dm_plus_series,
                tr_sum,
                out=np.zeros_like(dm_plus_series),
                where=tr_sum != 0,
            ) * 100.0
            minus_di = np.divide(
                dm_minus_series,
                tr_sum,
                out=np.zeros_like(dm_minus_series),
                where=tr_sum != 0,
            ) * 100.0
        denom = plus_di + minus_di
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.divide(
                np.abs(plus_di - minus_di),
                denom,
                out=np.zeros_like(denom),
                where=denom != 0,
            ) * 100.0
        adx_values = _wilder_smooth(dx, self.adx_window)
        if adx_values.size <= self.adx_window:
            adx_values[:] = np.nan
        else:
            adx_values[: self.adx_window - 1] = np.nan
        self.adx = pd.Series(adx_values, index=self.df.index)
        self.df["adx"] = self.adx
        self.last_adx = float(self.adx.iloc[-1])

        boll_window = max(1, int(getattr(self.cfg, "bollinger_window", 20)))
        rolling_mean = close.rolling(window=boll_window, min_periods=1).mean()
        rolling_std = close.rolling(window=boll_window, min_periods=1).std(ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            wband = (4.0 * rolling_std) / rolling_mean.replace(0.0, np.nan)
        self.df["bollinger_wband"] = wband.fillna(0.0)

        ulcer_window = max(1, int(getattr(self.cfg, "ulcer_window", 14)))
        rolling_max = close.rolling(window=ulcer_window, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max.replace(0.0, np.nan)
        ulcer = (
            drawdown.pow(2)
            .rolling(window=ulcer_window, min_periods=ulcer_window)
            .mean()
            .pow(0.5)
        )
        self.df["ulcer_index"] = ulcer.fillna(0.0)

        if (
            self.volume_profile_update_interval
            and len(self.df) >= self.volume_profile_update_interval
        ):
            tail = volume.tail(self.volume_profile_update_interval)
            self.volume_profile = tail.copy()
        else:
            self.volume_profile = None

        self.last_close = float(close.iloc[-1])
        self.last_high = float(high.iloc[-1])
        self.last_low = float(low.iloc[-1])


__all__ = [
    "DataHandler",
    "IndicatorsCache",
    "api_app",
    "DEFAULT_PRICE",
    "get_http_client",
    "close_http_client",
    "ema_fast",
    "atr_fast",
]
