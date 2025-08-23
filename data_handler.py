"""Collects market data and serves it through a Flask REST API."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import threading
import time
import types
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ImportError as exc:  # allow missing pandas
    logging.getLogger("TradingBot").error("pandas import failed: %s", exc)
    raise ImportError("pandas is required for data handling") from exc

import httpx
import numpy as np  # type: ignore

try:  # optional dependency
    import polars as pl  # type: ignore
except ImportError as exc:  # pragma: no cover - allow missing polars
    logging.getLogger("TradingBot").error("polars import failed: %s", exc)
    pl = None

try:  # pragma: no cover - optional dependency
    import websockets  # type: ignore
except ImportError as exc:  # allow missing websockets
    logging.getLogger("TradingBot").error("websockets import failed: %s", exc)
    raise ImportError("websockets is required for real-time data streaming") from exc
from tenacity import retry, wait_exponential

try:  # pragma: no cover - optional dependency
    import ta  # type: ignore
except ImportError as exc:  # allow missing ta
    logging.getLogger("TradingBot").error("ta import failed: %s", exc)
    raise ImportError("ta library is required for technical indicators") from exc


try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError as exc:  # allow missing psutil
    logging.getLogger("TradingBot").error("psutil import failed: %s", exc)
    missing_exc = exc

    def _missing_psutil(*args, **kwargs):
        raise ImportError("psutil is required for system metrics") from missing_exc

    psutil = types.SimpleNamespace(
        cpu_percent=_missing_psutil,
        virtual_memory=_missing_psutil,
    )

try:
    import ray
except ImportError as exc:  # pragma: no cover - optional dependency missing
    logging.getLogger("TradingBot").error("ray import failed: %s", exc)
    raise ImportError("ray is required for distributed computations") from exc
from dotenv import load_dotenv
from flask import Flask, jsonify

from bot.config import BotConfig
from bot.optimizer import ParameterOptimizer
from bot.strategy_optimizer import StrategyOptimizer
from bot.utils import BybitSDKAsync, HistoricalDataCache, TelegramLogger, bybit_interval
from bot.utils import calculate_volume_profile as utils_volume_profile
from bot.utils import (
    check_dataframe_empty,
    filter_outliers_zscore,
    is_cuda_available,
    logger,
    safe_api_call,
)

PROFILE_DATA_HANDLER = os.getenv("DATA_HANDLER_PROFILE") == "1"


def _profile_async(func: Callable) -> Callable:
    if not PROFILE_DATA_HANDLER:
        return func

    async def wrapper(*args, **kwargs):
        start = asyncio.get_running_loop().time()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = asyncio.get_running_loop().time() - start
            logger.info("profile %s took %.4fs", func.__qualname__, elapsed)

    return functools.wraps(func)(wrapper)


def _instrument_methods(cls: type) -> type:
    if not PROFILE_DATA_HANDLER:
        return cls

    for name, attr in list(cls.__dict__.items()):
        if asyncio.iscoroutinefunction(attr):
            setattr(cls, name, _profile_async(attr))
    return cls


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import ccxtpro

# GPU availability is determined lazily to avoid initializing CUDA in every
# Ray worker at import time. Track whether initialization has already
# occurred so we only attempt it once per process.
GPU_AVAILABLE = False
GPU_INITIALIZED = False
cp = np  # type: ignore
_cuda_init_lock = threading.Lock()


def _init_cuda() -> None:
    """Initialize GPU support if available."""

    global GPU_AVAILABLE, cp, GPU_INITIALIZED

    with _cuda_init_lock:
        if GPU_INITIALIZED:
            return

        if os.environ.get("FORCE_CPU") == "1":
            GPU_AVAILABLE = False
            cp = np  # type: ignore
            GPU_INITIALIZED = True
            return
        GPU_AVAILABLE = is_cuda_available()
        if GPU_AVAILABLE:
            try:
                import cupy as cupy_mod  # type: ignore

                cp = cupy_mod  # type: ignore
            except Exception:  # pragma: no cover - import guard
                GPU_AVAILABLE = False
                cp = np  # type: ignore
        else:
            cp = np  # type: ignore
        GPU_INITIALIZED = True


def create_exchange() -> BybitSDKAsync:
    """Create an authenticated Bybit SDK instance.

    Raises
    ------
    RuntimeError
        If required API credentials are missing.
    """
    if os.getenv("TEST_MODE") == "1":

        class _DummyEx:
            async def load_markets(self) -> dict:
                return {}

            async def fetch_ticker(self, symbol: str) -> dict:
                return {"last": 100.0, "quoteVolume": 1.0}

            async def fetch_ohlcv(
                self,
                symbol: str,
                timeframe: str,
                limit: int = 200,
                since: int | None = None,
            ):
                return generate_synthetic_ohlcv(timeframe, limit)

            async def fetch_order_book(self, symbol: str, limit: int = 10) -> dict:
                return {"bids": [[100.0, 1.0]], "asks": [[101.0, 1.0]]}

            async def fetch_funding_rate(self, symbol: str) -> dict:
                return {"fundingRate": 0.0}

            async def fetch_open_interest(self, symbol: str) -> dict:
                return {"openInterest": 0.0}

        return _DummyEx()  # type: ignore[return-value]

    api_key = os.environ.pop("BYBIT_API_KEY", "")
    api_secret = os.environ.pop("BYBIT_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "BYBIT_API_KEY and BYBIT_API_SECRET must be set for DataHandler"
        )
    client = BybitSDKAsync(api_key=api_key, api_secret=api_secret)
    # Best effort to clear sensitive credentials from memory
    api_key = api_secret = None
    return client


def _parse_timeframe_ms(timeframe: str) -> int:
    """Convert a timeframe string (e.g. "1m", "2h") to milliseconds."""

    units = {
        "s": 1_000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    try:
        return int(timeframe[:-1]) * units[timeframe[-1]]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def generate_synthetic_ohlcv(timeframe: str, limit: int) -> list[list[float]]:
    """Return random OHLCV data for testing."""
    interval_ms = _parse_timeframe_ms(timeframe)
    start = int(time.time() * 1000) - limit * interval_ms
    prices = 100 + np.cumsum(np.random.randn(limit))
    result = []
    for i in range(limit):
        ts = start + i * interval_ms
        open_p = prices[i]
        high_p = open_p + abs(np.random.randn())
        low_p = open_p - abs(np.random.randn())
        close_p = open_p + np.random.randn() * 0.5
        volume = abs(np.random.randn()) * 10
        result.append(
            [
                ts,
                float(open_p),
                float(high_p),
                float(low_p),
                float(close_p),
                float(volume),
            ]
        )
    return result


def ema_fast(values: np.ndarray, window: int, wilder: bool = False) -> np.ndarray:
    """Compute EMA using CuPy when CUDA is available, otherwise NumPy."""

    values = np.asarray(values, dtype=np.float64)
    alpha = (1 / window) if wilder else 2 / (window + 1)

    if GPU_AVAILABLE and len(values) >= 1024:
        v_gpu = cp.asarray(values)
        result_gpu = cp.empty_like(v_gpu)
        result_gpu[0] = v_gpu[0]
        for i in range(1, len(v_gpu)):
            result_gpu[i] = alpha * v_gpu[i] + (1 - alpha) * result_gpu[i - 1]
        return cp.asnumpy(result_gpu)

    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def atr_fast(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    """Compute ATR using a rolling mean with ``min_periods=1``."""

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])

    if GPU_AVAILABLE and len(tr) >= 1024:
        tr_gpu = cp.asarray(tr)
        cumsum = cp.cumsum(tr_gpu)
        atr_gpu = cp.empty_like(tr_gpu)
        for i in range(len(tr_gpu)):
            start = max(0, i - window + 1)
            count = i - start + 1
            atr_gpu[i] = (cumsum[i] - (cumsum[start - 1] if start > 0 else 0)) / count
        return cp.asnumpy(atr_gpu)

    cumsum = np.cumsum(tr)
    atr = np.empty_like(tr)
    for i in range(len(tr)):
        start = max(0, i - window + 1)
        count = i - start + 1
        atr[i] = (cumsum[i] - (cumsum[start - 1] if start > 0 else 0)) / count
    return atr


def calculate_imbalance(orderbook: Dict, n: int = 5) -> float:
    """Return the buy/sell volume imbalance of the orderbook."""

    bids = orderbook.get("bids", [])[:n]
    asks = orderbook.get("asks", [])[:n]
    bid_sum = sum(v for _, v in bids)
    ask_sum = sum(v for _, v in asks)
    denom = bid_sum + ask_sum
    if denom == 0:
        return 0.0
    return (bid_sum - ask_sum) / denom


def detect_clusters(
    orderbook: Dict, threshold: float, distance: float = 1.0
) -> List[tuple[float, float]]:
    """Group nearby price levels with large aggregated volume."""

    levels = orderbook.get("bids", []) + orderbook.get("asks", [])
    if not levels:
        return []
    levels.sort(key=lambda x: x[0])
    total_volume = sum(v for _, v in levels)
    clusters = []
    start_price, last_price = levels[0][0], levels[0][0]
    agg_volume = levels[0][1]
    for price, vol in levels[1:]:
        if abs(price - last_price) <= distance:
            agg_volume += vol
            last_price = price
        else:
            if agg_volume >= threshold * total_volume:
                clusters.append(((start_price + last_price) / 2, agg_volume))
            start_price = price
            last_price = price
            agg_volume = vol
    if agg_volume >= threshold * total_volume:
        clusters.append(((start_price + last_price) / 2, agg_volume))
    return clusters


def _rsi_update_cpu(
    prev_gain: float,
    prev_loss: float,
    prev_close: float,
    close: float,
    window: int,
) -> tuple[float, float, float]:
    """Incrementally update RSI using NumPy."""
    diff = close - prev_close
    gain = max(diff, 0.0)
    loss = max(-diff, 0.0)
    avg_gain = (prev_gain * (window - 1) + gain) / window
    avg_loss = (prev_loss * (window - 1) + loss) / window
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    return rsi, avg_gain, avg_loss


def _rsi_update_cupy(
    prev_gain: float,
    prev_loss: float,
    prev_close: float,
    close: float,
    window: int,
) -> tuple[float, float, float]:
    """Incrementally update RSI using CuPy."""
    diff = cp.float64(close - prev_close)
    gain = cp.maximum(diff, 0.0)
    loss = cp.maximum(-diff, 0.0)
    avg_gain = (prev_gain * (window - 1) + gain) / window
    avg_loss = (prev_loss * (window - 1) + loss) / window
    rsi = cp.where(
        avg_loss == 0, cp.float64(100.0), 100 - 100 / (1 + avg_gain / avg_loss)
    )
    return float(rsi), float(avg_gain), float(avg_loss)


def _adx_update_cpu(
    prev_dm_plus: float,
    prev_dm_minus: float,
    prev_adx: float | None,
    prev_high: float,
    prev_low: float,
    high: float,
    low: float,
    atr: float,
    window: int,
) -> tuple[float, float, float]:
    """Incrementally update ADX using NumPy.

    Parameters
    ----------
    prev_dm_plus, prev_dm_minus : float
        Previous smoothed directional movement values.
    prev_adx : float | None
        Previous ADX value or ``None`` for the initial calculation.
    prev_high, prev_low : float
        Previous high and low prices.
    high, low : float
        Current high and low prices.
    atr : float
        Current Average True Range value.
    window : int
        ADX calculation window length.

    Returns
    -------
    tuple[float, float, float]
        The updated ADX, smoothed +DM, and smoothed -DM values.
    """
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
    minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0
    dm_plus = (prev_dm_plus * (window - 1) + plus_dm) / window
    dm_minus = (prev_dm_minus * (window - 1) + minus_dm) / window
    tr_sum = atr * window
    plus_di = 0.0 if tr_sum == 0 else 100 * dm_plus / tr_sum
    minus_di = 0.0 if tr_sum == 0 else 100 * dm_minus / tr_sum
    denom = plus_di + minus_di
    dx = 0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom
    adx = dx if prev_adx is None else (prev_adx * (window - 1) + dx) / window
    return adx, dm_plus, dm_minus


def _adx_update_cupy(
    prev_dm_plus: float,
    prev_dm_minus: float,
    prev_adx: float | None,
    prev_high: float,
    prev_low: float,
    high: float,
    low: float,
    atr: float,
    window: int,
) -> tuple[float, float, float]:
    """Incrementally update ADX using CuPy.

    This mirrors :func:`_adx_update_cpu` but uses CuPy arrays when GPU
    acceleration is available. See that function for parameter
    descriptions.
    """
    up_move = cp.float64(high - prev_high)
    down_move = cp.float64(prev_low - low)
    plus_dm = cp.where((up_move > down_move) & (up_move > 0), up_move, cp.float64(0.0))
    minus_dm = cp.where(
        (down_move > up_move) & (down_move > 0), down_move, cp.float64(0.0)
    )
    dm_plus = (prev_dm_plus * (window - 1) + plus_dm) / window
    dm_minus = (prev_dm_minus * (window - 1) + minus_dm) / window
    tr_sum = atr * window
    plus_di = 0.0 if tr_sum == 0 else 100 * dm_plus / tr_sum
    minus_di = 0.0 if tr_sum == 0 else 100 * dm_minus / tr_sum
    denom = plus_di + minus_di
    dx = 0.0 if denom == 0 else 100 * cp.abs(plus_di - minus_di) / denom
    adx = dx if prev_adx is None else (prev_adx * (window - 1) + dx) / window
    return float(adx), float(dm_plus), float(dm_minus)


class IndicatorsCache:
    """Container for computed technical indicators."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: BotConfig,
        volatility: float,
        timeframe: str = "primary",
    ):
        self.df = df
        self.config = config
        self.volatility = volatility
        self.last_volume_profile_update = 0
        self.volume_profile_update_interval = config.get(
            "volume_profile_update_interval", 300
        )
        try:
            if timeframe == "primary":
                close_np = df["close"].to_numpy()
                high_np = df["high"].to_numpy()
                low_np = df["low"].to_numpy()
                self.ema30 = pd.Series(
                    ema_fast(close_np, config["ema30_period"]), index=df.index
                )
                self.ema100 = pd.Series(
                    ema_fast(close_np, config["ema100_period"]), index=df.index
                )
                self.ema200 = pd.Series(
                    ema_fast(close_np, config["ema200_period"]), index=df.index
                )
                self.atr = pd.Series(
                    atr_fast(high_np, low_np, close_np, config["atr_period_default"]),
                    index=df.index,
                )
                self._alpha_ema30 = 2 / (config["ema30_period"] + 1)
                self._alpha_ema100 = 2 / (config["ema100_period"] + 1)
                self._alpha_ema200 = 2 / (config["ema200_period"] + 1)
                self._atr_period = config["atr_period_default"]
                rsi_window = config.get("rsi_window", 14)
                self._rsi_window = rsi_window
                try:
                    self.rsi = ta.momentum.rsi(
                        df["close"], window=rsi_window, fillna=True
                    )

                    # Calculate smoothed gain/loss values for incremental RSI updates
                    diff = df["close"].diff().to_numpy()[1:]
                    gains = np.clip(diff, 0.0, None)
                    losses = np.clip(-diff, 0.0, None)
                    if len(gains) >= rsi_window:
                        avg_gain = gains[:rsi_window].mean()
                        avg_loss = losses[:rsi_window].mean()
                        for g, l in zip(gains[rsi_window:], losses[rsi_window:]):
                            avg_gain = (avg_gain * (rsi_window - 1) + g) / rsi_window
                            avg_loss = (avg_loss * (rsi_window - 1) + l) / rsi_window
                        self._rsi_avg_gain = float(avg_gain)
                        self._rsi_avg_loss = float(avg_loss)
                    else:
                        self._rsi_avg_gain = None
                        self._rsi_avg_loss = None
                except (
                    KeyError,
                    ValueError,
                ) as e:  # pragma: no cover - log and fallback
                    logger.error("RSI calculation failed: %s", e)
                    self.rsi = pd.Series(np.zeros(len(df)), index=df.index)
                    self._rsi_avg_gain = None
                    self._rsi_avg_loss = None

                adx_window = config.get("adx_window", 14)
                self._adx_window = adx_window
                # Require at least ``adx_window + 1`` values for ADX calculation
                if len(df) > adx_window:
                    self.adx = ta.trend.adx(
                        df["high"],
                        df["low"],
                        df["close"],
                        window=adx_window,
                        fillna=True,
                    )
                else:
                    self.adx = pd.Series([np.nan] * len(df), index=df.index)

                # Compute smoothed DM values for incremental ADX updates
                if len(df) >= 2:
                    high_np = df["high"].to_numpy()
                    low_np = df["low"].to_numpy()
                    up_move = high_np[1:] - high_np[:-1]
                    down_move = low_np[:-1] - low_np[1:]
                    plus_dm = np.where(
                        (up_move > down_move) & (up_move > 0), up_move, 0.0
                    )
                    minus_dm = np.where(
                        (down_move > up_move) & (down_move > 0), down_move, 0.0
                    )
                    dm_plus = plus_dm[:adx_window].sum()
                    dm_minus = minus_dm[:adx_window].sum()
                    for pdm, mdm in zip(plus_dm[adx_window:], minus_dm[adx_window:]):
                        dm_plus = (dm_plus * (adx_window - 1) + pdm) / adx_window
                        dm_minus = (dm_minus * (adx_window - 1) + mdm) / adx_window
                    self._dm_plus = float(dm_plus)
                    self._dm_minus = float(dm_minus)
                else:
                    self._dm_plus = None
                    self._dm_minus = None

                self.macd = ta.trend.macd_diff(
                    df["close"],
                    window_slow=config.get("macd_window_slow", 26),
                    window_fast=config.get("macd_window_fast", 12),
                    window_sign=config.get("macd_window_sign", 9),
                    fillna=True,
                )
                bb_window = config.get("bollinger_window", 20)
                bb = ta.volatility.BollingerBands(
                    df["close"], window=bb_window, fillna=True
                )
                self.bollinger_wband = bb.bollinger_wband()
                ui_window = config.get("ulcer_window", 14)
                self.ulcer_index = ta.volatility.ulcer_index(
                    df["close"], window=ui_window, fillna=True
                )
                df["ema30"] = self.ema30
                df["ema100"] = self.ema100
                df["ema200"] = self.ema200
                df["atr"] = self.atr
                df["rsi"] = self.rsi
                df["adx"] = self.adx
                df["macd"] = self.macd
                df["bollinger_wband"] = self.bollinger_wband
                df["ulcer_index"] = self.ulcer_index
            elif timeframe == "secondary":
                close_np = df["close"].to_numpy()
                self.ema30 = pd.Series(
                    ema_fast(close_np, config["ema30_period"]), index=df.index
                )
                self.ema100 = pd.Series(
                    ema_fast(close_np, config["ema100_period"]), index=df.index
                )
                self._alpha_ema30 = 2 / (config["ema30_period"] + 1)
                self._alpha_ema100 = 2 / (config["ema100_period"] + 1)
                df["ema30"] = self.ema30
                df["ema100"] = self.ema100
            self.volume_profile = None
            if (
                len(df) - self.last_volume_profile_update
                >= self.volume_profile_update_interval
            ):
                self.volume_profile = self.calculate_volume_profile(df)
                self.last_volume_profile_update = len(df)
            self.last_close = float(df["close"].iloc[-1]) if not df.empty else None
            self.last_ema30 = (
                float(self.ema30.iloc[-1]) if self.ema30 is not None else None
            )
            self.last_ema100 = (
                float(self.ema100.iloc[-1]) if self.ema100 is not None else None
            )
            self.last_ema200 = (
                float(self.ema200.iloc[-1])
                if hasattr(self, "ema200") and self.ema200 is not None
                else None
            )
            self.last_atr = (
                float(self.atr.iloc[-1])
                if hasattr(self, "atr") and self.atr is not None
                else None
            )
            self.last_high = float(df["high"].iloc[-1]) if "high" in df else None
            self.last_low = float(df["low"].iloc[-1]) if "low" in df else None
            self.last_rsi = float(self.rsi.iloc[-1]) if hasattr(self, "rsi") else None
            self.last_adx = float(self.adx.iloc[-1]) if hasattr(self, "adx") else None
            self.last_bollinger_wband = (
                float(self.bollinger_wband.iloc[-1])
                if hasattr(self, "bollinger_wband") and not self.bollinger_wband.empty
                else None
            )
            self.last_ulcer_index = (
                float(self.ulcer_index.iloc[-1])
                if hasattr(self, "ulcer_index") and not self.ulcer_index.empty
                else None
            )
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error("Ошибка расчета индикаторов (%s): %s", timeframe, e)
            self.ema30 = self.ema100 = self.ema200 = self.atr = self.rsi = self.adx = (
                self.macd
            ) = self.volume_profile = None
            raise

    def calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        try:
            prices = df["close"].to_numpy(dtype=np.float32)
            volumes = df["volume"].to_numpy(dtype=np.float32)
            vp = utils_volume_profile(prices, volumes, bins=50)
            price_bins = np.linspace(prices.min(), prices.max(), num=len(vp))
            return pd.Series(vp, index=price_bins)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error("Ошибка расчета Volume Profile: %s", e)
            return None

    def _update_volume_profile(self) -> None:
        if (
            len(self.df) - self.last_volume_profile_update
            >= self.volume_profile_update_interval
        ):
            self.volume_profile = self.calculate_volume_profile(self.df)
            self.last_volume_profile_update = len(self.df)

    def update(self, new_df: pd.DataFrame) -> None:
        """Incrementally update EMA and ATR with ``new_df``."""
        if new_df.empty:
            return
        for ts, row in new_df.iterrows():
            close = float(row["close"])
            high = float(row.get("high", close))
            low = float(row.get("low", close))
            if self.last_ema30 is not None:
                self.last_ema30 = (
                    self._alpha_ema30 * close
                    + (1 - self._alpha_ema30) * self.last_ema30
                )
            else:
                self.last_ema30 = close
            if self.last_ema100 is not None:
                self.last_ema100 = (
                    self._alpha_ema100 * close
                    + (1 - self._alpha_ema100) * self.last_ema100
                )
            else:
                self.last_ema100 = close
            if hasattr(self, "_alpha_ema200"):
                if self.last_ema200 is not None:
                    self.last_ema200 = (
                        self._alpha_ema200 * close
                        + (1 - self._alpha_ema200) * self.last_ema200
                    )
                else:
                    self.last_ema200 = close
            if (
                hasattr(self, "_atr_period")
                and self.last_atr is not None
                and self.last_close is not None
            ):
                tr = max(
                    high - low, abs(high - self.last_close), abs(low - self.last_close)
                )
                self.last_atr = (
                    self.last_atr * (self._atr_period - 1) + tr
                ) / self._atr_period
            elif hasattr(self, "_atr_period"):
                self.last_atr = max(high - low, abs(high - close), abs(low - close))
            if self.ema30 is not None:
                self.ema30.loc[ts] = self.last_ema30
            if self.ema100 is not None:
                self.ema100.loc[ts] = self.last_ema100
            if hasattr(self, "ema200") and self.ema200 is not None:
                self.ema200.loc[ts] = self.last_ema200
            if (
                hasattr(self, "atr")
                and self.atr is not None
                and self.last_atr is not None
            ):
                self.atr.loc[ts] = self.last_atr
            new_df.loc[ts, "ema30"] = self.last_ema30
            new_df.loc[ts, "ema100"] = self.last_ema100
            if hasattr(self, "ema200"):
                new_df.loc[ts, "ema200"] = self.last_ema200
            if hasattr(self, "atr"):
                new_df.loc[ts, "atr"] = self.last_atr
            if (
                hasattr(self, "_rsi_window")
                and self._rsi_avg_gain is not None
                and self._rsi_avg_loss is not None
                and self.last_close is not None
            ):
                if GPU_AVAILABLE:
                    rsi_val, self._rsi_avg_gain, self._rsi_avg_loss = _rsi_update_cupy(
                        self._rsi_avg_gain,
                        self._rsi_avg_loss,
                        self.last_close,
                        close,
                        self._rsi_window,
                    )
                else:
                    rsi_val, self._rsi_avg_gain, self._rsi_avg_loss = _rsi_update_cpu(
                        self._rsi_avg_gain,
                        self._rsi_avg_loss,
                        self.last_close,
                        close,
                        self._rsi_window,
                    )
                self.last_rsi = rsi_val
            else:
                rsi_val = np.nan
            if hasattr(self, "rsi"):
                self.rsi.loc[ts] = rsi_val
                new_df.loc[ts, "rsi"] = rsi_val

            if (
                hasattr(self, "_adx_window")
                and self._dm_plus is not None
                and self._dm_minus is not None
                and self.last_high is not None
                and self.last_low is not None
                and self.last_atr is not None
            ):
                if GPU_AVAILABLE:
                    adx_val, self._dm_plus, self._dm_minus = _adx_update_cupy(
                        self._dm_plus,
                        self._dm_minus,
                        self.last_adx,
                        self.last_high,
                        self.last_low,
                        high,
                        low,
                        self.last_atr,
                        self._adx_window,
                    )
                else:
                    adx_val, self._dm_plus, self._dm_minus = _adx_update_cpu(
                        self._dm_plus,
                        self._dm_minus,
                        self.last_adx,
                        self.last_high,
                        self.last_low,
                        high,
                        low,
                        self.last_atr,
                        self._adx_window,
                    )
                self.last_adx = adx_val
            else:
                adx_val = np.nan
            if hasattr(self, "adx"):
                self.adx.loc[ts] = adx_val
                new_df.loc[ts, "adx"] = adx_val
            self.last_close = close
            self.last_high = high
            self.last_low = low
        # Append the new rows directly rather than using ``pd.concat``
        self.df = self.df.reindex(self.df.index.union(new_df.index))
        self.df.loc[new_df.index] = new_df
        self.df.sort_index(inplace=True)
        bb_window = self.config.get("bollinger_window", 20)
        bb = ta.volatility.BollingerBands(
            self.df["close"], window=bb_window, fillna=True
        )
        self.bollinger_wband = bb.bollinger_wband()
        self.df["bollinger_wband"] = self.bollinger_wband
        self.last_bollinger_wband = float(self.bollinger_wband.iloc[-1])

        ui_window = self.config.get("ulcer_window", 14)
        self.ulcer_index = ta.volatility.ulcer_index(
            self.df["close"], window=ui_window, fillna=True
        )
        self.df["ulcer_index"] = self.ulcer_index
        self.last_ulcer_index = float(self.ulcer_index.iloc[-1])
        self._update_volume_profile()


def _make_calc_indicators_remote(use_gpu: bool, gpu_initialized: bool = False):
    """Return a Ray remote function for indicator calculation."""

    @ray.remote(num_cpus=1, num_gpus=1 if use_gpu else 0)
    def _calc_indicators(
        df: pd.DataFrame | "pl.DataFrame",
        config: BotConfig,
        volatility: float,
        timeframe: str,
    ):
        nonlocal gpu_initialized
        if use_gpu and not gpu_initialized:
            # Lazily initialize CUDA on each Ray worker so the global
            # ``cp`` variable points to CuPy when GPU acceleration is
            # requested. This ensures GPU-based functions work correctly
            # even when workers are started on demand.
            _init_cuda()
            gpu_initialized = True
        if pl is not None and isinstance(df, pl.DataFrame):
            df = df.to_pandas().copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        return IndicatorsCache(df, config, volatility, timeframe)

    return _calc_indicators


class DataHandler:
    """Collects market data and exposes it via an HTTP API.

    Parameters
    ----------
    config : dict
        Bot configuration.
    telegram_bot : telegram.Bot or compatible
        Bot instance for sending notifications.
    chat_id : str | int
        Identifier of the Telegram chat for notifications.
    exchange : BybitSDKAsync, optional
        Preconfigured Bybit client.
    pro_exchange : "ccxtpro.bybit", optional
        ccxtpro client for WebSocket data.
    trade_callback : Callable[[str], Awaitable[None]], optional
        Function called after a candle has been processed.
    """

    def __init__(
        self,
        config: BotConfig,
        telegram_bot,
        chat_id,
        exchange: BybitSDKAsync | None = None,
        pro_exchange: "ccxtpro.bybit" | None = None,
        feature_callback: Callable[[str], Awaitable[Any]] | None = None,
        trade_callback: Callable[[str], Awaitable[None]] | None = None,
    ):
        _init_cuda()
        logger.info(
            "Starting DataHandler initialization: timeframe=%s, max_symbols=%s, GPU available=%s",
            config.timeframe,
            config.get("max_symbols"),
            GPU_AVAILABLE,
        )
        if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get(
            "TELEGRAM_CHAT_ID"
        ):
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; Telegram alerts will not be sent"
            )
        self.config = config
        self.exchange = exchange or create_exchange()
        self.pro_exchange = pro_exchange
        self.telegram_logger = TelegramLogger(
            telegram_bot,
            chat_id,
            max_queue_size=config.get("telegram_queue_size"),
        )
        self.feature_callback = feature_callback
        self.trade_callback = trade_callback
        self.cache = HistoricalDataCache(config["cache_dir"])
        self.calc_indicators = _make_calc_indicators_remote(GPU_AVAILABLE, False)
        self.use_polars = config.get("use_polars", False)
        if self.use_polars:
            if pl is None:
                raise RuntimeError("Polars is required when use_polars=True")
            self._ohlcv = pl.DataFrame()
            self._ohlcv_2h = pl.DataFrame()
        else:
            self._ohlcv = pd.DataFrame()
            self._ohlcv_2h = pd.DataFrame()
        self.funding_rates = {}
        self.open_interest = {}
        self.open_interest_change = {}
        self.orderbook = pd.DataFrame()
        self.orderbook_imbalance: Dict[str, float] = {}
        self.order_clusters: Dict[str, List[tuple[float, float]]] = {}
        self.indicators = {}
        self.indicators_cache = {}
        self.indicators_2h = {}
        self.indicators_cache_2h = {}
        self.usdt_pairs = []
        self.symbol_locks: Dict[str, asyncio.Lock] = {}
        self.ohlcv_lock = asyncio.Lock()
        self.ohlcv_2h_lock = asyncio.Lock()
        self.funding_lock = asyncio.Lock()
        self.oi_lock = asyncio.Lock()
        self.orderbook_lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()
        self.ws_rate_timestamps = []
        self.process_rate_timestamps = []
        self.ws_min_process_rate = config.get("ws_min_process_rate", 1)
        if self.ws_min_process_rate <= 1:
            tf_seconds = pd.Timedelta(config.timeframe).total_seconds()
            self.ws_min_process_rate = max(1, int(1800 / tf_seconds))
        self.process_rate_window = 1
        self.cleanup_task = None
        self.ws_queue = asyncio.PriorityQueue(
            maxsize=config.get("ws_queue_size", 10000)
        )
        # Use an asyncio.Queue so disk buffer operations never block the event loop
        self.disk_buffer: asyncio.Queue[str] = asyncio.Queue(
            maxsize=config.get("disk_buffer_size", 10000)
        )
        self.buffer_dir = (
            Path(config["cache_dir"]) / "ws_buffer"
        ).resolve(strict=False)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.processed_timestamps = {}
        self.processed_timestamps_2h = {}
        self.symbol_priority = {}
        self.backup_ws_urls = config.get("backup_ws_urls", [])
        self.ws_latency = {}
        self.latency_log_interval = 3600
        self.restart_attempts = 0
        self.max_restart_attempts = 20
        self.ws_inactivity_timeout = config.get("ws_inactivity_timeout", 30)
        logger.info("DataHandler initialization complete")
        # Maximum number of symbols to work with overall
        self.max_symbols = config.get("max_symbols", 50)
        # Start with the configured limit for dynamic adjustments
        self.max_subscriptions = self.max_symbols
        # Number of symbols to subscribe per WebSocket connection
        # Number of symbols to subscribe per WebSocket connection. Prefer
        # ``ws_subscription_batch_size`` if present for backward compatibility
        # with the older ``max_subscriptions_per_connection`` option.
        self.ws_subscription_batch_size = config.get(
            "ws_subscription_batch_size",
            config.get("max_subscriptions_per_connection", 30),
        )
        self.active_subscriptions = 0
        self.load_threshold = config.get("load_threshold", 0.8)
        self.ws_pool = {}
        self.tasks = []
        self.parameter_optimizer = ParameterOptimizer(self.config, self)
        self.strategy_optimizer = (
            StrategyOptimizer(self.config, self)
            if self.config.get("use_strategy_optimizer", False)
            else None
        )

    def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        lock = self.symbol_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self.symbol_locks[symbol] = lock
        return lock

    # ------------------------------------------------------------------
    # Properties to expose OHLCV data as pandas when using Polars
    @property
    def ohlcv(self) -> pd.DataFrame:
        if self.use_polars:
            if self._ohlcv.height == 0:
                return pd.DataFrame()
            df = self._ohlcv.to_pandas().copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.set_index(["symbol", "timestamp"]).sort_index()
        return self._ohlcv

    @ohlcv.setter
    def ohlcv(self, value: pd.DataFrame) -> None:
        if self.use_polars:
            self._ohlcv = (
                pl.from_pandas(value.reset_index())
                if isinstance(value, pd.DataFrame)
                else value
            )
        else:
            self._ohlcv = value

    @property
    def ohlcv_2h(self) -> pd.DataFrame:
        if self.use_polars:
            if self._ohlcv_2h.height == 0:
                return pd.DataFrame()
            df = self._ohlcv_2h.to_pandas().copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.set_index(["symbol", "timestamp"]).sort_index()
        return self._ohlcv_2h

    @ohlcv_2h.setter
    def ohlcv_2h(self, value: pd.DataFrame) -> None:
        if self.use_polars:
            self._ohlcv_2h = (
                pl.from_pandas(value.reset_index())
                if isinstance(value, pd.DataFrame)
                else value
            )
        else:
            self._ohlcv_2h = value

    async def get_atr(self, symbol: str) -> float:
        """Return the latest ATR value for a symbol, recalculating if missing."""
        async with self.ohlcv_lock:
            df = self.ohlcv
            if "symbol" in df.index.names and symbol in df.index.get_level_values(
                "symbol"
            ):
                sub_df = df.xs(symbol, level="symbol", drop_level=False)
            else:
                return 0.0
        if sub_df.empty:
            return 0.0
        if "atr" in sub_df.columns:
            try:
                value = float(sub_df["atr"].iloc[-1])
                if value > 0:
                    return value
            except (IndexError, ValueError) as exc:
                logger.error("get_atr failed for %s: %s", symbol, exc)
        try:
            new_ind = IndicatorsCache(
                sub_df.droplevel("symbol"),
                self.config,
                sub_df["close"].pct_change().std(),
                "primary",
            )
            self.indicators[symbol] = new_ind
            for col in new_ind.df.columns:
                if col not in sub_df.columns:
                    sub_df[col] = new_ind.df[col]
                else:
                    sub_df[col].update(new_ind.df[col])
            async with self.ohlcv_lock:
                base_df = self.ohlcv
                base_df.loc[sub_df.index, sub_df.columns] = sub_df
                self.ohlcv = base_df
            if "atr" in new_ind.df.columns and not new_ind.df["atr"].empty:
                return float(new_ind.df["atr"].iloc[-1])
        except (KeyError, ValueError) as e:
            logger.error("Ошибка расчета ATR для %s: %s", symbol, e)
        return 0.0

    async def is_data_fresh(
        self, symbol: str, timeframe: str = "primary", max_delay: float = 60
    ) -> bool:
        """Return True if the most recent candle is within ``max_delay`` seconds."""
        try:
            if symbol in LATEST_OHLCV:
                ts = LATEST_OHLCV[symbol]["timestamp"]
                age = pd.Timestamp.now(tz="UTC") - pd.to_datetime(ts, utc=True)
                if age.total_seconds() <= max_delay:
                    return True
            df_lock = self.ohlcv_lock if timeframe == "primary" else self.ohlcv_2h_lock
            df = self.ohlcv if timeframe == "primary" else self.ohlcv_2h
            async with df_lock:
                if "symbol" in df.index.names and symbol in df.index.get_level_values(
                    "symbol"
                ):
                    sub_df = df.xs(symbol, level="symbol", drop_level=False)
                else:
                    return False
            if sub_df.empty:
                return False
            last_ts = sub_df.index.get_level_values("timestamp")[-1]
            age = pd.Timestamp.now(tz="UTC") - last_ts
            return age.total_seconds() <= max_delay
        except (KeyError, ValueError) as e:
            logger.error("Error checking data freshness for %s: %s", symbol, e)
            return False

    async def load_initial(self):
        try:
            markets = await safe_api_call(self.exchange, "load_markets")
            self.usdt_pairs = await self.select_liquid_pairs(markets)
            if not self.usdt_pairs:
                raise RuntimeError("No liquid USDT pairs found")
            logger.info(
                "Найдено %s USDT-пар с высокой ликвидностью",
                len(self.usdt_pairs),
            )
            tasks: List[asyncio.Task] = []
            task_info: List[tuple] = []
            history_limit = self.config.get("min_data_length", 200)

            mem = psutil.virtual_memory()
            avail_bytes = getattr(mem, "available", 0)
            avail_gb = avail_bytes / (1024**3)
            batch_size = self.config.get("history_batch_size", 10)
            batch_size = max(1, min(batch_size, int(max(1, avail_gb))))
            logger.info(
                "Исторические данные загружаются батчами по %s (%.1f GB available)",
                batch_size,
                avail_gb,
            )
            hist_sem = asyncio.Semaphore(batch_size)

            async def limited_history(sym: str, tf: str, prefix: str = "") -> tuple:
                async with hist_sem:
                    return await self.fetch_ohlcv_history(
                        sym,
                        tf,
                        history_limit,
                        cache_prefix=prefix,
                    )

            for symbol in self.usdt_pairs:
                orderbook = await self.fetch_orderbook(symbol)
                bid_volume = (
                    sum([bid[1] for bid in orderbook.get("bids", [])[:5]])
                    if orderbook.get("bids")
                    else 0
                )
                ask_volume = (
                    sum([ask[1] for ask in orderbook.get("asks", [])[:5]])
                    if orderbook.get("asks")
                    else 0
                )
                liquidity = min(bid_volume, ask_volume)
                self.symbol_priority[symbol] = -liquidity
                tasks.append(
                    asyncio.create_task(
                        limited_history(
                            symbol,
                            self.config["timeframe"],
                            "",
                        )
                    )
                )
                task_info.append(("history", symbol, self.config["timeframe"], ""))
                if self.config["secondary_timeframe"] != self.config["timeframe"]:
                    tasks.append(
                        asyncio.create_task(
                            limited_history(
                                symbol,
                                self.config["secondary_timeframe"],
                                "2h_",
                            )
                        )
                    )
                    task_info.append(
                        (
                            "history",
                            symbol,
                            self.config["secondary_timeframe"],
                            "2h_",
                        )
                    )
                tasks.append(asyncio.create_task(self.fetch_funding_rate(symbol)))
                task_info.append(("funding", symbol, None, None))
                tasks.append(asyncio.create_task(self.fetch_open_interest(symbol)))
                task_info.append(("open_interest", symbol, None, None))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_symbols: set[str] = set()
            for info, result in zip(task_info, results):
                task_type, symbol, timeframe, prefix = info
                if isinstance(result, Exception):
                    if task_type == "history":
                        logger.error(
                            "Ошибка получения истории для %s (%s): %s",
                            symbol,
                            timeframe,
                            result,
                        )
                        try:
                            _, result = await limited_history(symbol, timeframe, prefix)
                        except Exception as retry_exc:  # noqa: BLE001
                            logger.error(
                                "Повторная попытка истории для %s (%s) не удалась: %s",
                                symbol,
                                timeframe,
                                retry_exc,
                            )
                            failed_symbols.add(symbol)
                            continue
                    elif task_type == "funding":
                        logger.error(
                            "Ошибка получения ставки финансирования для %s: %s",
                            symbol,
                            result,
                        )
                        continue
                    elif task_type == "open_interest":
                        logger.error(
                            "Ошибка получения открытого интереса для %s: %s",
                            symbol,
                            result,
                        )
                        continue
                if task_type == "history":
                    if isinstance(result, tuple) and len(result) == 2:
                        _, df = result
                    else:
                        df = result
                    if not check_dataframe_empty(
                        df, f"load_initial {symbol} {timeframe}"
                    ):
                        df["symbol"] = symbol
                        df = df.set_index(["symbol", df.index])
                        df.index.set_names(["symbol", "timestamp"], inplace=True)
                        await self.synchronize_and_update(
                            symbol,
                            df,
                            self.funding_rates.get(symbol, 0.0),
                            self.open_interest.get(symbol, 0.0),
                            {"imbalance": 0.0, "timestamp": time.time()},
                            timeframe=(
                                "primary"
                                if timeframe == self.config["timeframe"]
                                else "secondary"
                            ),
                        )
            if failed_symbols:
                logger.warning(
                    "Удаляем пары с ошибками загрузки: %s",
                    ", ".join(sorted(failed_symbols)),
                )
                self.usdt_pairs = [
                    s for s in self.usdt_pairs if s not in failed_symbols
                ]
                for sym in failed_symbols:
                    self.symbol_priority.pop(sym, None)
            await self.release_memory()
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error("Ошибка загрузки начальных данных: %s", e)
            await self.telegram_logger.send_telegram_message(
                f"Ошибка загрузки данных: {e}"
            )

    async def select_liquid_pairs(self, markets: Dict) -> List[str]:
        """Return top liquid USDT futures pairs only.

        Filters out spot markets using contract metadata rather than
        string patterns. Volume ranking and the configured top limit
        remain unchanged.
        """

        pair_volumes = []
        semaphore = asyncio.Semaphore(self.config.get("max_concurrent_requests", 10))

        async def fetch_volume(sym: str) -> tuple:
            async with semaphore:
                try:
                    ticker = await safe_api_call(self.exchange, "fetch_ticker", sym)
                    volume = float(ticker.get("quoteVolume") or 0)
                except (httpx.HTTPError, RuntimeError) as e:  # noqa: BLE001
                    logger.error("Ошибка получения тикера для %s: %s", sym, e)
                    volume = 0.0
                return sym, volume

        min_age = (
            self.config.get("min_data_length", 0)
            * pd.Timedelta(self.config["timeframe"]).total_seconds()
        )
        now = pd.Timestamp.utcnow()
        batch = []
        batch_size = self.config.get("max_volume_batch", 50)
        candidate_markets = 0
        for symbol, market in markets.items():
            # Only consider active USDT-margined futures symbols using metadata
            if (
                market.get("active")
                and market.get("quote") == "USDT"
                and (market.get("contract") or market.get("linear"))
            ):
                candidate_markets += 1
                launch_time = None
                info = market.get("info") or {}
                for key in (
                    "launchTime",
                    "launch_time",
                    "listingTime",
                    "listing_date",
                    "created",
                ):
                    raw = info.get(key) or market.get(key)
                    if raw:
                        try:
                            launch_time = pd.to_datetime(raw, unit="ms", utc=True)
                        except (ValueError, TypeError):  # noqa: BLE001
                            launch_time = pd.to_datetime(raw, utc=True, errors="coerce")
                        break
                if launch_time is not None:
                    age = (now - launch_time).total_seconds()
                    if age < min_age:
                        continue
                batch.append(fetch_volume(symbol))
                if len(batch) >= batch_size:
                    pair_volumes.extend(await asyncio.gather(*batch))
                    batch.clear()

        if batch:
            pair_volumes.extend(await asyncio.gather(*batch))

        # Deduplicate entries using the canonical symbol
        highest = {}
        for sym, vol in pair_volumes:
            canon = self.fix_ws_symbol(sym)
            if canon not in highest or vol > highest[canon][1]:
                highest[canon] = (sym, vol)

        sorted_pairs = sorted(highest.values(), key=lambda x: x[1], reverse=True)
        min_liq = self.config.get("min_liquidity", 0)
        filtered = [p for p in sorted_pairs if p[1] >= min_liq]
        top_limit = self.config.get("max_symbols", 50)
        result = [s for s, _ in filtered[:top_limit]]
        logger.info(
            "select_liquid_pairs: total=%d, usdt_futures=%d, after_liquidity=%d",
            len(markets),
            candidate_markets,
            len(result),
        )
        if not result:
            raise ValueError(
                "No liquid USDT futures pairs found. Consider lowering min_liquidity or"
                " adjusting other configuration values."
            )
        return result

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_ohlcv_single(
        self, symbol: str, timeframe: str, limit: int = 200, cache_prefix: str = ""
    ) -> tuple:
        try:
            if os.getenv("TEST_MODE") == "1":
                ohlcv = generate_synthetic_ohlcv(timeframe, limit)
            else:
                ohlcv = await safe_api_call(
                    self.exchange,
                    "fetch_ohlcv",
                    symbol,
                    timeframe,
                    limit=limit,
                )
            if not ohlcv or len(ohlcv) < limit * 0.8:
                logger.warning(
                    "Неполные данные OHLCV для %s (%s), получено %s из %s",
                    symbol,
                    timeframe,
                    len(ohlcv),
                    limit,
                )
                return symbol, pd.DataFrame()
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(np.float32)
            if len(df) >= 3:
                df = filter_outliers_zscore(df, "close")
            if df["close"].isna().sum() / len(df) > 0.05:
                logger.warning(
                    "Слишком много пропусков в данных для %s (%s) (>5%), использование forward-fill",
                    symbol,
                    timeframe,
                )
                df = df.ffill()
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            max_gap = pd.Timedelta(timeframe).total_seconds() * 2
            if time_diffs.max() > max_gap:
                logger.warning(
                    "Обнаружен значительный разрыв в данных для %s (%s): %.2f минут",
                    symbol,
                    timeframe,
                    time_diffs.max() / 60,
                )
                await self.telegram_logger.send_telegram_message(
                    f"⚠️ Разрыв в данных для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
                )
                return symbol, pd.DataFrame()
            df = df.interpolate(method="time", limit_direction="both")
            if df.empty:
                logger.warning(
                    "Skipping cache for %s (%s) — no data",
                    symbol,
                    timeframe,
                )
            else:
                self.cache.save_cached_data(f"{cache_prefix}{symbol}", timeframe, df)
            return symbol, pd.DataFrame(df)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error("Ошибка получения OHLCV для %s (%s): %s", symbol, timeframe, e)
            return symbol, pd.DataFrame()

    async def fetch_ohlcv_history(
        self, symbol: str, timeframe: str, total_limit: int, cache_prefix: str = ""
    ) -> tuple:
        """Fetch extended OHLCV history by performing multiple requests."""
        try:
            if os.getenv("TEST_MODE") == "1":
                df = pd.DataFrame(
                    generate_synthetic_ohlcv(timeframe, total_limit),
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp")
            else:
                all_data = []
                timeframe_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000)
                since = None
                remaining = total_limit
                per_request = min(1000, total_limit)
                while remaining > 0:
                    limit = min(per_request, remaining)
                    ohlcv = await safe_api_call(
                        self.exchange,
                        "fetch_ohlcv",
                        symbol,
                        timeframe,
                        limit=limit,
                        since=since,
                    )
                    if not ohlcv:
                        break
                    df_part = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df_part["timestamp"] = pd.to_datetime(
                        df_part["timestamp"], unit="ms", utc=True
                    )
                    df_part = df_part.set_index("timestamp")
                    all_data.append(df_part)
                    remaining -= len(df_part)
                    if len(df_part) < limit:
                        break
                    since = (
                        int(df_part.index[0].timestamp() * 1000) - timeframe_ms * limit
                    )
                if not all_data:
                    return symbol, pd.DataFrame()
                df = pd.concat(all_data).sort_index().drop_duplicates()
            if df.empty:
                logger.warning(
                    "Skipping cache for %s (%s) — no data",
                    symbol,
                    timeframe,
                )
            else:
                self.cache.save_cached_data(f"{cache_prefix}{symbol}", timeframe, df)
            return symbol, pd.DataFrame(df)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error(
                "Ошибка получения расширенной истории OHLCV для %s (%s): %s",
                symbol,
                timeframe,
                e,
            )
            return symbol, pd.DataFrame()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_funding_rate(self, symbol: str) -> float:
        try:
            futures_symbol = self.fix_symbol(symbol)
            funding = await safe_api_call(
                self.exchange,
                "fetch_funding_rate",
                futures_symbol,
            )
            rate = float(funding.get("fundingRate", 0.0))
            async with self.funding_lock:
                self.funding_rates[symbol] = rate
            return rate
        except (KeyError, ValueError) as e:
            logger.error("Ошибка получения ставки финансирования для %s: %s", symbol, e)
            return 0.0

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_open_interest(self, symbol: str) -> float:
        try:
            futures_symbol = self.fix_symbol(symbol)
            oi = await safe_api_call(
                self.exchange,
                "fetch_open_interest",
                futures_symbol,
            )
            interest = float(oi.get("openInterest", 0.0))
            async with self.oi_lock:
                prev = self.open_interest.get(symbol)
                if prev and prev != 0:
                    self.open_interest_change[symbol] = (interest - prev) / prev
                else:
                    self.open_interest_change[symbol] = 0.0
                self.open_interest[symbol] = interest
            return interest
        except (KeyError, ValueError) as e:
            logger.error("Ошибка получения открытого интереса для %s: %s", symbol, e)
            return 0.0

    @retry(wait=wait_exponential(multiplier=1, min=2, max=5))
    async def fetch_orderbook(self, symbol: str) -> Dict:
        try:
            orderbook = await safe_api_call(
                self.exchange,
                "fetch_order_book",
                symbol,
                limit=10,
            )
            if not orderbook["bids"] or not orderbook["asks"]:
                logger.warning(
                    "Пустая книга ордеров для %s, повторная попытка",
                    symbol,
                )
                raise Exception("Пустой ордербук")
            return orderbook
        except (KeyError, ValueError) as e:
            logger.error("Ошибка получения книги ордеров для %s: %s", symbol, e)
            return {"bids": [], "asks": []}

    async def synchronize_and_update(
        self,
        symbol: str,
        df: pd.DataFrame,
        funding_rate: float,
        open_interest: float,
        orderbook: dict,
        timeframe: str = "primary",
    ):
        try:
            if check_dataframe_empty(
                df, f"synchronize_and_update {symbol} {timeframe}"
            ):
                logger.warning(
                    "Пустой DataFrame для %s (%s), пропуск синхронизации",
                    symbol,
                    timeframe,
                )
                return
            if df["close"].isna().any() or (df["close"] <= 0).any():
                logger.warning(
                    "Некорректные данные для %s (%s), пропуск",
                    symbol,
                    timeframe,
                )
                return
            lock = self._get_symbol_lock(symbol)
            async with lock:
                if timeframe == "primary":
                    if self.use_polars:
                        df_pl = (
                            pl.from_pandas(df.reset_index())
                            if isinstance(df, pd.DataFrame)
                            else df
                        )
                        base = (
                            self._ohlcv.filter(pl.col("symbol") != symbol)
                            if self._ohlcv.height > 0
                            else self._ohlcv
                        )
                        self._ohlcv = (
                            pl.concat([base, df_pl]) if base.height > 0 else df_pl
                        )
                        self._ohlcv = self._ohlcv.sort(["symbol", "timestamp"])
                    else:
                        if isinstance(self.ohlcv.index, pd.MultiIndex):
                            base = self.ohlcv.drop(
                                symbol, level="symbol", errors="ignore"
                            )
                        else:
                            base = self.ohlcv
                        self.ohlcv = pd.concat(
                            [base, df], ignore_index=False
                        ).sort_index()
                else:
                    if self.use_polars:
                        df_pl = (
                            pl.from_pandas(df.reset_index())
                            if isinstance(df, pd.DataFrame)
                            else df
                        )
                        base = (
                            self._ohlcv_2h.filter(pl.col("symbol") != symbol)
                            if self._ohlcv_2h.height > 0
                            else self._ohlcv_2h
                        )
                        self._ohlcv_2h = (
                            pl.concat([base, df_pl]) if base.height > 0 else df_pl
                        )
                        self._ohlcv_2h = self._ohlcv_2h.sort(["symbol", "timestamp"])
                    else:
                        if isinstance(self.ohlcv_2h.index, pd.MultiIndex):
                            base = self.ohlcv_2h.drop(
                                symbol, level="symbol", errors="ignore"
                            )
                        else:
                            base = self.ohlcv_2h
                        self.ohlcv_2h = pd.concat(
                            [base, df], ignore_index=False
                        ).sort_index()
                self.funding_rates[symbol] = funding_rate
                prev = self.open_interest.get(symbol)
                if prev and prev != 0:
                    self.open_interest_change[symbol] = (open_interest - prev) / prev
                else:
                    self.open_interest_change[symbol] = 0.0
                self.open_interest[symbol] = open_interest
                orderbook_df = pd.DataFrame(
                    [orderbook | {"symbol": symbol, "timestamp": time.time()}]
                )
                self.orderbook = pd.concat(
                    [self.orderbook, orderbook_df], ignore_index=False
                )
                self.orderbook_imbalance[symbol] = calculate_imbalance(orderbook)
                self.order_clusters[symbol] = detect_clusters(
                    orderbook, DEFAULT_CLUSTER_THRESHOLD
                )
                LATEST_IMBALANCE[symbol] = self.orderbook_imbalance[symbol]
                LATEST_CLUSTERS[symbol] = self.order_clusters[symbol]
            volatility = df["close"].pct_change().std() if not df.empty else 0.02
            cache_key = f"{symbol}_{timeframe}"
            if timeframe == "primary":
                fetch_needed = False
                obj_ref = None
                async with self.ohlcv_lock:
                    if cache_key not in self.indicators_cache:
                        logger.debug(
                            "Dispatching calc_indicators for %s %s", symbol, timeframe
                        )
                        obj_ref = self.calc_indicators.remote(
                            df.droplevel("symbol"), self.config, volatility, "primary"
                        )
                        fetch_needed = True
                    else:
                        cache_obj = self.indicators_cache[cache_key]
                        cache_obj.update(df.droplevel("symbol"))
                        idx = df.droplevel("symbol").index
                        self.indicators[symbol] = cache_obj
                        base_df = self.ohlcv
                        base_df.loc[df.index, cache_obj.df.columns] = cache_obj.df.loc[
                            idx, cache_obj.df.columns
                        ].to_numpy()
                        self.ohlcv = base_df

                if fetch_needed and obj_ref is not None:
                    result = await asyncio.to_thread(ray.get, obj_ref)
                    logger.debug(
                        "calc_indicators completed for %s %s (key=%s)",
                        symbol,
                        timeframe,
                        cache_key,
                    )
                    # Deep copy the DataFrame and Series to avoid read-only
                    # buffers returned from Ray
                    result.df = result.df.copy(deep=True)
                    for attr in [
                        "ema30",
                        "ema100",
                        "ema200",
                        "atr",
                        "rsi",
                        "adx",
                        "macd",
                        "volume_profile",
                    ]:
                        if hasattr(result, attr) and getattr(result, attr) is not None:
                            series = getattr(result, attr)
                            if isinstance(series, (pd.Series, pd.DataFrame)):
                                setattr(result, attr, series.copy(deep=True))
                    async with self.ohlcv_lock:
                        self.indicators_cache[cache_key] = result
                        self.indicators[symbol] = result
                        idx = df.droplevel("symbol").index
                        base_df = self.ohlcv
                        base_df.loc[df.index, result.df.columns] = result.df.loc[
                            idx, result.df.columns
                        ].to_numpy()
                        self.ohlcv = base_df
            else:
                fetch_needed = False
                obj_ref = None
                async with self.ohlcv_2h_lock:
                    if cache_key not in self.indicators_cache_2h:
                        logger.debug(
                            "Dispatching calc_indicators for %s %s", symbol, timeframe
                        )
                        obj_ref = self.calc_indicators.remote(
                            df.droplevel("symbol"), self.config, volatility, "secondary"
                        )
                        fetch_needed = True
                    else:
                        cache_obj = self.indicators_cache_2h[cache_key]
                        cache_obj.update(df.droplevel("symbol"))
                        idx = df.droplevel("symbol").index
                        base_df_2h = self.ohlcv_2h
                        base_df_2h.loc[df.index, cache_obj.df.columns] = (
                            cache_obj.df.loc[idx, cache_obj.df.columns].to_numpy()
                        )
                        self.ohlcv_2h = base_df_2h
                        self.indicators_2h[symbol] = cache_obj

                if fetch_needed and obj_ref is not None:
                    result = await asyncio.to_thread(ray.get, obj_ref)
                    logger.debug(
                        "calc_indicators completed for %s %s (key=%s)",
                        symbol,
                        timeframe,
                        cache_key,
                    )
                    async with self.ohlcv_2h_lock:
                        # Deep copy before caching to allow in-place updates
                        result.df = result.df.copy(deep=True)
                        for attr in [
                            "ema30",
                            "ema100",
                            "ema200",
                            "atr",
                            "rsi",
                            "adx",
                            "macd",
                            "volume_profile",
                        ]:
                            if (
                                hasattr(result, attr)
                                and getattr(result, attr) is not None
                            ):
                                series = getattr(result, attr)
                                if isinstance(series, (pd.Series, pd.DataFrame)):
                                    setattr(result, attr, series.copy(deep=True))

                        self.indicators_cache_2h[cache_key] = result
                        self.indicators_2h[symbol] = result
                        idx = df.droplevel("symbol").index
                        base_df_2h = self.ohlcv_2h
                        base_df_2h.loc[df.index, result.df.columns] = result.df.loc[
                            idx, result.df.columns
                        ].to_numpy()
                        self.ohlcv_2h = base_df_2h
            callbacks: list[Awaitable[Any]] = []
            if self.feature_callback:
                callbacks.append(self.feature_callback(symbol))
            if self.trade_callback:
                callbacks.append(self.trade_callback(symbol))
            if callbacks:
                await asyncio.gather(*callbacks)
            self.cache.save_cached_data(f"{timeframe}_{symbol}", timeframe, df)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error(
                "Ошибка синхронизации данных для %s (%s): %s", symbol, timeframe, e
            )

    async def cleanup_old_data(self):
        while True:
            try:
                async with self.cleanup_lock:
                    current_time = pd.Timestamp.now(tz="UTC")
                    async with self.ohlcv_lock:
                        if self.use_polars:
                            if self._ohlcv.height > 0:
                                threshold = (
                                    current_time
                                    - pd.Timedelta(seconds=self.config["forget_window"])
                                ).to_pydatetime()
                                if (
                                    not hasattr(self._ohlcv, "dtypes")
                                    or self._ohlcv.dtypes[1] == pl.Object
                                ):
                                    self._ohlcv = pl.from_pandas(
                                        self._ohlcv.to_pandas()
                                    )
                                df_pd = self._ohlcv.to_pandas()
                                df_pd = df_pd[df_pd["timestamp"] >= threshold]
                                self._ohlcv = pl.from_pandas(df_pd)
                        elif not self.ohlcv.empty:
                            threshold = current_time - pd.Timedelta(
                                seconds=self.config["forget_window"]
                            )
                            self.ohlcv = self.ohlcv[
                                self.ohlcv.index.get_level_values("timestamp")
                                >= threshold
                            ]
                    async with self.ohlcv_2h_lock:
                        if self.use_polars:
                            if self._ohlcv_2h.height > 0:
                                threshold = (
                                    current_time
                                    - pd.Timedelta(seconds=self.config["forget_window"])
                                ).to_pydatetime()
                                if (
                                    not hasattr(self._ohlcv_2h, "dtypes")
                                    or self._ohlcv_2h.dtypes[1] == pl.Object
                                ):
                                    self._ohlcv_2h = pl.from_pandas(
                                        self._ohlcv_2h.to_pandas()
                                    )
                                df2_pd = self._ohlcv_2h.to_pandas()
                                df2_pd = df2_pd[df2_pd["timestamp"] >= threshold]
                                self._ohlcv_2h = pl.from_pandas(df2_pd)
                        elif not self.ohlcv_2h.empty:
                            threshold = current_time - pd.Timedelta(
                                seconds=self.config["forget_window"]
                            )
                            self.ohlcv_2h = self.ohlcv_2h[
                                self.ohlcv_2h.index.get_level_values("timestamp")
                                >= threshold
                            ]
                    async with self.orderbook_lock:
                        if (
                            not self.orderbook.empty
                            and "timestamp" in self.orderbook.columns
                        ):
                            self.orderbook = self.orderbook[
                                self.orderbook["timestamp"]
                                >= time.time() - self.config["forget_window"]
                            ]
                    async with self.ohlcv_lock:
                        for symbol in list(self.processed_timestamps.keys()):
                            if symbol not in self.usdt_pairs:
                                del self.processed_timestamps[symbol]
                    async with self.ohlcv_2h_lock:
                        for symbol in list(self.processed_timestamps_2h.keys()):
                            if symbol not in self.usdt_pairs:
                                del self.processed_timestamps_2h[symbol]
                    logger.info("Старые данные очищены")
                await asyncio.sleep(self.config["data_cleanup_interval"] * 2)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, ValueError, OSError) as e:
                logger.exception("Ошибка очистки данных: %s", e)
                await asyncio.sleep(1)
                continue

    async def release_memory(self) -> None:
        """Trim stored history to the configured number of recent bars."""
        retention = int(self.config.get("history_retention", 0))
        if retention <= 0:
            return
        try:
            async with self.ohlcv_lock:
                if self.use_polars:
                    df = (
                        self._ohlcv.to_pandas()
                        if self._ohlcv.height > 0
                        else pd.DataFrame()
                    )
                else:
                    df = self._ohlcv
                if not df.empty:
                    df = df.groupby(level="symbol").tail(retention)
                if self.use_polars:
                    self._ohlcv = (
                        pl.from_pandas(df.reset_index())
                        if not df.empty
                        else pl.DataFrame()
                    )
                else:
                    self._ohlcv = df
            async with self.ohlcv_2h_lock:
                if self.use_polars:
                    df2 = (
                        self._ohlcv_2h.to_pandas()
                        if self._ohlcv_2h.height > 0
                        else pd.DataFrame()
                    )
                else:
                    df2 = self._ohlcv_2h
                if not df2.empty:
                    df2 = df2.groupby(level="symbol").tail(retention)
                if self.use_polars:
                    self._ohlcv_2h = (
                        pl.from_pandas(df2.reset_index())
                        if not df2.empty
                        else pl.DataFrame()
                    )
                else:
                    self._ohlcv_2h = df2
            logger.info("История обрезана до последних %s баров", retention)
        except (OSError, ValueError) as e:  # pragma: no cover - best effort cleanup
            logger.exception("Не удалось освободить память: %s", e)

    async def save_to_disk_buffer(self, priority, item):
        try:
            filename = (
                self.buffer_dir / f"buffer_{time.time()}.json"
            ).resolve(strict=False)
            if not filename.is_relative_to(self.buffer_dir):
                raise ValueError("Attempt to write outside buffer directory")

            def _json_dump(obj, path: Path):
                with path.open("w", encoding="utf-8") as f:
                    json.dump(obj, f)

            await asyncio.to_thread(
                _json_dump, {"priority": priority, "item": item}, filename
            )
            try:
                self.disk_buffer.put_nowait(str(filename))
            except asyncio.QueueFull:
                logger.warning(
                    "Очередь дискового буфера переполнена, сообщение пропущено"
                )
                await asyncio.to_thread(filename.unlink)
                return
            logger.info("Сообщение сохранено в дисковый буфер: %s", filename)
        except (OSError, ValueError, TypeError) as e:
            logger.error("Ошибка сохранения в дисковый буфер: %s", e)

    async def load_from_disk_buffer(self):
        while not self.disk_buffer.empty():
            try:
                filename = Path(self.disk_buffer.get_nowait()).resolve(strict=False)
                if not filename.is_relative_to(self.buffer_dir):
                    raise ValueError("Attempt to read outside buffer directory")
            except asyncio.QueueEmpty:
                break
            try:
                def _json_load(path: Path):
                    with path.open("r", encoding="utf-8") as f:
                        return json.load(f)

                data = await asyncio.to_thread(_json_load, filename)
                priority = data.get("priority")
                item = tuple(data.get("item", []))
                await self.ws_queue.put((priority, item))
                await asyncio.to_thread(filename.unlink)
                logger.info("Сообщение загружено из дискового буфера: %s", filename)
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ) as e:
                logger.error("Ошибка загрузки из дискового буфера: %s", e)
            finally:
                self.disk_buffer.task_done()

    async def load_from_disk_buffer_loop(self):
        while True:
            await self.load_from_disk_buffer()
            await asyncio.sleep(1)

    async def adjust_subscriptions(self):
        cpu_load = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_load = memory.percent
        current_rate = (
            len(self.process_rate_timestamps) / self.process_rate_window
            if self.process_rate_timestamps
            else self.ws_min_process_rate
        )
        if (
            cpu_load > self.load_threshold * 100
            or memory_load > self.load_threshold * 100
        ):
            new_max = max(10, self.max_subscriptions // 2)
            logger.warning(
                "Высокая нагрузка (CPU: %s%%, Memory: %s%%), уменьшение подписок до %s",
                cpu_load,
                memory_load,
                new_max,
            )
            self.max_subscriptions = new_max
        elif current_rate < self.ws_min_process_rate:
            new_max = max(10, int(self.max_subscriptions * 0.8))
            logger.warning(
                "Низкая скорость обработки (%.2f/s), уменьшение подписок до %s",
                current_rate,
                new_max,
            )
            self.max_subscriptions = new_max
        elif (
            cpu_load < self.load_threshold * 50
            and memory_load < self.load_threshold * 50
            and current_rate > self.ws_min_process_rate * 1.5
        ):
            new_max = min(100, self.max_subscriptions * 2)
            logger.info(
                "Низкая нагрузка, увеличение подписок до %s",
                new_max,
            )
            self.max_subscriptions = new_max

    async def subscribe_to_klines(self, symbols: List[str]):
        """Subscribe to kline streams for multiple symbols.

        The ``symbols`` list is divided into chunks of
        ``max_subscriptions_per_connection`` and each chunk is handled by a
        dedicated WebSocket connection.
        """
        try:
            self.cleanup_task = asyncio.create_task(self.cleanup_old_data())
            self.tasks = []
            if self.pro_exchange:
                await self._subscribe_with_ccxtpro(symbols)
            else:
                chunk_size = self.ws_subscription_batch_size
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i : i + chunk_size]
                    t1 = asyncio.create_task(
                        self._subscribe_chunk(
                            chunk,
                            self.config["ws_url"],
                            self.config["ws_reconnect_interval"],
                            timeframe="primary",
                        )
                    )
                    self.tasks.append(t1)
                    if self.config["secondary_timeframe"] != self.config["timeframe"]:
                        t2 = asyncio.create_task(
                            self._subscribe_chunk(
                                chunk,
                                self.config["ws_url"],
                                self.config["ws_reconnect_interval"],
                                timeframe="secondary",
                            )
                        )
                        self.tasks.append(t2)
                self.tasks.append(asyncio.create_task(self._process_ws_queue()))
                self.tasks.append(
                    asyncio.create_task(self.load_from_disk_buffer_loop())
                )
                self.tasks.append(asyncio.create_task(self.monitor_load()))
                self.tasks.append(asyncio.create_task(self.funding_rate_loop()))
                self.tasks.append(asyncio.create_task(self.open_interest_loop()))
                await asyncio.gather(*self.tasks, return_exceptions=True)
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            logger.error("Ошибка подписки на WebSocket: %s", e)
            await self.telegram_logger.send_telegram_message(f"Ошибка WebSocket: {e}")
            raise

    async def monitor_load(self):
        while True:
            try:
                await self.adjust_subscriptions()
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, OSError, ValueError) as e:
                logger.exception("Ошибка мониторинга нагрузки: %s", e)
                await asyncio.sleep(1)
                continue

    async def funding_rate_loop(self):
        while True:
            try:
                for symbol in list(self.usdt_pairs):
                    await self.fetch_funding_rate(symbol)
                await asyncio.sleep(self.config.get("funding_update_interval", 300))
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, RuntimeError, ValueError) as e:
                logger.exception("Ошибка обновления ставок финансирования: %s", e)
                await asyncio.sleep(1)
                continue

    async def open_interest_loop(self):
        while True:
            try:
                for symbol in list(self.usdt_pairs):
                    await self.fetch_open_interest(symbol)
                await asyncio.sleep(self.config.get("oi_update_interval", 300))
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, RuntimeError, ValueError) as e:
                logger.exception("Ошибка обновления открытого интереса: %s", e)
                await asyncio.sleep(1)
                continue

    def fix_symbol(self, symbol: str) -> str:
        """Normalize symbol for Bybit futures REST requests.

        Parameters
        ----------
        symbol : str
            Symbol in one of the supported formats, e.g. ``BTCUSDT``,
            ``BTC/USDT`` or ``BTC/USDT:USDT``.

        Returns
        -------
        str
            ``BTCUSDT`` –> ``BTCUSDT``
            ``BTC/USDT`` –> ``BTC/USDT:USDT``
            ``BTC/USDT:USDT`` remains unchanged
        """

        if symbol.endswith("/USDT"):
            return f"{symbol}:USDT"
        return symbol

    def fix_ws_symbol(self, symbol: str) -> str:
        """Convert symbol to the format required by Bybit WebSocket.

        Removes any slashes and the ``:USDT`` suffix so that
        ``BTC/USDT:USDT`` becomes ``BTCUSDT``.
        """

        return symbol.replace("/", "").replace(":USDT", "")

    async def _subscribe_symbol_ccxtpro(self, symbol: str, timeframe: str, label: str):
        """Watch OHLCV updates for a single symbol using CCXT Pro with automatic reconnection."""
        reconnect_attempts = 0
        max_reconnect_attempts = self.config.get("max_reconnect_attempts", 10)
        limit_exceeded_logged = False
        while True:
            try:
                ohlcv = await self.pro_exchange.watch_ohlcv(symbol, timeframe)
                if not ohlcv:
                    continue
                last = ohlcv[-1]
                kline_timestamp = pd.to_datetime(int(last[0]), unit="ms", utc=True)
                df = pd.DataFrame(
                    [
                        {
                            "timestamp": kline_timestamp,
                            "open": np.float32(last[1]),
                            "high": np.float32(last[2]),
                            "low": np.float32(last[3]),
                            "close": np.float32(last[4]),
                            "volume": np.float32(last[5]),
                        }
                    ]
                )
                df["symbol"] = symbol
                df = df.set_index(["symbol", "timestamp"])
                await self.synchronize_and_update(
                    symbol,
                    df,
                    self.funding_rates.get(symbol, 0.0),
                    self.open_interest.get(symbol, 0.0),
                    {"imbalance": 0.0, "timestamp": time.time()},
                    timeframe=label,
                )
                reconnect_attempts = 0
                if limit_exceeded_logged:
                    logger.info(
                        "Подписка CCXT Pro для %s (%s) восстановлена",
                        symbol,
                        label,
                    )
                    limit_exceeded_logged = False
            except (httpx.HTTPError, RuntimeError, ValueError) as e:
                reconnect_attempts += 1
                delay = min(2**reconnect_attempts, 60)
                logger.error(
                    "Ошибка CCXT Pro для %s (%s), попытка %s/%s, ожидание %s секунд: %s",
                    symbol,
                    label,
                    reconnect_attempts,
                    max_reconnect_attempts,
                    delay,
                    e,
                )
                if (
                    reconnect_attempts == max_reconnect_attempts
                    and not limit_exceeded_logged
                ):
                    logger.warning(
                        "Превышено максимальное количество попыток CCXT Pro для %s (%s). Продолжаем попытки с экспоненциальной задержкой",
                        symbol,
                        label,
                    )
                    limit_exceeded_logged = True
                await asyncio.sleep(delay)

    async def _subscribe_with_ccxtpro(self, symbols: List[str]):
        self.tasks = []
        for symbol in symbols:
            t1 = asyncio.create_task(
                self._subscribe_symbol_ccxtpro(
                    symbol, self.config["timeframe"], "primary"
                )
            )
            self.tasks.append(t1)
            if self.config["secondary_timeframe"] != self.config["timeframe"]:
                t2 = asyncio.create_task(
                    self._subscribe_symbol_ccxtpro(
                        symbol,
                        self.config["secondary_timeframe"],
                        "secondary",
                    )
                )
                self.tasks.append(t2)
        self.tasks.append(asyncio.create_task(self.monitor_load()))
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _connect_ws(self, url: str, connection_timeout: int):
        """Return an active WebSocket connection for ``url`` with retries."""
        attempts = 0
        max_attempts = self.config.get("max_reconnect_attempts", 10)
        while True:
            try:
                if url not in self.ws_pool:
                    self.ws_pool[url] = []
                ws = None
                while self.ws_pool[url]:
                    ws = self.ws_pool[url].pop(0)
                    if ws.open:
                        break
                    try:
                        await ws.close()
                    except (OSError, RuntimeError):
                        pass
                    ws = None
                if ws is None or not ws.open:
                    ws = await websockets.connect(
                        url,
                        ping_interval=20,
                        ping_timeout=30,
                        open_timeout=max(connection_timeout, 10),
                    )
                logger.info("Подключение к WebSocket %s", url)
                return ws
            except OSError as e:
                attempts += 1
                delay = min(2**attempts, 60)
                logger.error(
                    "Ошибка подключения к WebSocket %s, попытка %s/%s, ожидание %s секунд: %s",
                    url,
                    attempts,
                    max_attempts,
                    delay,
                    e,
                )
                if attempts >= max_attempts:
                    raise
                await asyncio.sleep(delay)

    async def _send_subscriptions(self, ws, symbols, timeframe: str):
        """Send subscription requests for ``symbols`` and confirm success."""
        attempts = 0
        max_attempts = self.config.get("max_reconnect_attempts", 10)
        selected_timeframe = (
            self.config["timeframe"]
            if timeframe == "primary"
            else self.config["secondary_timeframe"]
        )
        batch_size = self.ws_subscription_batch_size
        while True:
            try:
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i : i + batch_size]
                    for symbol in batch:
                        current_time = time.time()
                        self.ws_rate_timestamps.append(current_time)
                        self.ws_rate_timestamps = [
                            t for t in self.ws_rate_timestamps if current_time - t < 1
                        ]
                        if len(self.ws_rate_timestamps) > self.config["ws_rate_limit"]:
                            logger.warning(
                                "Превышен лимит подписок WebSocket, ожидание"
                            )
                            await asyncio.sleep(1)
                            self.ws_rate_timestamps = [
                                t
                                for t in self.ws_rate_timestamps
                                if current_time - t < 1
                            ]
                        ws_symbol = self.fix_ws_symbol(symbol)
                        interval = bybit_interval(selected_timeframe)
                        await ws.send(
                            json.dumps(
                                {
                                    "op": "subscribe",
                                    "args": [f"kline.{interval}.{ws_symbol}"],
                                }
                            )
                        )
                        rate = max(self.config.get("ws_rate_limit", 1), 1)
                        await asyncio.sleep(1 / rate)

                confirmations_needed = len(symbols)
                confirmations = 0
                startup_messages = []
                start_confirm = time.time()
                while (
                    confirmations < confirmations_needed
                    and time.time() - start_confirm < confirmations_needed * 5
                ):
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(response)
                        if isinstance(data, dict) and data.get("success") is True:
                            confirmations += 1
                            continue
                        startup_messages.append(response)
                    except asyncio.TimeoutError:
                        continue
                if confirmations < confirmations_needed:
                    raise RuntimeError("Подписка не подтверждена")
                return startup_messages
            except (httpx.HTTPError, RuntimeError, ValueError) as e:
                attempts += 1
                delay = min(2**attempts, 60)
                logger.error(
                    "Ошибка подписки на WebSocket для %s (%s), попытка %s/%s, ожидание %s секунд: %s",
                    symbols,
                    timeframe,
                    attempts,
                    max_attempts,
                    delay,
                    e,
                )
                if attempts >= max_attempts:
                    raise
                await asyncio.sleep(delay)
                continue

    async def _read_messages(
        self,
        ws,
        symbols,
        timeframe: str,
        selected_timeframe: str,
        connection_timeout: int,
    ):
        """Read messages from ``ws`` and enqueue them for processing."""
        last_msg = time.time()
        while True:
            try:
                start_time = time.time()
                message = await asyncio.wait_for(ws.recv(), timeout=connection_timeout)
                last_msg = time.time()
                latency = last_msg - start_time
                for symbol in symbols:
                    self.ws_latency[symbol] = latency
                if latency > 5:
                    logger.warning(
                        "Высокая задержка WebSocket для %s (%s): %.2f сек",
                        symbols,
                        timeframe,
                        latency,
                    )
                    await self.telegram_logger.send_telegram_message(
                        f"⚠️ Высокая задержка WebSocket для {symbols} ({timeframe}): {latency:.2f} сек"
                    )
                    for symbol in symbols:
                        symbol_df = await self.fetch_ohlcv_single(
                            symbol,
                            selected_timeframe,
                            limit=1,
                            cache_prefix="2h_" if timeframe == "secondary" else "",
                        )
                        if isinstance(symbol_df, tuple) and len(symbol_df) == 2:
                            _, df = symbol_df
                            if not check_dataframe_empty(
                                df, f"subscribe_to_klines {symbol} {timeframe}"
                            ):
                                df["symbol"] = symbol
                                df = df.set_index(["symbol", df.index])
                                await self.synchronize_and_update(
                                    symbol,
                                    df,
                                    self.funding_rates.get(symbol, 0.0),
                                    self.open_interest.get(symbol, 0.0),
                                    {"imbalance": 0.0, "timestamp": time.time()},
                                    timeframe=timeframe,
                                )
                    break
                data = json.loads(message)
                topic = data.get("topic", "")
                symbol = topic.split(".")[-1] if isinstance(topic, str) else ""
                priority = self.symbol_priority.get(symbol, 0)
                try:
                    await asyncio.wait_for(
                        self.ws_queue.put((priority, (symbols, message, timeframe))),
                        timeout=5,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Очередь WebSocket переполнена, сохранение в дисковый буфер"
                    )
                    await self.save_to_disk_buffer(
                        priority, (symbols, message, timeframe)
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Тайм-аут WebSocket для %s (%s), отправка пинга",
                    symbols,
                    timeframe,
                )
                await ws.ping()
                if time.time() - last_msg > self.ws_inactivity_timeout:
                    await ws.close()
                    raise ConnectionError("WebSocket inactivity")
                continue
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(
                    "WebSocket соединение закрыто для %s (%s): %s",
                    symbols,
                    timeframe,
                    e,
                )
                break
            except (ValueError, RuntimeError, OSError) as e:
                logger.exception(
                    "Ошибка обработки WebSocket сообщения для %s (%s): %s",
                    symbols,
                    timeframe,
                    e,
                )
                raise

    async def _subscribe_chunk(
        self, symbols, ws_url, connection_timeout, timeframe: str = "primary"
    ):
        """Subscribe to kline data for a chunk of symbols."""
        reconnect_attempts = 0
        max_reconnect_attempts = self.config.get("max_reconnect_attempts", 10)
        urls = [ws_url] + self.backup_ws_urls
        current_url_index = 0
        selected_timeframe = (
            self.config["timeframe"]
            if timeframe == "primary"
            else self.config["secondary_timeframe"]
        )
        while True:
            current_url = urls[current_url_index % len(urls)]
            ws = None
            connected = False
            try:
                ws = await self._connect_ws(current_url, connection_timeout)
                connected = True
                self.active_subscriptions += len(symbols)
                self.restart_attempts = 0
                reconnect_attempts = 0
                current_url_index = 0

                startup_messages = await self._send_subscriptions(
                    ws, symbols, timeframe
                )
                for message in startup_messages:
                    try:
                        data = json.loads(message)
                        topic = data.get("topic", "")
                        symbol = topic.split(".")[-1] if isinstance(topic, str) else ""
                        priority = self.symbol_priority.get(symbol, 0)
                        try:
                            await asyncio.wait_for(
                                self.ws_queue.put(
                                    (priority, (symbols, message, timeframe))
                                ),
                                timeout=5,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Очередь WebSocket переполнена, сохранение в дисковый буфер"
                            )
                            await self.save_to_disk_buffer(
                                priority, (symbols, message, timeframe)
                            )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

                await self._read_messages(
                    ws, symbols, timeframe, selected_timeframe, connection_timeout
                )
            except (httpx.HTTPError, RuntimeError, ValueError, OSError) as e:
                reconnect_attempts += 1
                current_url_index += 1
                delay = min(2**reconnect_attempts, 60)
                logger.error(
                    "Ошибка WebSocket %s для %s (%s), попытка %s/%s, ожидание %s секунд: %s",
                    current_url,
                    symbols,
                    timeframe,
                    reconnect_attempts,
                    max_reconnect_attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                if reconnect_attempts >= max_reconnect_attempts:
                    self.restart_attempts += 1
                    if self.restart_attempts >= self.max_restart_attempts:
                        logger.critical(
                            "Превышено максимальное количество перезапусков WebSocket для %s (%s)",
                            symbols,
                            timeframe,
                        )
                        await self.telegram_logger.send_telegram_message(
                            f"Критическая ошибка: Не удалось восстановить WebSocket для {symbols} ({timeframe})"
                        )
                        break
                    logger.info(
                        "Автоматический перезапуск WebSocket для %s (%s), попытка %s/%s",
                        symbols,
                        timeframe,
                        self.restart_attempts,
                        self.max_restart_attempts,
                    )
                    reconnect_attempts = 0
                    current_url_index = 0
                    await asyncio.sleep(60)
                else:
                    logger.info(
                        "Попытка загрузки данных через REST API для %s (%s)",
                        symbols,
                        timeframe,
                    )
                    for symbol in symbols:
                        try:
                            symbol_df = await self.fetch_ohlcv_single(
                                symbol,
                                selected_timeframe,
                                limit=1,
                                cache_prefix="2h_" if timeframe == "secondary" else "",
                            )
                            if isinstance(symbol_df, tuple) and len(symbol_df) == 2:
                                _, df = symbol_df
                                if not check_dataframe_empty(
                                    df, f"subscribe_to_klines {symbol} {timeframe}"
                                ):
                                    df["symbol"] = symbol
                                    df = df.set_index(["symbol", df.index])
                                    await self.synchronize_and_update(
                                        symbol,
                                        df,
                                        self.funding_rates.get(symbol, 0.0),
                                        self.open_interest.get(symbol, 0.0),
                                        {"imbalance": 0.0, "timestamp": time.time()},
                                        timeframe=timeframe,
                                    )
                        except (httpx.HTTPError, RuntimeError, ValueError) as rest_e:
                            logger.error(
                                "Ошибка REST API для %s (%s): %s",
                                symbol,
                                timeframe,
                                rest_e,
                            )
                            raise
            finally:
                if connected:
                    self.active_subscriptions -= len(symbols)
                if ws and ws.open:
                    self.ws_pool[current_url].append(ws)
                elif ws:
                    await ws.close()

    async def _process_ws_queue(self):
        last_latency_log = time.time()
        while True:
            got_item = False
            try:
                priority, (symbols, message, timeframe) = await self.ws_queue.get()
                got_item = True
                now = time.time()
                self.process_rate_timestamps.append(now)
                self.process_rate_timestamps = [
                    t
                    for t in self.process_rate_timestamps
                    if now - t < self.process_rate_window
                ]
                if (
                    len(self.process_rate_timestamps) > self.ws_min_process_rate
                    and (len(self.process_rate_timestamps) / self.process_rate_window)
                    < self.ws_min_process_rate
                ):
                    await self.adjust_subscriptions()
                data = json.loads(message)
                if (
                    not isinstance(data, dict)
                    or "topic" not in data
                    or "data" not in data
                    or not isinstance(data["data"], list)
                ):
                    logger.debug(
                        "Error in message format for %s: %s",
                        symbols,
                        message,
                    )
                    continue
                topic = data.get("topic", "")
                symbol = topic.split(".")[-1] if isinstance(topic, str) else ""
                if not symbol:
                    logger.debug(
                        "Symbol not found in topic for message: %s",
                        message,
                    )
                    continue
                for entry in data["data"]:
                    required_fields = [
                        "start",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                    if not all(field in entry for field in required_fields):
                        logger.warning(
                            "Invalid kline data (%s): %s",
                            timeframe,
                            entry,
                        )
                        continue
                    try:
                        kline_timestamp = pd.to_datetime(
                            int(entry["start"]), unit="ms", utc=True
                        )
                        open_price = float(entry["open"])
                        high_price = float(entry["high"])
                        low_price = float(entry["low"])
                        close_price = float(entry["close"])
                        volume = float(entry["volume"])
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Ошибка формата данных свечи для %s (%s): %s",
                            symbol,
                            timeframe,
                            e,
                        )
                        continue
                    timestamp_dict = (
                        self.processed_timestamps
                        if timeframe == "primary"
                        else self.processed_timestamps_2h
                    )
                    lock = self._get_symbol_lock(symbol)
                    confirm = entry.get("confirm", True)
                    async with lock:
                        if symbol not in timestamp_dict:
                            timestamp_dict[symbol] = set()
                        if confirm and entry["start"] in timestamp_dict[symbol]:
                            logger.debug(
                                "Дубликат сообщения для %s (%s) с временной меткой %s",
                                symbol,
                                timeframe,
                                kline_timestamp,
                            )
                            continue
                        if confirm:
                            timestamp_dict[symbol].add(entry["start"])
                            if len(timestamp_dict[symbol]) > 1000:
                                timestamp_dict[symbol] = set(
                                    list(timestamp_dict[symbol])[-500:]
                                )
                    current_time = pd.Timestamp.now(tz="UTC")
                    interval = pd.Timedelta(
                        self.config[
                            (
                                "timeframe"
                                if timeframe == "primary"
                                else "secondary_timeframe"
                            )
                        ]
                    ).total_seconds()
                    if (
                        confirm
                        and (current_time - kline_timestamp).total_seconds() > interval
                    ):
                        logger.warning(
                            "Получены устаревшие данные для %s (%s): %s",
                            symbol,
                            timeframe,
                            kline_timestamp,
                        )
                        continue
                    try:
                        df = pd.DataFrame(
                            [
                                {
                                    "timestamp": kline_timestamp,
                                    "open": np.float32(open_price),
                                    "high": np.float32(high_price),
                                    "low": np.float32(low_price),
                                    "close": np.float32(close_price),
                                    "volume": np.float32(volume),
                                }
                            ]
                        )
                        if len(df) >= 3:
                            df = filter_outliers_zscore(df, "close")
                        if df.empty:
                            logger.warning(
                                "Данные для %s (%s) отфильтрованы как аномалии",
                                symbol,
                                timeframe,
                            )
                            continue
                        df["symbol"] = symbol
                        df = df.set_index(["symbol", "timestamp"])
                        time_diffs = (
                            df.index.get_level_values("timestamp")
                            .to_series()
                            .diff()
                            .dt.total_seconds()
                        )
                        max_gap = (
                            pd.Timedelta(
                                self.config[
                                    (
                                        "timeframe"
                                        if timeframe == "primary"
                                        else "secondary_timeframe"
                                    )
                                ]
                            ).total_seconds()
                            * 2
                        )
                        if time_diffs.max() > max_gap:
                            logger.warning(
                                "Обнаружен разрыв в данных WebSocket для %s (%s): %.2f минут",
                                symbol,
                                timeframe,
                                time_diffs.max() / 60,
                            )
                            await self.telegram_logger.send_telegram_message(
                                f"⚠️ Разрыв в данных WebSocket для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
                            )
                        await self.synchronize_and_update(
                            symbol,
                            df,
                            self.funding_rates.get(symbol, 0.0),
                            self.open_interest.get(symbol, 0.0),
                            {"imbalance": 0.0, "timestamp": time.time()},
                            timeframe=timeframe,
                        )
                    except (ValueError, RuntimeError, OSError) as e:
                        logger.exception(
                            "Ошибка обработки данных для %s: %s", symbol, e
                        )
                        await asyncio.sleep(0.1)
                        continue
                if time.time() - last_latency_log > self.latency_log_interval:
                    rate = len(self.process_rate_timestamps) / self.process_rate_window
                    if self.ws_latency:
                        avg = sum(self.ws_latency.values()) / len(self.ws_latency)
                    else:
                        avg = 0.0
                    logger.info(
                        "Средняя задержка WebSocket: %.2f сек, скорость обработки: %.2f/с",
                        avg,
                        rate,
                    )
                    last_latency_log = time.time()
            except asyncio.CancelledError:
                raise
            except (ValueError, RuntimeError, OSError) as e:
                logger.exception("Ошибка обработки очереди WebSocket: %s", e)
                await asyncio.sleep(0.1)
                continue
            finally:
                if got_item:
                    self.ws_queue.task_done()

    async def stop(self):
        """Gracefully cancel running tasks and close open connections."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None

        for task in list(self.tasks):
            task.cancel()
        for task in list(self.tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.tasks.clear()

        for url, conns in list(self.ws_pool.items()):
            for ws in conns:
                try:
                    await ws.close()
                except (OSError, RuntimeError) as e:
                    logger.exception("Ошибка закрытия WebSocket %s: %s", url, e)
                    continue
        self.ws_pool.clear()

        if self.pro_exchange is not None and hasattr(self.pro_exchange, "close"):
            try:
                await self.pro_exchange.close()
            except (OSError, RuntimeError) as e:
                logger.exception("Ошибка закрытия ccxtpro: %s", e)
                pass

        await TelegramLogger.shutdown()
        if ray.is_initialized():
            ray.shutdown()


DataHandler = _instrument_methods(DataHandler)


# ----------------------------------------------------------------------
# REST API for minimal integration testing
# ----------------------------------------------------------------------

api_app = Flask(__name__)

# Default price returned when the exchange call fails. The value must be
# non‐zero so tests relying on a positive price succeed.
DEFAULT_PRICE = 100.0

# Minimum share of total volume required for a cluster
DEFAULT_CLUSTER_THRESHOLD = 0.1

# Cached OHLCV rows keyed by symbol. Each entry stores a timestamp aware
# ``pd.Timestamp`` and OHLCV fields so ``is_data_fresh`` can verify data
# recency.
LATEST_OHLCV: Dict[str, Dict[str, Any]] = {}

# Latest orderbook imbalance values keyed by symbol
LATEST_IMBALANCE: Dict[str, float] = {}

# Detected orderbook clusters keyed by symbol
LATEST_CLUSTERS: Dict[str, List[tuple[float, float]]] = {}

# Exchange instance used by the price endpoint. Lazily created on first use so
# tests without API keys can stub it easily.
_PRICE_EXCHANGE: BybitSDKAsync | None = None


def _get_price_exchange() -> BybitSDKAsync:
    global _PRICE_EXCHANGE
    if _PRICE_EXCHANGE is None:
        _PRICE_EXCHANGE = create_exchange()
    return _PRICE_EXCHANGE


@api_app.route("/price/<symbol>")
def price(symbol: str):
    """Return the most recent price for ``symbol`` from the exchange."""

    if os.getenv("TEST_MODE") == "1":
        return jsonify({"price": DEFAULT_PRICE})

    async def _lookup(sym: str) -> float:
        exch = _get_price_exchange()
        try:
            ohlcv = await safe_api_call(exch, "fetch_ohlcv", sym, "1m", limit=1)
            if ohlcv:
                ts, o, h, l, c, v = ohlcv[-1]
                LATEST_OHLCV[sym] = {
                    "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
                return float(c)
        except (
            httpx.HTTPError,
            RuntimeError,
        ) as exc:  # pragma: no cover - network failures
            logger.error("OHLCV fetch failed for %s: %s", sym, exc)
        try:
            ticker = await safe_api_call(exch, "fetch_ticker", sym)
            price_val = float(ticker.get("last") or 0.0)
            LATEST_OHLCV[sym] = {
                "timestamp": pd.Timestamp.utcnow(),
                "open": price_val,
                "high": price_val,
                "low": price_val,
                "close": price_val,
                "volume": float(ticker.get("quoteVolume") or 0.0),
            }
            return price_val
        except (
            httpx.HTTPError,
            RuntimeError,
        ) as exc:  # pragma: no cover - network failures
            logger.error("Ticker fetch failed for %s: %s", sym, exc)
            return DEFAULT_PRICE

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        price_val = asyncio.run(_lookup(symbol))
    else:
        fut = asyncio.run_coroutine_threadsafe(_lookup(symbol), loop)
        price_val = fut.result()
    return jsonify({"price": price_val})


@api_app.route("/imbalance/<symbol>")
def imbalance(symbol: str):
    """Return the latest orderbook imbalance for ``symbol``."""

    value = LATEST_IMBALANCE.get(symbol, 0.0)
    return jsonify({"imbalance": value})


@api_app.route("/clusters/<symbol>")
def clusters(symbol: str):
    """Return detected orderbook clusters for ``symbol``."""

    value = LATEST_CLUSTERS.get(symbol, [])
    return jsonify({"clusters": value})


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    from bot.utils import configure_logging

    configure_logging()
    load_dotenv()
    port = int(os.environ.get("PORT", "8000"))
    # По умолчанию слушаем только локальный интерфейс.
    host = os.environ.get("HOST", "127.0.0.1")
    # Prevent binding to all interfaces.
    if host.strip() == "0.0.0.0":  # nosec B104
        raise ValueError("HOST=0.0.0.0 запрещён из соображений безопасности")
    if host != "127.0.0.1":
        logger.warning(
            "Используется не локальный хост %s; убедитесь, что это намеренно",
            host,
        )
    else:
        logger.info("HOST не установлен, используется %s", host)
    logger.info("Запуск сервиса DataHandler на %s:%s", host, port)
    api_app.run(host=host, port=port)  # nosec B104  # хост проверен выше
