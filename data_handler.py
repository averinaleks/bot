from __future__ import annotations

import asyncio
import json
import time
import pandas as pd
import numpy as np
import websockets
from utils import (
    BybitSDKAsync,
    logger,
    check_dataframe_empty,
    HistoricalDataCache,
    filter_outliers_zscore,
    TelegramLogger,
    calculate_volume_profile as utils_volume_profile,
    safe_api_call,
)
from tenacity import retry, wait_exponential
from typing import List, Dict, TYPE_CHECKING
from config import BotConfig
import ta
import os
from queue import Queue
import pickle
import psutil
import ray
from flask import Flask, jsonify

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import ccxtpro

try:
    from numba import cuda  # type: ignore
    GPU_AVAILABLE = hasattr(cuda, "is_available") and cuda.is_available()
except Exception:  # pragma: no cover - numba without cuda support
    cuda = None  # type: ignore
    GPU_AVAILABLE = False


def create_exchange() -> BybitSDKAsync:
    """Create an authenticated Bybit SDK instance."""
    return BybitSDKAsync(
        api_key=os.environ.get("BYBIT_API_KEY", ""),
        api_secret=os.environ.get("BYBIT_API_SECRET", ""),
    )


def ema_fast(values: np.ndarray, window: int, wilder: bool = False) -> np.ndarray:
    """Compute EMA using GPU if available, otherwise CPU."""
    values = np.asarray(values, dtype=np.float64)
    alpha = (1 / window) if wilder else 2 / (window + 1)
    if cuda is not None and GPU_AVAILABLE:
        values_dev = cuda.to_device(values)
        result_dev = cuda.device_array_like(values)

        @cuda.jit
        def _ema_kernel(v, a, out):
            if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
                out[0] = v[0]
                for i in range(1, v.size):
                    out[i] = a * v[i] + (1 - a) * out[i - 1]

        _ema_kernel[1, 1](values_dev, alpha, result_dev)
        return result_dev.copy_to_host()

    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def atr_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Compute ATR using Wilder's smoothing with optional GPU acceleration."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = np.zeros_like(tr)
    if len(tr) >= window:
        atr[window - 1] = tr[:window].mean()
        for i in range(window, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / float(window)
    return atr


class IndicatorsCache:
    def __init__(self, df: pd.DataFrame, config: BotConfig, volatility: float, timeframe: str = "primary"):
        self.df = df
        self.config = config
        self.volatility = volatility
        self.last_volume_profile_update = 0
        self.volume_profile_update_interval = 5
        try:
            if timeframe == "primary":
                close_np = df["close"].to_numpy()
                high_np = df["high"].to_numpy()
                low_np = df["low"].to_numpy()
                self.ema30 = pd.Series(ema_fast(close_np, config["ema30_period"]), index=df.index)
                self.ema100 = pd.Series(ema_fast(close_np, config["ema100_period"]), index=df.index)
                self.ema200 = pd.Series(ema_fast(close_np, config["ema200_period"]), index=df.index)
                self.atr = pd.Series(
                    atr_fast(high_np, low_np, close_np, config["atr_period_default"]), index=df.index
                )
                self.rsi = ta.momentum.rsi(df["close"], window=14, fillna=True)
                self.adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
                self.macd = ta.trend.macd_diff(df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            elif timeframe == "secondary":
                close_np = df["close"].to_numpy()
                self.ema30 = pd.Series(ema_fast(close_np, config["ema30_period"]), index=df.index)
                self.ema100 = pd.Series(ema_fast(close_np, config["ema100_period"]), index=df.index)
            self.volume_profile = None
            if len(df) - self.last_volume_profile_update >= self.volume_profile_update_interval:
                self.volume_profile = self.calculate_volume_profile(df)
                self.last_volume_profile_update = len(df)
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов ({timeframe}): {e}")
            self.ema30 = self.ema100 = self.ema200 = self.atr = self.rsi = self.adx = self.macd = (
                self.volume_profile
            ) = None

    def calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        try:
            prices = df["close"].to_numpy(dtype=np.float32)
            volumes = df["volume"].to_numpy(dtype=np.float32)
            vp = utils_volume_profile(prices, volumes, bins=50)
            price_bins = np.linspace(prices.min(), prices.max(), num=len(vp))
            return pd.Series(vp, index=price_bins)
        except Exception as e:
            logger.error(f"Ошибка расчета Volume Profile: {e}")
            return None


@ray.remote(num_cpus=1)
def calc_indicators(df: pd.DataFrame, config: BotConfig, volatility: float, timeframe: str):
    return IndicatorsCache(df, config, volatility, timeframe)


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
    """

    def __init__(self, config: BotConfig, telegram_bot, chat_id,
                 exchange: BybitSDKAsync | None = None,
                 pro_exchange: "ccxtpro.bybit" | None = None):
        self.config = config
        self.exchange = exchange or create_exchange()
        self.pro_exchange = pro_exchange
        self.telegram_logger = TelegramLogger(
            telegram_bot,
            chat_id,
            max_queue_size=config.get("telegram_queue_size"),
        )
        self.cache = HistoricalDataCache(config["cache_dir"])
        self.ohlcv = pd.DataFrame()
        self.ohlcv_2h = pd.DataFrame()
        self.funding_rates = {}
        self.open_interest = {}
        self.orderbook = pd.DataFrame()
        self.indicators = {}
        self.indicators_cache = {}
        self.indicators_2h = {}
        self.indicators_cache_2h = {}
        self.usdt_pairs = []
        self.ohlcv_lock = asyncio.Lock()
        self.ohlcv_2h_lock = asyncio.Lock()
        self.funding_lock = asyncio.Lock()
        self.oi_lock = asyncio.Lock()
        self.orderbook_lock = asyncio.Lock()
        self.cleanup_lock = asyncio.Lock()
        self.ws_rate_timestamps = []
        self.process_rate_timestamps = []
        self.ws_min_process_rate = config.get("ws_min_process_rate", 30)
        self.process_rate_window = 1
        self.cleanup_task = None
        self.ws_queue = asyncio.PriorityQueue(maxsize=config.get("ws_queue_size", 10000))
        self.disk_buffer = Queue(maxsize=config.get("disk_buffer_size", 10000))
        self.buffer_dir = os.path.join(config["cache_dir"], "ws_buffer")
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.processed_timestamps = {}
        self.processed_timestamps_2h = {}
        self.symbol_priority = {}
        self.backup_ws_urls = config.get("backup_ws_urls", [])
        self.ws_latency = {}
        self.latency_log_interval = 3600
        self.restart_attempts = 0
        self.max_restart_attempts = 20
        # Maximum number of symbols to work with overall
        self.max_symbols = config.get("max_symbols", 50)
        # Start with the configured limit for dynamic adjustments
        self.max_subscriptions = self.max_symbols
        # Number of symbols to subscribe per WebSocket connection
        self.ws_subscription_batch_size = config.get("max_subscriptions_per_connection", 30)
        self.active_subscriptions = 0
        self.load_threshold = 0.8
        self.ws_pool = {}
        self.tasks = []

    async def get_atr(self, symbol: str) -> float:
        """Return the latest ATR value for a symbol, recalculating if missing."""
        indicators = self.indicators.get(symbol)
        if indicators and getattr(indicators, "atr", None) is not None:
            try:
                value = float(indicators.atr.iloc[-1])
                if value > 0:
                    return value
            except Exception:
                pass
        async with self.ohlcv_lock:
            if (
                "symbol" in self.ohlcv.index.names
                and symbol in self.ohlcv.index.get_level_values("symbol")
            ):
                df = self.ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                return 0.0
        if df.empty:
            return 0.0
        try:
            new_ind = IndicatorsCache(
                df.droplevel("symbol"),
                self.config,
                df["close"].pct_change().std(),
                "primary",
            )
            self.indicators[symbol] = new_ind
            if new_ind.atr is not None and not new_ind.atr.empty:
                return float(new_ind.atr.iloc[-1])
        except Exception as e:
            logger.error(f"Ошибка расчета ATR для {symbol}: {e}")
        return 0.0

    async def is_data_fresh(
        self, symbol: str, timeframe: str = "primary", max_delay: float = 60
    ) -> bool:
        """Return True if the most recent candle is within ``max_delay`` seconds."""
        try:
            df_lock = self.ohlcv_lock if timeframe == "primary" else self.ohlcv_2h_lock
            df = self.ohlcv if timeframe == "primary" else self.ohlcv_2h
            async with df_lock:
                if "symbol" in df.index.names and symbol in df.index.get_level_values("symbol"):
                    sub_df = df.xs(symbol, level="symbol", drop_level=False)
                else:
                    return False
            if sub_df.empty:
                return False
            last_ts = sub_df.index.get_level_values("timestamp")[-1]
            age = pd.Timestamp.now(tz="UTC") - last_ts
            return age.total_seconds() <= max_delay
        except Exception as e:
            logger.error(f"Error checking data freshness for {symbol}: {e}")
            return False

    async def load_initial(self):
        try:
            markets = await safe_api_call(self.exchange, "load_markets")
            self.usdt_pairs = await self.select_liquid_pairs(markets)
            logger.info(f"Найдено {len(self.usdt_pairs)} USDT-пар с высокой ликвидностью")
            tasks = []
            history_limit = self.config.get("min_data_length", 200)
            for symbol in self.usdt_pairs:
                orderbook = await self.fetch_orderbook(symbol)
                bid_volume = sum([bid[1] for bid in orderbook.get("bids", [])[:5]]) if orderbook.get("bids") else 0
                ask_volume = sum([ask[1] for ask in orderbook.get("asks", [])[:5]]) if orderbook.get("asks") else 0
                liquidity = min(bid_volume, ask_volume)
                self.symbol_priority[symbol] = -liquidity
                tasks.append(self.fetch_ohlcv_history(symbol, self.config["timeframe"], history_limit, cache_prefix=""))
                tasks.append(
                    self.fetch_ohlcv_history(
                        symbol, self.config["secondary_timeframe"], history_limit, cache_prefix="2h_"
                    )
                )
                tasks.append(self.fetch_funding_rate(symbol))
                tasks.append(self.fetch_open_interest(symbol))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, tuple) and len(result) == 2:
                    symbol, df = result
                    mod = i % 4
                    if mod == 0:
                        timeframe = self.config["timeframe"]
                    elif mod == 1:
                        timeframe = self.config["secondary_timeframe"]
                    else:
                        continue
                    if not check_dataframe_empty(df, f"load_initial {symbol} {timeframe}"):
                        df["symbol"] = symbol
                        df = df.set_index(["symbol", df.index])
                        await self.synchronize_and_update(
                            symbol,
                            df,
                            self.funding_rates.get(symbol, 0.0),
                            self.open_interest.get(symbol, 0.0),
                            {"imbalance": 0.0, "timestamp": time.time()},
                            timeframe="primary" if timeframe == self.config["timeframe"] else "secondary",
                        )
        except Exception as e:
            logger.error(f"Ошибка загрузки начальных данных: {e}")
            await self.telegram_logger.send_telegram_message(f"Ошибка загрузки данных: {e}")

    async def select_liquid_pairs(self, markets: Dict) -> List[str]:
        """Return top liquid USDT futures pairs only.

        Filters out spot markets by selecting symbols that contain a colon
        (``:``) or explicitly end with ``":USDT"``. Volume ranking and
        the configured top limit remain unchanged.
        """

        pair_volumes = []
        for symbol, market in markets.items():
            # Only consider active USDT-margined futures symbols
            if market.get("active") and symbol.endswith("USDT") and (":" in symbol or symbol.endswith(":USDT")):
                try:
                    ticker = await safe_api_call(self.exchange, "fetch_ticker", symbol)
                    volume = float(ticker.get("quoteVolume") or 0)
                except Exception as e:
                    logger.error(f"Ошибка получения тикера для {symbol}: {e}")
                    volume = 0.0
                pair_volumes.append((symbol, volume))

        pair_volumes.sort(key=lambda x: x[1], reverse=True)
        top_limit = self.config.get("max_symbols", 50)
        return [s for s, _ in pair_volumes[:top_limit]]

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_ohlcv_single(self, symbol: str, timeframe: str, limit: int = 200, cache_prefix: str = "") -> tuple:
        try:
            ohlcv = await safe_api_call(
                self.exchange,
                "fetch_ohlcv",
                symbol,
                timeframe,
                limit=limit,
            )
            if not ohlcv or len(ohlcv) < limit * 0.8:
                logger.warning(f"Неполные данные OHLCV для {symbol} ({timeframe}), получено {len(ohlcv)} из {limit}")
                return symbol, pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(np.float32)
            if len(df) >= 3:
                df = filter_outliers_zscore(df, "close")
            if df["close"].isna().sum() / len(df) > 0.05:
                logger.warning(
                    f"Слишком много пропусков в данных для {symbol} ({timeframe}) (>5%), использование forward-fill"
                )
                df = df.fillna(method="ffill")
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            max_gap = pd.Timedelta(timeframe).total_seconds() * 2
            if time_diffs.max() > max_gap:
                logger.warning(
                    f"Обнаружен значительный разрыв в данных для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
                )
                await self.telegram_logger.send_telegram_message(
                    f"⚠️ Разрыв в данных для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
                )
                return symbol, pd.DataFrame()
            df = df.interpolate(method="time", limit_direction="both")
            self.cache.save_cached_data(f"{cache_prefix}{symbol}", timeframe, df)
            return symbol, pd.DataFrame(df)
        except Exception as e:
            logger.error(f"Ошибка получения OHLCV для {symbol} ({timeframe}): {e}")
            return symbol, pd.DataFrame()

    async def fetch_ohlcv_history(self, symbol: str, timeframe: str, total_limit: int, cache_prefix: str = "") -> tuple:
        """Fetch extended OHLCV history by performing multiple requests."""
        try:
            all_data = []
            timeframe_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000)
            since = None
            remaining = total_limit
            # bybit allows up to 1000 candles per request, default to 200 on errors
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
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp")
                all_data.append(df)
                remaining -= len(df)
                if len(df) < limit:
                    break
                since = int(df.index[0].timestamp() * 1000) - timeframe_ms * limit
            if not all_data:
                return symbol, pd.DataFrame()
            df = pd.concat(all_data).sort_index().drop_duplicates()
            self.cache.save_cached_data(f"{cache_prefix}{symbol}", timeframe, df)
            return symbol, pd.DataFrame(df)
        except Exception as e:
            logger.error(f"Ошибка получения расширенной истории OHLCV для {symbol} ({timeframe}): {e}")
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
        except Exception as e:
            logger.error(f"Ошибка получения ставки финансирования для {symbol}: {e}")
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
                self.open_interest[symbol] = interest
            return interest
        except Exception as e:
            logger.error(f"Ошибка получения открытого интереса для {symbol}: {e}")
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
                logger.warning(f"Пустая книга ордеров для {symbol}, повторная попытка")
                raise Exception("Пустой ордербук")
            return orderbook
        except Exception as e:
            logger.error(f"Ошибка получения книги ордеров для {symbol}: {e}")
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
            if check_dataframe_empty(df, f"synchronize_and_update {symbol} {timeframe}"):
                logger.warning(f"Пустой DataFrame для {symbol} ({timeframe}), пропуск синхронизации")
                return
            if df["close"].isna().any() or (df["close"] <= 0).any():
                logger.warning(f"Некорректные данные для {symbol} ({timeframe}), пропуск")
                return
            if timeframe == "primary":
                async with self.ohlcv_lock:
                    if isinstance(self.ohlcv.index, pd.MultiIndex):
                        base = self.ohlcv.drop(symbol, level="symbol", errors="ignore")
                    else:
                        base = self.ohlcv
                    self.ohlcv = pd.concat([base, df], ignore_index=False).sort_index()
            else:
                async with self.ohlcv_2h_lock:
                    if isinstance(self.ohlcv_2h.index, pd.MultiIndex):
                        base = self.ohlcv_2h.drop(symbol, level="symbol", errors="ignore")
                    else:
                        base = self.ohlcv_2h
                    self.ohlcv_2h = pd.concat([base, df], ignore_index=False).sort_index()
            async with self.funding_lock:
                self.funding_rates[symbol] = funding_rate
            async with self.oi_lock:
                self.open_interest[symbol] = open_interest
            async with self.orderbook_lock:
                orderbook_df = pd.DataFrame([orderbook | {"symbol": symbol, "timestamp": time.time()}])
                self.orderbook = pd.concat([self.orderbook, orderbook_df], ignore_index=False)
            volatility = df["close"].pct_change().std() if not df.empty else 0.02
            cache_key = f"{symbol}_{timeframe}"
            if timeframe == "primary":
                async with self.ohlcv_lock:
                    if cache_key not in self.indicators_cache:
                        obj_ref = calc_indicators.remote(df.droplevel("symbol"), self.config, volatility, "primary")
                        self.indicators_cache[cache_key] = ray.get(obj_ref)
                    self.indicators[symbol] = self.indicators_cache[cache_key]
            else:
                async with self.ohlcv_2h_lock:
                    if cache_key not in self.indicators_cache_2h:
                        obj_ref = calc_indicators.remote(df.droplevel("symbol"), self.config, volatility, "secondary")
                        self.indicators_cache_2h[cache_key] = ray.get(obj_ref)
                    self.indicators_2h[symbol] = self.indicators_cache_2h[cache_key]
            self.cache.save_cached_data(f"{timeframe}_{symbol}", timeframe, df)
        except Exception as e:
            logger.error(f"Ошибка синхронизации данных для {symbol} ({timeframe}): {e}")

    async def cleanup_old_data(self):
        while True:
            try:
                async with self.cleanup_lock:
                    current_time = pd.Timestamp.now(tz="UTC")
                    async with self.ohlcv_lock:
                        if not self.ohlcv.empty:
                            threshold = current_time - pd.Timedelta(seconds=self.config["forget_window"])
                            self.ohlcv = self.ohlcv[self.ohlcv.index.get_level_values("timestamp") >= threshold]
                    async with self.ohlcv_2h_lock:
                        if not self.ohlcv_2h.empty:
                            threshold = current_time - pd.Timedelta(seconds=self.config["forget_window"])
                            self.ohlcv_2h = self.ohlcv_2h[
                                self.ohlcv_2h.index.get_level_values("timestamp") >= threshold
                            ]
                    async with self.orderbook_lock:
                        if not self.orderbook.empty and "timestamp" in self.orderbook.columns:
                            self.orderbook = self.orderbook[
                                self.orderbook["timestamp"] >= time.time() - self.config["forget_window"]
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
            except Exception as e:
                logger.error(f"Ошибка очистки данных: {e}")
                await asyncio.sleep(60)

    async def save_to_disk_buffer(self, priority, item):
        try:
            filename = os.path.join(self.buffer_dir, f"buffer_{time.time()}.pkl")
            with open(filename, "wb") as f:
                pickle.dump((priority, item), f)
            self.disk_buffer.put(filename)
            logger.info(f"Сообщение сохранено в дисковый буфер: {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения в дисковый буфер: {e}")

    async def load_from_disk_buffer(self):
        while not self.disk_buffer.empty():
            try:
                filename = self.disk_buffer.get()
                with open(filename, "rb") as f:
                    priority, item = pickle.load(f)
                await self.ws_queue.put((priority, item))
                os.remove(filename)
                logger.info(f"Сообщение загружено из дискового буфера: {filename}")
            except Exception as e:
                logger.error(f"Ошибка загрузки из дискового буфера: {e}")

    async def adjust_subscriptions(self):
        cpu_load = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_load = memory.percent
        current_rate = (
            len(self.process_rate_timestamps) / self.process_rate_window
            if self.process_rate_timestamps
            else self.ws_min_process_rate
        )
        if cpu_load > self.load_threshold * 100 or memory_load > self.load_threshold * 100:
            new_max = max(10, self.max_subscriptions // 2)
            logger.warning(
                f"Высокая нагрузка (CPU: {cpu_load}%, Memory: {memory_load}%), уменьшение подписок до {new_max}"
            )
            self.max_subscriptions = new_max
        elif current_rate < self.ws_min_process_rate:
            new_max = max(10, int(self.max_subscriptions * 0.8))
            logger.warning(f"Низкая скорость обработки ({current_rate:.2f}/s), уменьшение подписок до {new_max}")
            self.max_subscriptions = new_max
        elif (
            cpu_load < self.load_threshold * 50
            and memory_load < self.load_threshold * 50
            and current_rate > self.ws_min_process_rate * 1.5
        ):
            new_max = min(100, self.max_subscriptions * 2)
            logger.info(f"Низкая нагрузка, увеличение подписок до {new_max}")
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
                    t2 = asyncio.create_task(
                        self._subscribe_chunk(
                            chunk,
                            self.config["ws_url"],
                            self.config["ws_reconnect_interval"],
                            timeframe="secondary",
                        )
                    )
                    self.tasks.extend([t1, t2])
                self.tasks.append(asyncio.create_task(self._process_ws_queue()))
                self.tasks.append(asyncio.create_task(self.load_from_disk_buffer()))
                self.tasks.append(asyncio.create_task(self.monitor_load()))
                await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Ошибка подписки на WebSocket: {e}")
            await self.telegram_logger.send_telegram_message(f"Ошибка WebSocket: {e}")

    async def monitor_load(self):
        while True:
            try:
                await self.adjust_subscriptions()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Ошибка мониторинга нагрузки: {e}")
                await asyncio.sleep(60)

    def fix_symbol(self, symbol: str) -> str:
        """Normalize symbol for Bybit futures REST requests.

        Parameters
        ----------
        symbol : str
            Symbol in one of the supported formats, e.g. ``BTC/USDT`` or
            ``BTC/USDT:USDT``.

        Returns
        -------
        str
            Symbol formatted as ``BTC/USDT:USDT``. If the input already
            contains ``:USDT`` it will not be duplicated.
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
        while True:
            try:
                ohlcv = await self.pro_exchange.watch_ohlcv(symbol, timeframe)
                if not ohlcv:
                    continue
                last = ohlcv[-1]
                kline_timestamp = pd.to_datetime(int(last[0]), unit="ms", utc=True)
                df = pd.DataFrame([
                    {
                        "timestamp": kline_timestamp,
                        "open": np.float32(last[1]),
                        "high": np.float32(last[2]),
                        "low": np.float32(last[3]),
                        "close": np.float32(last[4]),
                        "volume": np.float32(last[5]),
                    }
                ])
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
            except Exception as e:
                reconnect_attempts += 1
                delay = min(2**reconnect_attempts, 60)
                logger.error(
                    f"Ошибка CCXT Pro для {symbol} ({label}), попытка {reconnect_attempts}/{max_reconnect_attempts}, ожидание {delay} секунд: {e}"
                )
                await asyncio.sleep(delay)
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(
                        f"Не удалось восстановить подписку CCXT Pro для {symbol} ({label})"
                    )
                    break

    async def _subscribe_with_ccxtpro(self, symbols: List[str]):
        self.tasks = []
        for symbol in symbols:
            t1 = asyncio.create_task(
                self._subscribe_symbol_ccxtpro(
                    symbol, self.config["timeframe"], "primary"
                )
            )
            t2 = asyncio.create_task(
                self._subscribe_symbol_ccxtpro(
                    symbol, self.config["secondary_timeframe"], "secondary"
                )
            )
            self.tasks.extend([t1, t2])
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
                if not self.ws_pool[url]:
                    ws = await websockets.connect(
                        url,
                        ping_interval=20,
                        ping_timeout=30,
                        open_timeout=max(connection_timeout, 10),
                    )
                    self.ws_pool[url].append(ws)
                else:
                    ws = self.ws_pool[url].pop(0)
                logger.info(f"Подключение к WebSocket {url}")
                return ws
            except Exception as e:
                attempts += 1
                delay = min(2 ** attempts, 60)
                logger.error(
                    f"Ошибка подключения к WebSocket {url}, попытка {attempts}/{max_attempts}, ожидание {delay} секунд: {e}"
                )
                if attempts >= max_attempts:
                    raise
                await asyncio.sleep(delay)

    async def _send_subscriptions(self, ws, symbols, timeframe: str):
        """Send subscription requests for ``symbols`` and confirm success."""
        attempts = 0
        max_attempts = self.config.get("max_reconnect_attempts", 10)
        selected_timeframe = (
            self.config["timeframe"] if timeframe == "primary" else self.config["secondary_timeframe"]
        )
        batch_size = self.ws_subscription_batch_size
        while True:
            try:
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i : i + batch_size]
                    for symbol in batch:
                        current_time = time.time()
                        self.ws_rate_timestamps.append(current_time)
                        self.ws_rate_timestamps = [t for t in self.ws_rate_timestamps if current_time - t < 1]
                        if len(self.ws_rate_timestamps) > self.config["ws_rate_limit"]:
                            logger.warning("Превышен лимит подписок WebSocket, ожидание")
                            await asyncio.sleep(1)
                            self.ws_rate_timestamps = [t for t in self.ws_rate_timestamps if current_time - t < 1]
                        ws_symbol = self.fix_ws_symbol(symbol)
                        await ws.send(
                            json.dumps({"op": "subscribe", "args": [f"kline.{selected_timeframe}.{ws_symbol}"]})
                        )
                        await asyncio.sleep(max(0, 1 / self.config["ws_rate_limit"]))

                confirmations_needed = len(symbols)
                confirmations = 0
                startup_messages = []
                start_confirm = time.time()
                while confirmations < confirmations_needed and time.time() - start_confirm < confirmations_needed * 5:
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
                    raise Exception("Подписка не подтверждена")
                return startup_messages
            except Exception as e:
                attempts += 1
                delay = min(2 ** attempts, 60)
                logger.error(
                    f"Ошибка подписки на WebSocket для {symbols} ({timeframe}), попытка {attempts}/{max_attempts}, ожидание {delay} секунд: {e}"
                )
                if attempts >= max_attempts:
                    raise
                await asyncio.sleep(delay)

    async def _read_messages(
        self,
        ws,
        symbols,
        timeframe: str,
        selected_timeframe: str,
        connection_timeout: int,
    ):
        """Read messages from ``ws`` and enqueue them for processing."""
        start_time = time.time()
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=connection_timeout)
                latency = time.time() - start_time
                for symbol in symbols:
                    self.ws_latency[symbol] = latency
                if latency > 5:
                    logger.warning(
                        f"Высокая задержка WebSocket для {symbols} ({timeframe}): {latency:.2f} сек"
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
                            if not check_dataframe_empty(df, f"subscribe_to_klines {symbol} {timeframe}"):
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
                try:
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    symbol = topic.split(".")[-1] if isinstance(topic, str) else ""
                    priority = self.symbol_priority.get(symbol, 0)
                    try:
                        await self.ws_queue.put((priority, (symbols, message, timeframe)), timeout=5)
                    except asyncio.TimeoutError:
                        logger.warning("Очередь WebSocket переполнена, сохранение в дисковый буфер")
                        await self.save_to_disk_buffer(priority, (symbols, message, timeframe))
                except asyncio.TimeoutError:
                    logger.warning("Очередь WebSocket переполнена, сохранение в дисковый буфер")
                    await self.save_to_disk_buffer(priority, (symbols, message, timeframe))
                    continue
            except asyncio.TimeoutError:
                logger.warning(f"Тайм-аут WebSocket для {symbols} ({timeframe}), отправка пинга")
                await ws.ping()
                continue
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket соединение закрыто для {symbols} ({timeframe}): {e}")
                break
            except Exception as e:
                logger.error(f"Ошибка обработки WebSocket сообщения для {symbols} ({timeframe}): {e}")
                break

    async def _subscribe_chunk(self, symbols, ws_url, connection_timeout, timeframe: str = "primary"):
        """Subscribe to kline data for a chunk of symbols."""
        reconnect_attempts = 0
        max_reconnect_attempts = self.config.get("max_reconnect_attempts", 10)
        urls = [ws_url] + self.backup_ws_urls
        current_url_index = 0
        selected_timeframe = self.config["timeframe"] if timeframe == "primary" else self.config["secondary_timeframe"]
        while True:
            current_url = urls[current_url_index % len(urls)]
            ws = None
            try:
                ws = await self._connect_ws(current_url, connection_timeout)
                self.active_subscriptions += len(symbols)
                self.restart_attempts = 0
                reconnect_attempts = 0
                current_url_index = 0

                startup_messages = await self._send_subscriptions(ws, symbols, timeframe)
                for message in startup_messages:
                    try:
                        data = json.loads(message)
                        topic = data.get("topic", "")
                        symbol = topic.split(".")[-1] if isinstance(topic, str) else ""
                        priority = self.symbol_priority.get(symbol, 0)
                        try:
                            await self.ws_queue.put((priority, (symbols, message, timeframe)), timeout=5)
                        except asyncio.TimeoutError:
                            logger.warning("Очередь WebSocket переполнена, сохранение в дисковый буфер")
                            await self.save_to_disk_buffer(priority, (symbols, message, timeframe))
                    except Exception:
                        continue

                await self._read_messages(ws, symbols, timeframe, selected_timeframe, connection_timeout)
            except Exception as e:
                reconnect_attempts += 1
                current_url_index += 1
                delay = min(2**reconnect_attempts, 60)
                logger.error(
                    f"Ошибка WebSocket {current_url} для {symbols} ({timeframe}), попытка {reconnect_attempts}/{max_reconnect_attempts}, ожидание {delay} секунд: {e}"
                )
                await asyncio.sleep(delay)
                if reconnect_attempts >= max_reconnect_attempts:
                    self.restart_attempts += 1
                    if self.restart_attempts >= self.max_restart_attempts:
                        logger.critical(
                            f"Превышено максимальное количество перезапусков WebSocket для {symbols} ({timeframe})"
                        )
                        await self.telegram_logger.send_telegram_message(
                            f"Критическая ошибка: Не удалось восстановить WebSocket для {symbols} ({timeframe})"
                        )
                        break
                    logger.info(
                        f"Автоматический перезапуск WebSocket для {symbols} ({timeframe}), попытка {self.restart_attempts}/{self.max_restart_attempts}"
                    )
                    reconnect_attempts = 0
                    current_url_index = 0
                    await asyncio.sleep(60)
                else:
                    logger.info(f"Попытка загрузки данных через REST API для {symbols} ({timeframe})")
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
                                if not check_dataframe_empty(df, f"subscribe_to_klines {symbol} {timeframe}"):
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
                        except Exception as rest_e:
                            logger.error(f"Ошибка REST API для {symbol} ({timeframe}): {rest_e}")
            finally:
                self.active_subscriptions -= len(symbols)
                if ws and ws.open:
                    self.ws_pool[current_url].append(ws)
                elif ws:
                    await ws.close()

    async def _process_ws_queue(self):
        last_latency_log = time.time()
        while True:
            try:
                priority, (symbols, message, timeframe) = await self.ws_queue.get()
                now = time.time()
                self.process_rate_timestamps.append(now)
                self.process_rate_timestamps = [
                    t for t in self.process_rate_timestamps if now - t < self.process_rate_window
                ]
                if (
                    len(self.process_rate_timestamps) > self.ws_min_process_rate
                    and (len(self.process_rate_timestamps) / self.process_rate_window) < self.ws_min_process_rate
                ):
                    await self.adjust_subscriptions()
                data = json.loads(message)
                if (
                    not isinstance(data, dict)
                    or "topic" not in data
                    or "data" not in data
                    or not isinstance(data["data"], list)
                ):
                    logger.debug(f"Error in message format for {symbols}: {message}")
                    continue
                topic = data.get("topic", "")
                symbol = topic.split(".")[-1] if isinstance(topic, str) else ""
                if not symbol:
                    logger.debug(f"Symbol not found in topic for message: {message}")
                    continue
                for entry in data["data"]:
                    required_fields = ["start", "open", "high", "low", "close", "volume"]
                    if not all(field in entry for field in required_fields):
                        logger.warning(f"Invalid kline data ({timeframe}): {entry}")
                        continue
                    try:
                        kline_timestamp = pd.to_datetime(int(entry["start"]), unit="ms", utc=True)
                        open_price = float(entry["open"])
                        high_price = float(entry["high"])
                        low_price = float(entry["low"])
                        close_price = float(entry["close"])
                        volume = float(entry["volume"])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ошибка формата данных свечи для {symbol} ({timeframe}): {e}")
                        continue
                    timestamp_dict = (
                        self.processed_timestamps if timeframe == "primary" else self.processed_timestamps_2h
                    )
                    lock = self.ohlcv_lock if timeframe == "primary" else self.ohlcv_2h_lock
                    async with lock:
                        if symbol not in timestamp_dict:
                            timestamp_dict[symbol] = set()
                        if entry["start"] in timestamp_dict[symbol]:
                            logger.debug(
                                f"Дубликат сообщения для {symbol} ({timeframe}) с временной меткой {kline_timestamp}"
                            )
                            continue
                        timestamp_dict[symbol].add(entry["start"])
                        if len(timestamp_dict[symbol]) > 1000:
                            timestamp_dict[symbol] = set(list(timestamp_dict[symbol])[-500:])
                    current_time = pd.Timestamp.now(tz="UTC")
                    if (current_time - kline_timestamp).total_seconds() > 5:
                        logger.warning(f"Получены устаревшие данные для {symbol} ({timeframe}): {kline_timestamp}")
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
                            logger.warning(f"Данные для {symbol} ({timeframe}) отфильтрованы как аномалии")
                            continue
                        df["symbol"] = symbol
                        df = df.set_index(["symbol", "timestamp"])
                        time_diffs = df.index.get_level_values("timestamp").to_series().diff().dt.total_seconds()
                        max_gap = (
                            pd.Timedelta(
                                self.config["timeframe" if timeframe == "primary" else "secondary_timeframe"]
                            ).total_seconds()
                            * 2
                        )
                        if time_diffs.max() > max_gap:
                            logger.warning(
                                f"Обнаружен разрыв в данных WebSocket для {symbol} ({timeframe}): {time_diffs.max()/60:.2f} минут"
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
                    except Exception as e:
                        logger.error(f"Ошибка обработки данных для {symbol}: {e}")
                        continue
                if time.time() - last_latency_log > self.latency_log_interval:
                    rate = len(self.process_rate_timestamps) / self.process_rate_window
                    logger.info(
                        f"Средняя задержка WebSocket: {sum(self.ws_latency.values()) / len(self.ws_latency):.2f} сек, скорость обработки: {rate:.2f}/с"
                    )
                    last_latency_log = time.time()
            except Exception as e:
                logger.error(f"Ошибка обработки очереди WebSocket: {e}")
                await asyncio.sleep(2)
            finally:
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
                except Exception as e:
                    logger.error(f"Ошибка закрытия WebSocket {url}: {e}")
        self.ws_pool.clear()

        if self.pro_exchange is not None and hasattr(self.pro_exchange, "close"):
            try:
                await self.pro_exchange.close()
            except Exception as e:
                logger.error(f"Ошибка закрытия ccxtpro: {e}")

        await TelegramLogger.shutdown()

# ----------------------------------------------------------------------
# REST API for minimal integration testing
# ----------------------------------------------------------------------

api_app = Flask(__name__)
PRICES = {"TEST": 100.0}


@api_app.route("/price/<symbol>")
def price(symbol: str):
    price = PRICES.get(symbol, 0.0)
    return jsonify({"price": price})


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting DataHandler service on port %s", port)
    api_app.run(host="0.0.0.0", port=port)
