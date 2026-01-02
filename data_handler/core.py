"""Core data handler implementation used in tests."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import types
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore

from .utils import expected_ws_rate


def _normalise_polars_timestamp(df: Any) -> Any:
    """Ensure ``timestamp`` columns are comparable in Polars DataFrames."""

    if pl is None or not isinstance(df, pl.DataFrame) or df.height == 0:
        return df
    if "timestamp" not in df.columns:
        return df
    return df.with_columns(
        pl.col("timestamp")
        .map_elements(
            lambda ts: ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        )
        .cast(pl.Datetime(time_zone="UTC"), strict=False)
    )


class DataHandler:
    """Simplified DataHandler focused on functionality used in tests."""

    def __init__(
        self,
        cfg: Any,
        http_client: Any,
        optimizer: Any,
        exchange: Any = None,
    ) -> None:
        self.cfg = cfg
        self.exchange = exchange
        self.ws_queue: asyncio.PriorityQueue[Tuple[int, Any]] = asyncio.PriorityQueue()
        self.ws_min_process_rate = expected_ws_rate(cfg.timeframe)
        self.disk_buffer: Dict[int, list] = {}
        self.indicators: Dict[str, Any] = {}
        configured_pairs = self._resolve_configured_pairs()
        self._pairs_from_config = bool(configured_pairs)
        self._pairs_discovered = False
        self.usdt_pairs = configured_pairs or ["BTCUSDT"]
        self._ohlcv: Any
        self._ohlcv_2h: Any
        self.logger = logging.getLogger(__name__)
        if getattr(cfg, "use_polars", False) and pl is not None:
            self._ohlcv = pl.DataFrame()
            self._ohlcv_2h = pl.DataFrame()
        else:
            self._ohlcv = pd.DataFrame()
            self._ohlcv_2h = pd.DataFrame()

    @property
    def ohlcv(self) -> Any:
        """Return the primary OHLCV history."""

        return self._ohlcv

    @ohlcv.setter
    def ohlcv(self, value: Any) -> None:
        self._ohlcv = value

    @property
    def ohlcv_2h(self) -> Any:
        """Return the secondary timeframe OHLCV history."""

        return self._ohlcv_2h

    @ohlcv_2h.setter
    def ohlcv_2h(self, value: Any) -> None:
        self._ohlcv_2h = value

    def _resolve_configured_pairs(self) -> list[str]:
        symbols = getattr(self.cfg, "symbols", None)
        if symbols is None:
            return []
        if isinstance(symbols, (str, bytes)):
            symbols = [symbols]
        elif not isinstance(symbols, Iterable):
            return []
        pairs = [
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        ]
        # Preserve order while removing duplicates
        return list(dict.fromkeys(pairs))

    async def _load_markets(self) -> Dict[str, Dict[str, Any]]:
        if self.exchange is None:
            return {}
        loader = getattr(self.exchange, "load_markets", None)
        if loader is None:
            return {}
        try:
            markets = loader()
            if asyncio.iscoroutine(markets):
                markets = await markets
            if markets is None:
                return {}
            if isinstance(markets, dict):
                return markets
            return dict(markets)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.debug("Не удалось загрузить рынки: %s", exc)
            return {}

    async def _discover_usdt_pairs(self) -> None:
        if self._pairs_from_config or self._pairs_discovered:
            return
        if self.exchange is None or not hasattr(self.exchange, "fetch_ticker"):
            self._pairs_discovered = True
            return
        markets = await self._load_markets()
        if not markets:
            self._pairs_discovered = True
            return
        try:
            discovered = await self.select_liquid_pairs(markets)
        except Exception as exc:  # pragma: no cover - runtime protection
            self.logger.warning("Не удалось определить ликвидные пары: %s", exc)
            self._pairs_discovered = True
            return
        if discovered:
            self.usdt_pairs = list(dict.fromkeys(discovered))
        self._pairs_discovered = True

    async def select_liquid_pairs(self, markets: Dict[str, Dict[str, Any]]) -> list[str]:
        results: Dict[str, Tuple[str, float]] = {}
        for name, info in markets.items():
            if not info.get("active"):
                continue
            if not info.get("contract"):
                continue
            if info.get("quote") != "USDT":
                continue
            ticker = await self.exchange.fetch_ticker(name)
            volume = float(ticker.get("quoteVolume", 0))
            if volume < self.cfg.min_liquidity:
                continue
            launch = info.get("info", {}).get("launchTime")
            if launch:
                age_ms = time.time() * 1000 - launch
                tf_ms = pd.Timedelta(self.cfg.timeframe).total_seconds() * 1000
                min_age = self.cfg.min_data_length * tf_ms
                if age_ms < min_age:
                    continue
            base = name.split(":")[0].replace("/", "").upper()
            existing = results.get(base)
            if existing is None or volume > existing[1]:
                results[base] = (name, volume)
        if not results:
            raise ValueError("no liquid markets found")
        ordered = sorted(results.values(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in ordered[: self.cfg.max_symbols]]

    async def _send_subscriptions(self, ws, pairs: Iterable[str], tag: str) -> None:
        for p in pairs:
            msg = json.dumps({"op": "subscribe", "args": [p]})
            await ws.send(msg)

    async def save_to_disk_buffer(self, priority: int, item: Any) -> None:
        self.disk_buffer.setdefault(priority, []).append(item)

    async def load_from_disk_buffer_loop(self) -> None:
        while True:
            for priority in sorted(list(self.disk_buffer.keys())):
                items = self.disk_buffer.pop(priority)
                for it in items:
                    await self.ws_queue.put((priority, it))
            await asyncio.sleep(0.1)

    async def synchronize_and_update(self, symbol: str, df: pd.DataFrame, *_args) -> None:
        pdf = df.reset_index()
        if "timestamp" not in pdf.columns:
            pdf.rename(columns={pdf.columns[1]: "timestamp"}, inplace=True)
        span = getattr(self.cfg, "ema30_period", 30)
        pdf["ema30"] = pdf["close"].ewm(span=span, adjust=False).mean().shift(1)
        self.indicators[symbol] = types.SimpleNamespace(df=pdf)
        if getattr(self.cfg, "use_polars", False) and pl is not None:
            subset = pdf[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
            data = subset.to_dict("list")
            polars_columns: Dict[str, Any] = {}
            for name, values in data.items():
                if name == "timestamp":
                    converted = [
                        value.to_pydatetime()
                        if hasattr(value, "to_pydatetime")
                        else value
                        for value in values
                    ]
                    polars_columns[name] = pl.Series(
                        name,
                        converted,
                        dtype=pl.Datetime(time_zone="UTC"),
                    )
                else:
                    polars_columns[name] = values
            self._ohlcv = pl.DataFrame(polars_columns)
            self._ohlcv = _normalise_polars_timestamp(self._ohlcv)

    async def cleanup_old_data(self) -> None:
        while True:
            await asyncio.sleep(getattr(self.cfg, "data_cleanup_interval", 1))
            forget_window = getattr(self.cfg, "forget_window", 0)
            if forget_window <= 0:
                continue
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=forget_window)
            cutoff_dt = cutoff.to_pydatetime()
            if pl is not None and isinstance(self._ohlcv, pl.DataFrame) and self._ohlcv.height > 0:
                self._ohlcv = _normalise_polars_timestamp(self._ohlcv)
                self._ohlcv = self._ohlcv.filter(pl.col("timestamp") >= cutoff_dt)
            if pl is not None and isinstance(self._ohlcv_2h, pl.DataFrame) and self._ohlcv_2h.height > 0:
                self._ohlcv_2h = _normalise_polars_timestamp(self._ohlcv_2h)
                self._ohlcv_2h = self._ohlcv_2h.filter(pl.col("timestamp") >= cutoff_dt)
            if isinstance(self._ohlcv, pd.DataFrame) and not self._ohlcv.empty:
                ts_index = self._ohlcv.index.get_level_values("timestamp")
                self._ohlcv = self._ohlcv[ts_index >= cutoff]
                history = getattr(self.cfg, "history_retention", 0)
                if history > 0 and len(self._ohlcv) > history:
                    self._ohlcv = self._ohlcv.groupby(level="symbol").tail(history)
            if isinstance(self._ohlcv_2h, pd.DataFrame) and not self._ohlcv_2h.empty:
                ts_index = self._ohlcv_2h.index.get_level_values("timestamp")
                self._ohlcv_2h = self._ohlcv_2h[ts_index >= cutoff]
                history = getattr(self.cfg, "history_retention", 0)
                if history > 0 and len(self._ohlcv_2h) > history:
                    self._ohlcv_2h = self._ohlcv_2h.groupby(level="symbol").tail(history)

    async def fetch_ohlcv_history(self, symbol: str, timeframe: str, limit: int) -> Tuple[str, pd.DataFrame]:
        """Fetch OHLCV history for ``symbol`` from the configured exchange."""

        if not hasattr(self.exchange, "fetch_ohlcv"):
            raise RuntimeError("exchange does not support fetch_ohlcv")

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["symbol"] = symbol
            df = df.set_index(["symbol", "timestamp"])
        return symbol, df

    async def load_initial(self) -> None:
        """Load initial OHLCV history and populate indicators."""

        await self._discover_usdt_pairs()
        symbols = getattr(self, "usdt_pairs", None) or ["BTCUSDT"]
        self.usdt_pairs = list(dict.fromkeys(symbols))
        timeframe = getattr(self.cfg, "timeframe", "1m")
        secondary_timeframe = getattr(self.cfg, "secondary_timeframe", "2h")
        history_limit = int(getattr(self.cfg, "history_retention", 200))

        frames: list[pd.DataFrame] = []
        frames_secondary: list[pd.DataFrame] = []
        for sym in self.usdt_pairs:
            try:
                _sym, df = await self.fetch_ohlcv_history(sym, timeframe, history_limit)
            except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
                self.logger.warning("Не удалось загрузить историю %s: %s", sym, exc)
                continue
            if not df.empty:
                frames.append(df)
                await self.synchronize_and_update(sym, df.reset_index())

            try:
                _sym, df_secondary = await self.fetch_ohlcv_history(
                    sym, secondary_timeframe, max(1, history_limit // 2)
                )
            except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
                self.logger.debug("Пропускаем загрузку вторичного таймфрейма %s: %s", sym, exc)
                df_secondary = pd.DataFrame()
            if not df_secondary.empty:
                frames_secondary.append(df_secondary)

        if frames:
            self._ohlcv = pd.concat(frames).sort_index()
        if frames_secondary:
            self._ohlcv_2h = pd.concat(frames_secondary).sort_index()

    async def subscribe_to_klines(self, pairs: Iterable[str]) -> None:
        """Poll exchange klines and enqueue updates into the websocket queue."""

        if not hasattr(self.exchange, "fetch_ohlcv"):
            raise AttributeError("exchange does not implement fetch_ohlcv")

        timeframe = getattr(self.cfg, "timeframe", "1m")
        sleep_interval = max(1.0, pd.Timedelta(timeframe).total_seconds())
        try:
            while True:
                for sym in pairs:
                    try:
                        _, df = await self.fetch_ohlcv_history(sym, timeframe, 1)
                    except Exception as exc:  # pragma: no cover - runtime protection
                        self.logger.debug("Не удалось получить обновление свечи %s: %s", sym, exc)
                        continue
                    if df.empty:
                        continue
                    candle = df.reset_index().iloc[-1].to_dict()
                    await self.ws_queue.put((0, ([sym], candle, "primary")))
                await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise
