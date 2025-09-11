"""Core data handler implementation used in tests."""
from __future__ import annotations

import asyncio
import json
import time
import types
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - optional
    pl = None  # type: ignore

from .utils import expected_ws_rate, ensure_utc


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
        if getattr(cfg, "use_polars", False) and pl is not None:
            self._ohlcv = pl.DataFrame()
            self._ohlcv_2h = pl.DataFrame()
        else:
            self._ohlcv = pd.DataFrame()
            self._ohlcv_2h = pd.DataFrame()

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
            self._ohlcv = pl.DataFrame(subset.to_dict("list"))

    async def cleanup_old_data(self) -> None:
        while True:
            await asyncio.sleep(getattr(self.cfg, "data_cleanup_interval", 1))
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=getattr(self.cfg, "forget_window", 0))
            if pl is not None and isinstance(self._ohlcv, pl.DataFrame) and self._ohlcv.height > 0:
                self._ohlcv = self._ohlcv.filter(pl.col("timestamp") >= cutoff)
            if pl is not None and isinstance(self._ohlcv_2h, pl.DataFrame) and self._ohlcv_2h.height > 0:
                self._ohlcv_2h = self._ohlcv_2h.filter(pl.col("timestamp") >= cutoff)
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
