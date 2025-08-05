"""Historical simulation running TradeManager logic."""

from __future__ import annotations

import asyncio
import pandas as pd
from typing import Dict

from bot.utils import logger


class HistoricalSimulator:
    """Replay historical candles and execute TradeManager methods."""

    def __init__(self, data_handler, trade_manager) -> None:
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.history: Dict[str, pd.DataFrame] = {}

    async def load(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
        """Load cached OHLCV data for all symbols."""
        timeframe = self.data_handler.config.get("timeframe", "1m")
        for symbol in self.data_handler.usdt_pairs:
            df = None
            if hasattr(self.data_handler, "history"):
                df = self.data_handler.history
                if "symbol" in df.index.names:
                    df = df.xs(symbol, level="symbol", drop_level=False)
            else:
                cache = getattr(self.data_handler, "cache", None)
                if cache:
                    df = cache.load_cached_data(symbol, timeframe)
            if df is None:
                continue
            if isinstance(df.index, pd.MultiIndex):
                ts_level = "timestamp" if "timestamp" in df.index.names else df.index.names[0]
                if ts_level != "timestamp":
                    logger.warning("Timestamp level not found in DataFrame index. Using '%s' instead", ts_level)
                idx = df.index.get_level_values(ts_level)
            else:
                idx = df.index
            df = df[(idx >= start_ts) & (idx <= end_ts)]
            self.history[symbol] = df

    async def _manage_positions_once(self) -> None:
        async with self.trade_manager.position_lock:
            idx_names = getattr(self.trade_manager.positions.index, "names", [])
            if "symbol" not in idx_names:
                return
            symbols = self.trade_manager.positions.index.get_level_values(
                "symbol"
            ).unique()
        for symbol in symbols:
            ohlcv = self.data_handler.ohlcv
            if "symbol" in ohlcv.index.names and symbol in ohlcv.index.get_level_values("symbol"):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                continue
            if df.empty:
                continue
            price = df["close"].iloc[-1]
            idx_names = getattr(self.trade_manager.positions.index, "names", [])
            if "symbol" in idx_names and symbol in self.trade_manager.positions.index.get_level_values("symbol"):
                await self.trade_manager.check_trailing_stop(symbol, price)
            if "symbol" in idx_names and symbol in self.trade_manager.positions.index.get_level_values("symbol"):
                await self.trade_manager.check_stop_loss_take_profit(symbol, price)
            if "symbol" in idx_names and symbol in self.trade_manager.positions.index.get_level_values("symbol"):
                await self.trade_manager.check_exit_signal(symbol, price)

    async def run(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp, speed: float = 1.0) -> None:
        await self.load(start_ts, end_ts)
        if not self.history:
            logger.warning(
                "No cached OHLCV data found between %s and %s; simulation aborted",
                start_ts,
                end_ts,
            )
            return
        timestamps = sorted({
            ts
            for df in self.history.values()
            for ts in (
                df.index.get_level_values("timestamp")
                if isinstance(df.index, pd.MultiIndex)
                else df.index
            )
        })
        for i, ts in enumerate(timestamps):
            for symbol, df in self.history.items():
                if ts in df.index:
                    if isinstance(df.index, pd.MultiIndex):
                        row = df.loc[df.index.get_level_values("timestamp") == ts]
                        if list(row.index.names) != ["symbol", "timestamp"]:
                            if "symbol" in row.index.names and "timestamp" in row.index.names:
                                row = row.swaplevel("timestamp", "symbol")
                            else:
                                row.index.names = ["timestamp", "symbol"]
                                row = row.swaplevel(0, 1)
                    else:
                        row = df.loc[[ts]]
                        row = row.assign(symbol=symbol)
                        row.index.name = "timestamp"
                        row = row.set_index("symbol", append=True).swaplevel(0, 1)
                    await self.data_handler.synchronize_and_update(
                        symbol,
                        row,
                        self.data_handler.funding_rates.get(symbol, 0.0),
                        self.data_handler.open_interest.get(symbol, 0.0),
                        {"bids": [], "asks": []},
                    )
            for symbol in self.data_handler.usdt_pairs:
                signal = await self.trade_manager.evaluate_signal(symbol)
                if signal:
                    ohlcv = self.data_handler.ohlcv
                    if "symbol" in ohlcv.index.names and symbol in ohlcv.index.get_level_values("symbol"):
                        price_df = ohlcv.xs(symbol, level="symbol")
                        if isinstance(price_df.index, pd.MultiIndex):
                            mask = price_df.index.get_level_values("timestamp") <= ts
                        else:
                            mask = price_df.index <= ts
                        price_subset = price_df.loc[mask]
                        if price_subset.empty:
                            continue
                        price = price_subset["close"].iloc[-1]
                    else:
                        continue
                    params = await self.data_handler.parameter_optimizer.optimize(symbol)
                    await self.trade_manager.open_position(symbol, signal, float(price), params)
            await self._manage_positions_once()
            if i < len(timestamps) - 1:
                dt = (timestamps[i + 1] - ts).total_seconds() / max(speed, 1e-6)
                if dt > 0:
                    await asyncio.sleep(dt)
