"""Historical simulation running TradeManager logic."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from bot.utils import logger

sys.modules.setdefault("simulation", sys.modules[__name__])
sys.modules.setdefault("bot.simulation", sys.modules[__name__])


@dataclass(frozen=True)
class SimulationResult:
    """Structured information returned after a simulation run."""

    start: pd.Timestamp
    end: pd.Timestamp
    processed_symbols: List[str]
    missing_symbols: List[str]
    total_iterations: int
    total_updates: int


class SimulationDataError(RuntimeError):
    """Raised when the simulator cannot run due to missing historical data."""


class HistoricalSimulator:
    """Replay historical candles and execute TradeManager methods."""

    def __init__(self, data_handler, trade_manager) -> None:
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.history: Dict[str, pd.DataFrame] = {}

    async def load(
        self, start_ts: pd.Timestamp, end_ts: pd.Timestamp
    ) -> Tuple[List[str], List[str]]:
        """Load cached OHLCV data for all symbols.

        Returns:
            A tuple consisting of a list of symbols that were successfully loaded
            and a list of symbols for which no data was found in the requested
            range.
        """
        self.history.clear()
        loaded_symbols: List[str] = []
        missing_symbols: List[str] = []
        config = getattr(self.data_handler, "config", None)
        timeframe = "1m"
        if config is not None:
            if hasattr(config, "get") and callable(config.get):
                timeframe = config.get("timeframe", timeframe)
            else:
                timeframe = getattr(config, "timeframe", timeframe)
        for symbol in self.data_handler.usdt_pairs:
            df = None
            if hasattr(self.data_handler, "history"):
                candidate = self.data_handler.history
                if isinstance(candidate, pd.DataFrame):
                    df = candidate
                    if "symbol" in df.index.names:
                        df = df.xs(symbol, level="symbol", drop_level=False)
            else:
                cache = getattr(self.data_handler, "cache", None)
                if cache:
                    df = cache.load_cached_data(symbol, timeframe)
            if df is None:
                if symbol not in missing_symbols:
                    missing_symbols.append(symbol)
                continue
            if isinstance(df.index, pd.MultiIndex):
                ts_level = "timestamp" if "timestamp" in df.index.names else df.index.names[0]
                if ts_level != "timestamp":
                    logger.warning("Timestamp level not found in DataFrame index. Using '%s' instead", ts_level)
                idx = df.index.get_level_values(ts_level)
            else:
                idx = df.index
            df = df[(idx >= start_ts) & (idx <= end_ts)]
            if df.empty:
                if symbol not in missing_symbols:
                    missing_symbols.append(symbol)
                continue
            self.history[symbol] = df
            loaded_symbols.append(symbol)
        return loaded_symbols, missing_symbols

    async def _manage_positions_once(self) -> None:
        async with self.trade_manager.position_lock:
            idx_names = getattr(self.trade_manager.positions.index, "names", [])
            if "symbol" not in idx_names:
                return
            symbols = self.trade_manager.positions.index.get_level_values(
                "symbol"
            ).unique()
        for symbol in symbols:
            df = self.history.get(symbol)
            if df is None:
                df = getattr(self.data_handler, "history", None)
            if df is None:
                continue
            if symbol in getattr(df.index, "names", []):
                df = df.xs(symbol, level="symbol", drop_level=False)
            elif getattr(df.index, "names", None) == ["timestamp"]:
                pass
            elif isinstance(df.index, pd.MultiIndex) and "timestamp" in df.index.names:
                pass
            elif "symbol" in getattr(df, "columns", []):
                df = df[df["symbol"] == symbol].set_index("timestamp")
            if df.empty:
                continue
            price = df["close"].iloc[-1]
            idx_names = getattr(self.trade_manager.positions.index, "names", [])
            if self.trade_manager._has_position(symbol):
                await self.trade_manager.check_trailing_stop(symbol, price)
            if self.trade_manager._has_position(symbol):
                await self.trade_manager.check_stop_loss_take_profit(symbol, price)
            if self.trade_manager._has_position(symbol):
                await self.trade_manager.check_exit_signal(symbol, price)

    async def run(
        self, start_ts: pd.Timestamp, end_ts: pd.Timestamp, speed: float = 1.0
    ) -> SimulationResult:
        loaded_symbols, missing_symbols = await self.load(start_ts, end_ts)
        if not self.history:
            missing = f" for symbols: {', '.join(missing_symbols)}" if missing_symbols else ""
            message = (
                "No cached OHLCV data found between "
                f"{start_ts} and {end_ts}{missing}."
            )
            raise SimulationDataError(message)
        timestamps = sorted({
            ts
            for df in self.history.values()
            for ts in (
                df.index.get_level_values("timestamp")
                if isinstance(df.index, pd.MultiIndex)
                else df.index
            )
        })
        total_updates = 0
        for i, ts in enumerate(timestamps):
            for symbol, df in self.history.items():
                if isinstance(df.index, pd.MultiIndex):
                    ts_index = df.index.get_level_values("timestamp")
                else:
                    ts_index = df.index
                if ts in ts_index:
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
                    total_updates += len(row)
            for symbol in self.data_handler.usdt_pairs:
                signal = await self.trade_manager.evaluate_signal(symbol)
                if signal:
                    history_df = self.history.get(symbol)
                    if history_df is None:
                        continue
                    if isinstance(history_df.index, pd.MultiIndex):
                        idx = history_df.index.get_level_values("timestamp")
                    else:
                        idx = history_df.index
                    price_subset = history_df.loc[idx <= ts]
                    if price_subset.empty:
                        continue
                    price = price_subset["close"].iloc[-1]
                    params = await self.data_handler.parameter_optimizer.optimize(symbol)
                    await self.trade_manager.open_position(symbol, signal, float(price), params)
            await self._manage_positions_once()
            if i < len(timestamps) - 1:
                dt = (timestamps[i + 1] - ts).total_seconds() / max(speed, 1e-6)
                if dt > 0:
                    await asyncio.sleep(dt)
        return SimulationResult(
            start=start_ts,
            end=end_ts,
            processed_symbols=sorted(loaded_symbols),
            missing_symbols=sorted(missing_symbols),
            total_iterations=len(timestamps),
            total_updates=total_updates,
        )
