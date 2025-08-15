"""Periodically checks open positions and manages risk controls."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Optional

from bot.utils import check_dataframe_empty_async as _check_df_async, logger


class PositionManager:
    """Monitor open positions and trigger close requests.

    Parameters
    ----------
    trade_manager : TradeManager
        Trading engine responsible for executing orders.
    data_handler : DataHandler
        Component providing market data and indicators.
    check_interval : float, optional
        Delay between position checks in seconds, by default ``1.0``.
    """

    def __init__(self, trade_manager, data_handler, check_interval: float = 1.0) -> None:
        self.trade_manager = trade_manager
        self.data_handler = data_handler
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start periodic position monitoring."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop position monitoring."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        """Internal loop checking positions at regular intervals."""
        while True:
            try:
                await self._check_positions()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                raise
            except (ValueError, RuntimeError, KeyError) as exc:
                logger.exception("PositionManager failed (%s): %s", type(exc).__name__, exc)
                await asyncio.sleep(self.check_interval)

    async def _check_positions(self) -> None:
        """Iterate through open positions and evaluate exit conditions."""
        async with self.trade_manager.position_lock:
            symbols = []
            if "symbol" in self.trade_manager.positions.index.names:
                symbols = self.trade_manager.positions.index.get_level_values("symbol").unique()

        for symbol in symbols:
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            empty = await _check_df_async(df, f"position_manager {symbol}")
            if empty:
                continue
            current_price = df["close"].iloc[-1]

            if (
                "symbol" in self.trade_manager.positions.index.names
                and symbol in self.trade_manager.positions.index.get_level_values("symbol")
            ):
                res = self.trade_manager.check_stop_loss_take_profit(symbol, current_price)
                if inspect.isawaitable(res):
                    await res

            if (
                "symbol" in self.trade_manager.positions.index.names
                and symbol in self.trade_manager.positions.index.get_level_values("symbol")
            ):
                res = self.trade_manager.check_trailing_stop(symbol, current_price)
                if inspect.isawaitable(res):
                    await res
