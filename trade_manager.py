"""Trading engine for executing and managing positions.

This module coordinates order placement, risk management and Telegram
notifications while interacting with the :class:`ModelBuilder` and exchange.
"""

import asyncio
import atexit
import signal
import ray
import pandas as pd
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
import inspect
from utils import (
    logger,
    TelegramLogger,
    is_cuda_available,
    check_dataframe_empty_async as _check_df_async,
    safe_api_call,
)
from config import BotConfig, load_config
import torch
import joblib
import os
import time
from typing import Dict, Optional, Tuple
import shutil
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import threading
import multiprocessing as mp

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# Determine computation device once

device_type = "cuda" if is_cuda_available() else "cpu"


def _register_cleanup_handlers(tm: "TradeManager") -> None:
    """Register atexit and signal handlers for graceful shutdown."""

    if getattr(tm, "_cleanup_registered", False):
        return

    tm._cleanup_registered = True

    def _handler(*_args):
        logger.info("Stopping TradeManager")
        tm.shutdown()
        try:
            asyncio.run(TelegramLogger.shutdown())
        except RuntimeError:
            # event loop may already be closed
            pass

    atexit.register(_handler)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda s, f: _handler())
        except ValueError:
            # signal may fail if not in main thread
            pass


class TradeManager:
    """Handles trading logic and sends Telegram notifications.

    Parameters
    ----------
    config : dict
        Bot configuration.
    data_handler : DataHandler
        Instance providing market data.
    model_builder
        Associated ModelBuilder instance.
    telegram_bot : telegram.Bot or compatible
        Bot used to send messages.
    chat_id : str | int
        Telegram chat identifier.
    rl_agent : optional
        Reinforcement learning agent used for decisions.
    """

    def __init__(
        self,
        config: BotConfig,
        data_handler,
        model_builder,
        telegram_bot,
        chat_id,
        rl_agent=None,
    ):
        self.config = config
        self.data_handler = data_handler
        self.model_builder = model_builder
        self.rl_agent = rl_agent
        if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; Telegram alerts will not be sent"
            )
        self.telegram_logger = TelegramLogger(
            telegram_bot,
            chat_id,
            max_queue_size=config.get("telegram_queue_size"),
        )
        self.positions = pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "size",
                "entry_price",
                "tp_multiplier",
                "sl_multiplier",
                "highest_price",
                "lowest_price",
                "breakeven_triggered",
            ],
            index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
        )
        self.returns_by_symbol = {symbol: [] for symbol in data_handler.usdt_pairs}
        self.position_lock = asyncio.Lock()
        self.returns_lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.exchange = data_handler.exchange
        self.max_positions = config.get("max_positions", 5)
        self.leverage = config.get("leverage", 10)
        self.min_risk_per_trade = config.get("min_risk_per_trade", 0.01)
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.05)
        self.check_interval = config.get("check_interval", 60)
        self.performance_window = config.get("performance_window", 86400)
        self.state_file = os.path.join(config["cache_dir"], "trade_manager_state.pkl")
        self.returns_file = os.path.join(
            config["cache_dir"], "trade_manager_returns.pkl"
        )
        self.last_save_time = time.time()
        self.save_interval = 900
        self.positions_changed = False
        self.last_volatility = {symbol: 0.0 for symbol in data_handler.usdt_pairs}
        self.load_state()

    async def compute_risk_per_trade(self, symbol: str, volatility: float) -> float:
        base_risk = self.config.get("risk_per_trade", self.min_risk_per_trade)
        async with self.returns_lock:
            returns = [
                r
                for t, r in self.returns_by_symbol.get(symbol, [])
                if time.time() - t <= self.performance_window
            ]
        sharpe = (
            np.mean(returns)
            / (np.std(returns) + 1e-6)
            * np.sqrt(365 * 24 * 60 * 60 / self.performance_window)
            if returns
            else 0.0
        )
        if sharpe < 0:
            base_risk *= 0.5
        elif sharpe > 1:
            base_risk *= 1.5
        threshold = max(self.config.get("volatility_threshold", 0.02), 1e-6)
        vol_coeff = volatility / threshold
        base_risk *= max(0.5, min(2.0, vol_coeff))
        return min(self.max_risk_per_trade, max(self.min_risk_per_trade, base_risk))

    async def get_sharpe_ratio(self, symbol: str) -> float:
        async with self.returns_lock:
            returns = [
                r
                for t, r in self.returns_by_symbol.get(symbol, [])
                if time.time() - t <= self.performance_window
            ]
        if not returns:
            return 0.0
        return (
            np.mean(returns)
            / (np.std(returns) + 1e-6)
            * np.sqrt(365 * 24 * 60 * 60 / self.performance_window)
        )

    async def get_loss_streak(self, symbol: str) -> int:
        async with self.returns_lock:
            returns = [r for _, r in self.returns_by_symbol.get(symbol, [])]
        count = 0
        for r in reversed(returns):
            if r < 0:
                count += 1
            else:
                break
        return count

    async def get_win_streak(self, symbol: str) -> int:
        async with self.returns_lock:
            returns = [r for _, r in self.returns_by_symbol.get(symbol, [])]
        count = 0
        for r in reversed(returns):
            if r > 0:
                count += 1
            else:
                break
        return count

    def save_state(self):
        if not self.positions_changed or (
            time.time() - self.last_save_time < self.save_interval
        ):
            return
        try:
            disk_usage = shutil.disk_usage(self.config["cache_dir"])
            if disk_usage.free / (1024**3) < 0.5:
                logger.warning(
                    "Not enough space to persist state: %.2f GB left",
                    disk_usage.free / (1024 ** 3),
                )
                return
            self.positions.to_pickle(self.state_file)
            with open(self.returns_file, "wb") as f:
                joblib.dump(self.returns_by_symbol, f)
            self.last_save_time = time.time()
            self.positions_changed = False
            logger.info("TradeManager state saved")
        except Exception as e:
            logger.exception("Failed to save state: %s", e)
            raise

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                self.positions = pd.read_pickle(self.state_file)
            if os.path.exists(self.returns_file):
                with open(self.returns_file, "rb") as f:
                    self.returns_by_symbol = joblib.load(f)
                logger.info("TradeManager state loaded")
        except Exception as e:
            logger.exception("Failed to load state: %s", e)
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3)
    )
    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        params: Dict | None = None,
        *,
        use_lock: bool = True,
    ) -> Optional[Dict]:
        params = params or {}

        async def _execute_order() -> Optional[Dict]:
            try:
                order_params = {"category": "linear", **params}
                order_type = order_params.get("type", "market")
                tp_price = order_params.pop("takeProfitPrice", None)
                sl_price = order_params.pop("stopLossPrice", None)
                if (tp_price is not None or sl_price is not None) and hasattr(
                    self.exchange, "create_order_with_take_profit_and_stop_loss"
                ):
                    order = await safe_api_call(
                        self.exchange,
                        "create_order_with_take_profit_and_stop_loss",
                        symbol,
                        order_type,
                        side,
                        size,
                        price if order_type != "market" else None,
                        tp_price,
                        sl_price,
                        order_params,
                    )
                else:
                    order = await safe_api_call(
                        self.exchange,
                        "create_order",
                        symbol,
                        order_type,
                        side,
                        size,
                        price,
                        order_params,
                    )
                logger.info(
                    "Order placed: %s, %s, size=%s, price=%s, type=%s",
                    symbol,
                    side,
                    size,
                    price,
                    order_type,
                )
                await self.telegram_logger.send_telegram_message(
                    f"✅ Order: {symbol} {side.upper()} size={size:.4f} @ {price:.2f} ({order_type})"
                )

                if isinstance(order, dict):
                    ret_code = order.get("retCode") or order.get("ret_code")
                    if ret_code is not None and ret_code != 0:
                        logger.error("Order not confirmed: %s", order)
                        await self.telegram_logger.send_telegram_message(
                            f"❌ Order not confirmed {symbol}: retCode {ret_code}"
                        )
                        return None

                return order
            except Exception as e:
                logger.exception("Failed to place order for %s: %s", symbol, e)
                await self.telegram_logger.send_telegram_message(
                    f"❌ Order error {symbol}: {e}"
                )
                raise

        if use_lock:
            async with self.position_lock:
                return await _execute_order()
        else:
            return await _execute_order()

    async def calculate_position_size(
        self, symbol: str, price: float, atr: float, sl_multiplier: float
    ) -> float:
        try:
            if price <= 0 or atr <= 0:
                logger.warning(
                    "Invalid inputs for %s: price=%s, atr=%s",
                    symbol,
                    price,
                    atr,
                )
                return 0.0
            account = await safe_api_call(self.exchange, "fetch_balance")
            equity = float(account["total"].get("USDT", 0))
            if equity <= 0:
                logger.warning("Insufficient balance for %s", symbol)
                await self.telegram_logger.send_telegram_message(
                    f"⚠️ Insufficient balance for {symbol}: equity={equity}"
                )
                return 0.0
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            volatility = (
                df["close"].pct_change().std()
                if df is not None and not df.empty
                else self.config.get("volatility_threshold", 0.02)
            )
            risk_per_trade = await self.compute_risk_per_trade(symbol, volatility)
            risk_amount = equity * risk_per_trade
            stop_loss_distance = atr * sl_multiplier
            if stop_loss_distance <= 0:
                logger.warning("Invalid stop_loss_distance for %s", symbol)
                return 0.0
            position_size = risk_amount / (stop_loss_distance * self.leverage)
            position_size = min(position_size, equity * self.leverage / price * 0.1)
            logger.info(
                "Position size for %s: %.4f (risk %.2f USDT, ATR %.2f)",
                symbol,
                position_size,
                risk_amount,
                atr,
            )
            return position_size
        except Exception as e:
            logger.exception("Failed to calculate position size for %s: %s", symbol, e)
            raise

    def calculate_stop_loss_take_profit(
        self,
        side: str,
        price: float,
        atr: float,
        sl_multiplier: float,
        tp_multiplier: float,
    ) -> Tuple[float, float]:
        """Return stop-loss and take-profit prices."""
        stop_loss_price = (
            price - sl_multiplier * atr if side == "buy" else price + sl_multiplier * atr
        )
        take_profit_price = (
            price + tp_multiplier * atr if side == "buy" else price - tp_multiplier * atr
        )
        return stop_loss_price, take_profit_price

    async def open_position(self, symbol: str, side: str, price: float, params: Dict):
        try:
            async with self.position_lock:
                if len(self.positions) >= self.max_positions:
                    logger.warning(
                        "Maximum number of positions reached: %s",
                        self.max_positions,
                    )
                    return
                if side not in {"buy", "sell"}:
                    logger.warning("Invalid side %s for %s", side, symbol)
                    return
                if (
                    "symbol" in self.positions.index.names
                    and symbol in self.positions.index.get_level_values("symbol")
                ):
                    logger.warning("Position for %s already open", symbol)
                    return
                if not await self.data_handler.is_data_fresh(symbol):
                    logger.warning("Stale data for %s, skipping trade", symbol)
                    return
                atr = await self.data_handler.get_atr(symbol)
                if atr <= 0:
                    logger.warning(
                        "ATR data missing for %s, retrying later",
                        symbol,
                    )
                    return
                sl_mult = params.get("sl_multiplier", self.config["sl_multiplier"])
                tp_mult = params.get("tp_multiplier", self.config["tp_multiplier"])
                size = await self.calculate_position_size(symbol, price, atr, sl_mult)
                if size <= 0:
                    logger.warning("Position size too small for %s", symbol)
                    return
                stop_loss_price, take_profit_price = self.calculate_stop_loss_take_profit(
                    side, price, atr, sl_mult, tp_mult
                )

                order_params = {
                    "leverage": self.leverage,
                    "stopLossPrice": stop_loss_price,
                    "takeProfitPrice": take_profit_price,
                    "tpslMode": "full",
                }
                order = await self.place_order(
                    symbol, side, size, price, order_params, use_lock=False
                )
                if not order:
                    logger.error(
                        "Order failed for %s: no confirmation returned",
                        symbol,
                    )
                    await self.telegram_logger.send_telegram_message(
                        f"❌ Order failed {symbol}: no confirmation"
                    )
                    return
                if isinstance(order, dict):
                    ret_code = order.get("retCode") or order.get("ret_code")
                    if ret_code is not None and ret_code != 0:
                        logger.error("Order error for %s: %s", symbol, order)
                        await self.telegram_logger.send_telegram_message(
                            f"❌ Order error {symbol}: retCode {ret_code}"
                        )
                        return
                    if not (order.get("id") or order.get("orderId") or order.get("result")):
                        logger.error("Order confirmation missing id for %s: %s", symbol, order)
                        await self.telegram_logger.send_telegram_message(
                            f"❌ Order confirmation missing id {symbol}"
                        )
                        return
                new_position = {
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "entry_price": price,
                    "tp_multiplier": tp_mult,
                    "sl_multiplier": sl_mult,
                    "highest_price": price if side == "buy" else float("inf"),
                    "lowest_price": price if side == "sell" else 0.0,
                    "breakeven_triggered": False,
                }
                new_position_df = pd.DataFrame(
                    [new_position],
                    index=pd.MultiIndex.from_tuples(
                        [(symbol, pd.Timestamp.now())], names=["symbol", "timestamp"]
                    ),
                    dtype=object,
                )
                if (
                    "symbol" in self.positions.index.names
                    and symbol in self.positions.index.get_level_values("symbol")
                ):
                    logger.warning(
                        "Position for %s already open after order placed",
                        symbol,
                    )
                    return
                if self.positions.empty:
                    self.positions = new_position_df
                else:
                    self.positions = pd.concat(
                        [self.positions, new_position_df], ignore_index=False
                    )
                self.positions_changed = True
                self.save_state()
                logger.info(
                    "Position opened: %s, %s, size=%s, entry=%s",
                    symbol,
                    side,
                    size,
                    price,
                )
                await self.telegram_logger.send_telegram_message(
                    f"📈 {symbol} {side.upper()} size={size:.4f} @ {price:.2f} SL={stop_loss_price:.2f} TP={take_profit_price:.2f}",
                    urgent=True,
                )
        except Exception as e:
            logger.exception("Failed to open position for %s: %s", symbol, e)
            await self.telegram_logger.send_telegram_message(
                f"❌ Failed to open position {symbol}: {e}"
            )
            raise

    async def close_position(
        self, symbol: str, exit_price: float, reason: str = "Manual"
    ):
        async with self.position_lock:
            async with self.returns_lock:
                try:
                    if "symbol" in self.positions.index.names:
                        position = self.positions.loc[
                            self.positions.index.get_level_values("symbol") == symbol
                        ]
                    else:
                        position = pd.DataFrame()
                    if position.empty:
                        logger.warning("Position for %s not found", symbol)
                        return
                    position = position.iloc[0]
                    side = "sell" if position["side"] == "buy" else "buy"
                    order = await self.place_order(
                        symbol,
                        side,
                        position["size"],
                        exit_price,
                        use_lock=False,
                    )
                    if order:
                        profit = (
                            (exit_price - position["entry_price"]) * position["size"]
                            if position["side"] == "buy"
                            else (position["entry_price"] - exit_price)
                            * position["size"]
                        )
                        profit *= self.leverage
                        self.returns_by_symbol[symbol].append(
                            (pd.Timestamp.now(tz="UTC").timestamp(), profit)
                        )
                        self.positions = self.positions.drop(symbol, level="symbol")
                        self.positions_changed = True
                        self.save_state()
                        logger.info(
                            "Position closed: %s, profit=%.2f, reason=%s",
                            symbol,
                            profit,
                            reason,
                        )
                        await self.telegram_logger.send_telegram_message(
                            f"📉 {symbol} {position['side'].upper()} exit={exit_price:.2f} PnL={profit:.2f} USDT ({reason})",
                            urgent=True,
                        )
                except Exception as e:
                    logger.exception("Failed to close position for %s: %s", symbol, e)
                    await self.telegram_logger.send_telegram_message(
                        f"❌ Failed to close position {symbol}: {e}"
                    )
                    raise

    async def check_trailing_stop(self, symbol: str, current_price: float):
        async with self.position_lock:
            try:
                if "symbol" in self.positions.index.names:
                    position = self.positions.loc[
                        self.positions.index.get_level_values("symbol") == symbol
                    ]
                else:
                    position = pd.DataFrame()
                if position.empty:
                    logger.warning("Position for %s not found", symbol)
                    return
                position = position.iloc[0]
                atr = await self.data_handler.get_atr(symbol)
                if atr <= 0:
                    logger.warning(
                        "ATR data missing for %s, retrying later",
                        symbol,
                    )
                    return
                trailing_stop_distance = atr * self.config.get(
                    "trailing_stop_multiplier", 1.0
                )

                profit_pct = (
                    (current_price - position["entry_price"])
                    / position["entry_price"]
                    * 100
                    if position["side"] == "buy"
                    else (position["entry_price"] - current_price)
                    / position["entry_price"]
                    * 100
                )
                profit_atr = (
                    current_price - position["entry_price"]
                    if position["side"] == "buy"
                    else position["entry_price"] - current_price
                )

                trigger_pct = self.config.get("trailing_stop_percentage", 1.0)
                trigger_atr = self.config.get("trailing_stop_coeff", 1.0) * atr

                if not position["breakeven_triggered"] and (
                    profit_pct >= trigger_pct or profit_atr >= trigger_atr
                ):
                    close_size = position["size"] * 0.5
                    side = "sell" if position["side"] == "buy" else "buy"
                    await self.place_order(
                        symbol,
                        side,
                        close_size,
                        current_price,
                        use_lock=False,
                    )
                    remaining_size = position["size"] - close_size
                    self.positions.loc[(symbol, slice(None)), "size"] = remaining_size
                    self.positions.loc[(symbol, slice(None)), "sl_multiplier"] = 0.0
                    self.positions.loc[(symbol, slice(None)), "breakeven_triggered"] = (
                        True
                    )
                    self.positions_changed = True
                    self.save_state()
                    await self.telegram_logger.send_telegram_message(
                        f"🏁 {symbol} moved to breakeven, partial profits taken"
                    )

                if position["side"] == "buy":
                    new_highest = max(position["highest_price"], current_price)
                    self.positions.loc[(symbol, slice(None)), "highest_price"] = (
                        new_highest
                    )
                    trailing_stop_price = new_highest - trailing_stop_distance
                    if current_price <= trailing_stop_price:
                        await self.close_position(
                            symbol, current_price, "Trailing Stop"
                        )
                else:
                    new_lowest = min(position["lowest_price"], current_price)
                    self.positions.loc[(symbol, slice(None)), "lowest_price"] = (
                        new_lowest
                    )
                    trailing_stop_price = new_lowest + trailing_stop_distance
                    if current_price >= trailing_stop_price:
                        await self.close_position(
                            symbol, current_price, "Trailing Stop"
                        )
            except Exception as e:
                logger.exception("Failed trailing stop check for %s: %s", symbol, e)
                raise

    async def check_stop_loss_take_profit(self, symbol: str, current_price: float):
        async with self.position_lock:
            try:
                if "symbol" in self.positions.index.names:
                    position = self.positions.loc[
                        self.positions.index.get_level_values("symbol") == symbol
                    ]
                else:
                    position = pd.DataFrame()
                if position.empty:
                    return
                position = position.iloc[0]
                indicators = self.data_handler.indicators.get(symbol)
                if not indicators or not indicators.atr.iloc[-1]:
                    return
                atr = indicators.atr.iloc[-1]
                stop_loss = (
                    position["entry_price"]
                    * (1 - position["sl_multiplier"] * atr / position["entry_price"])
                    if position["side"] == "buy"
                    else position["entry_price"]
                    * (1 + position["sl_multiplier"] * atr / position["entry_price"])
                )
                take_profit = (
                    position["entry_price"]
                    * (1 + position["tp_multiplier"] * atr / position["entry_price"])
                    if position["side"] == "buy"
                    else position["entry_price"]
                    * (1 - position["tp_multiplier"] * atr / position["entry_price"])
                )
                if position["side"] == "buy" and current_price <= stop_loss:
                    await self.close_position(symbol, current_price, "Stop Loss")
                elif position["side"] == "sell" and current_price >= stop_loss:
                    await self.close_position(symbol, current_price, "Stop Loss")
                elif position["side"] == "buy" and current_price >= take_profit:
                    await self.close_position(symbol, current_price, "Take Profit")
                elif position["side"] == "sell" and current_price <= take_profit:
                    await self.close_position(symbol, current_price, "Take Profit")
            except Exception as e:
                logger.exception("Failed SL/TP check for %s: %s", symbol, e)
                raise

    async def check_lstm_exit_signal(self, symbol: str, current_price: float):
        try:
            model = self.model_builder.lstm_models.get(symbol)
            if not model:
                logger.debug("Model for %s not found", symbol)
                return
            if "symbol" in self.positions.index.names:
                position = self.positions.loc[
                    self.positions.index.get_level_values("symbol") == symbol
                ]
            else:
                position = pd.DataFrame()
            if position.empty:
                return
            position = position.iloc[0]
            indicators = self.data_handler.indicators.get(symbol)
            empty = await _check_df_async(
                indicators.df, f"check_lstm_exit_signal {symbol}"
            )
            if not indicators or empty:
                return
            features = await self.model_builder.prepare_lstm_features(
                symbol, indicators
            )
            if len(features) < self.config["lstm_timesteps"]:
                return
            X = np.array([features[-self.config["lstm_timesteps"] :]])
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.model_builder.device
            )
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type):
                prediction = model(X_tensor).squeeze().float().cpu().numpy()
            calibrator = self.model_builder.calibrators.get(symbol)
            if calibrator is not None:
                prediction = calibrator.predict_proba([[prediction]])[0, 1]
            long_threshold, short_threshold = (
                await self.model_builder.adjust_thresholds(symbol, prediction)
            )
            if position["side"] == "buy" and prediction < short_threshold:
                logger.info(
                    "CNN-LSTM exit long signal for %s: pred=%.4f, threshold=%.2f",
                    symbol,
                    prediction,
                    short_threshold,
                )
                await self.close_position(symbol, current_price, "CNN-LSTM Exit Signal")
            elif position["side"] == "sell" and prediction > long_threshold:
                logger.info(
                    "CNN-LSTM exit short signal for %s: pred=%.4f, threshold=%.2f",
                    symbol,
                    prediction,
                    long_threshold,
                )
                await self.close_position(symbol, current_price, "CNN-LSTM Exit Signal")
        except Exception as e:
            logger.exception("Failed to check CNN-LSTM signal for %s: %s", symbol, e)
            raise

    async def monitor_performance(self):
        while True:
            try:
                async with self.returns_lock:
                    current_time = pd.Timestamp.now(tz="UTC").timestamp()
                    for symbol in self.returns_by_symbol:
                        returns = [
                            r
                            for t, r in self.returns_by_symbol[symbol]
                            if current_time - t <= self.performance_window
                        ]
                        self.returns_by_symbol[symbol] = [
                            (t, r)
                            for t, r in self.returns_by_symbol[symbol]
                            if current_time - t <= self.performance_window
                        ]
                        if returns:
                            sharpe_ratio = (
                                np.mean(returns)
                                / (np.std(returns) + 1e-6)
                                * np.sqrt(365 * 24 * 60 * 60 / self.performance_window)
                            )
                            logger.info(
                                "Sharpe Ratio for %s: %.2f",
                                symbol,
                                sharpe_ratio,
                            )
                            ohlcv = self.data_handler.ohlcv
                            if (
                                "symbol" in ohlcv.index.names
                                and symbol in ohlcv.index.get_level_values("symbol")
                            ):
                                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                            else:
                                df = None
                            if df is not None and not df.empty:
                                volatility = df["close"].pct_change().std()
                                volatility_change = abs(
                                    volatility - self.last_volatility.get(symbol, 0.0)
                                ) / max(self.last_volatility.get(symbol, 0.01), 0.01)
                                self.last_volatility[symbol] = volatility
                                if (
                                    sharpe_ratio
                                    < self.config.get("min_sharpe_ratio", 0.5)
                                    or volatility_change > 0.5
                                ):
                                    logger.info(
                                        "Retraining triggered for %s: Sharpe=%.2f, Volatility change=%.2f",
                                        symbol,
                                        sharpe_ratio,
                                        volatility_change,
                                    )
                                    await self.model_builder.retrain_symbol(symbol)
                                    await self.telegram_logger.send_telegram_message(
                                        f"🔄 Retraining {symbol}: Sharpe={sharpe_ratio:.2f}, Volatility={volatility_change:.2f}"
                                    )
                            if sharpe_ratio < self.config.get("min_sharpe_ratio", 0.5):
                                logger.warning(
                                    "Low Sharpe Ratio for %s: %.2f",
                                    symbol,
                                    sharpe_ratio,
                                )
                                await self.telegram_logger.send_telegram_message(
                                    f"⚠️ Low Sharpe Ratio for {symbol}: {sharpe_ratio:.2f}"
                                )
                await asyncio.sleep(self.performance_window / 10)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Performance monitoring error: %s", e)
                await asyncio.sleep(1)
                continue

    async def manage_positions(self):
        while True:
            try:
                symbols = []
                if "symbol" in self.positions.index.names:
                    symbols = self.positions.index.get_level_values("symbol").unique()
                for symbol in symbols:
                    ohlcv = self.data_handler.ohlcv
                    if (
                        "symbol" in ohlcv.index.names
                        and symbol in ohlcv.index.get_level_values("symbol")
                    ):
                        df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                    else:
                        df = None
                    empty = await _check_df_async(df, f"manage_positions {symbol}")
                    if empty:
                        continue
                    current_price = df["close"].iloc[-1]
                    await self.check_trailing_stop(symbol, current_price)
                    await self.check_stop_loss_take_profit(symbol, current_price)
                    await self.check_lstm_exit_signal(symbol, current_price)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Error managing positions: %s", e)
                await asyncio.sleep(1)
                continue

    async def evaluate_ema_condition(self, symbol: str, signal: str) -> bool:
        try:
            ohlcv_2h = self.data_handler.ohlcv_2h
            if (
                "symbol" in ohlcv_2h.index.names
                and symbol in ohlcv_2h.index.get_level_values("symbol")
            ):
                df_2h = ohlcv_2h.xs(symbol, level="symbol", drop_level=False)
            else:
                df_2h = None
            indicators_2h = self.data_handler.indicators_2h.get(symbol)
            empty = await _check_df_async(df_2h, f"evaluate_ema_condition {symbol}")
            if empty or not indicators_2h:
                logger.warning(
                    "No data or indicators for %s on 2h timeframe",
                    symbol,
                )
                return False
            ema30 = indicators_2h.ema30
            ema100 = indicators_2h.ema100
            close = df_2h["close"]
            timestamps = df_2h.index.get_level_values("timestamp")
            lookback_period = pd.Timedelta(
                seconds=self.config["ema_crossover_lookback"]
            )
            recent_data = df_2h[timestamps >= timestamps[-1] - lookback_period]
            if len(recent_data) < 2:
                logger.debug(
                    "Not enough data to check EMA crossover for %s",
                    symbol,
                )
                return False
            ema30_recent = ema30[-len(recent_data) :]
            ema100_recent = ema100[-len(recent_data) :]
            crossover_long = (ema30_recent.iloc[-2] <= ema100_recent.iloc[-2]) and (
                ema30_recent.iloc[-1] > ema100_recent.iloc[-1]
            )
            crossover_short = (ema30_recent.iloc[-2] >= ema100_recent.iloc[-2]) and (
                ema30_recent.iloc[-1] < ema100_recent.iloc[-1]
            )
            if (signal == "buy" and not crossover_long) or (
                signal == "sell" and not crossover_short
            ):
                logger.debug(
                    "EMA crossover not confirmed for %s, signal=%s",
                    symbol,
                    signal,
                )
                return False
            pullback_period = pd.Timedelta(seconds=self.config["pullback_period"])
            pullback_data = df_2h[timestamps >= timestamps[-1] - pullback_period]
            volatility = close.pct_change().std() if not close.empty else 0.02
            pullback_threshold = (
                ema30.iloc[-1] * self.config["pullback_volatility_coeff"] * volatility
            )
            pullback_zone_high = ema30.iloc[-1] + pullback_threshold
            pullback_zone_low = ema30.iloc[-1] - pullback_threshold
            pullback_occurred = False
            for i in range(len(pullback_data)):
                price = pullback_data["close"].iloc[i]
                if pullback_zone_low <= price <= pullback_zone_high:
                    pullback_occurred = True
                    break
            if not pullback_occurred:
                logger.debug(
                    "No pullback to EMA30 for %s, signal=%s",
                    symbol,
                    signal,
                )
                return False
            current_price = close.iloc[-1]
            if (signal == "buy" and current_price <= ema30.iloc[-1]) or (
                signal == "sell" and current_price >= ema30.iloc[-1]
            ):
                logger.debug("Price not consolidated for %s, signal=%s", symbol, signal)
                return False
            logger.info("EMA conditions satisfied for %s, signal=%s", symbol, signal)
            return True
        except Exception as e:
            logger.exception("Failed to check EMA conditions for %s: %s", symbol, e)
            raise

    async def evaluate_signal(self, symbol: str):
        try:
            model = self.model_builder.lstm_models.get(symbol)
            if not model:
                logger.debug("Model for %s not yet trained", symbol)
                return None
            indicators = self.data_handler.indicators.get(symbol)
            empty = (
                await _check_df_async(indicators.df, f"evaluate_signal {symbol}")
                if indicators
                else True
            )
            if not indicators or empty:
                return None
            ohlcv = self.data_handler.ohlcv
            if (
                "symbol" in ohlcv.index.names
                and symbol in ohlcv.index.get_level_values("symbol")
            ):
                df = ohlcv.xs(symbol, level="symbol", drop_level=False)
            else:
                df = None
            if not await self.data_handler.is_data_fresh(symbol):
                logger.debug("Stale data for %s, skipping signal", symbol)
                return None
            if df is not None and not df.empty:
                volatility = df["close"].pct_change().std()
            else:
                volatility = self.config.get("volatility_threshold", 0.02)
            loss_streak = await self.get_loss_streak(symbol)
            if (
                indicators.adx.iloc[-1] < 20
                and volatility > self.config.get("volatility_threshold", 0.02)
                and loss_streak >= 2
            ):
                logger.info(
                    "Skipping signal for %s: weak trend and loss streak",
                    symbol,
                )
                return None
            features = await self.model_builder.prepare_lstm_features(
                symbol, indicators
            )
            if len(features) < self.config["lstm_timesteps"]:
                return None
            X = np.array([features[-self.config["lstm_timesteps"] :]])
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.model_builder.device
            )
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type):
                prediction = model(X_tensor).squeeze().float().cpu().numpy()
            calibrator = self.model_builder.calibrators.get(symbol)
            if calibrator is not None:
                prediction = calibrator.predict_proba([[prediction]])[0, 1]
            long_threshold, short_threshold = (
                await self.model_builder.adjust_thresholds(symbol, prediction)
            )
            signal = None
            if prediction > long_threshold:
                signal = "buy"
            elif prediction < short_threshold:
                signal = "sell"

            rl_signal = None
            if self.rl_agent and symbol in self.rl_agent.models:
                rl_feat = features[-1]
                rl_signal = self.rl_agent.predict(symbol, rl_feat)
                if rl_signal:
                    logger.info("RL signal for %s: %s", symbol, rl_signal)

            if signal:
                logger.info(
                    "CNN-LSTM signal for %s: %s (pred %.4f, thresholds %.2f/%.2f)",
                    symbol,
                    signal,
                    prediction,
                    long_threshold,
                    short_threshold,
                )
                ema_condition_met = await self.evaluate_ema_condition(symbol, signal)
                if not ema_condition_met:
                    logger.info(
                        "EMA conditions not met for %s, signal rejected",
                        symbol,
                    )
                    return None
                logger.info(
                    "All conditions met for %s, confirmed signal: %s",
                    symbol,
                    signal,
                )

            if signal and rl_signal:
                if signal == rl_signal:
                    return signal
                logger.info(
                    "Signal mismatch for %s: CNN-LSTM %s, RL %s",
                    symbol,
                    signal,
                    rl_signal,
                )
                return None
            if signal:
                return signal
            if rl_signal:
                ema_condition_met = await self.evaluate_ema_condition(symbol, rl_signal)
                if ema_condition_met:
                    return rl_signal
            return None
        except Exception as e:
            logger.exception("Failed to evaluate signal for %s: %s", symbol, e)
            raise

    async def run(self):
        try:
            self.loop = asyncio.get_running_loop()
            self.tasks = [
                asyncio.create_task(
                    self.monitor_performance(), name="monitor_performance"
                ),
                asyncio.create_task(self.manage_positions(), name="manage_positions"),
            ]
            for symbol in self.data_handler.usdt_pairs:
                task_name = f"process_symbol_{symbol}"
                self.tasks.append(
                    asyncio.create_task(self.process_symbol(symbol), name=task_name)
                )
            results = await asyncio.gather(*self.tasks, return_exceptions=True)
            for task, result in zip(self.tasks, results):
                if isinstance(result, Exception):
                    logger.error("Task %s failed: %s", task.get_name(), result)
                    await self.telegram_logger.send_telegram_message(
                        f"❌ Task {task.get_name()} failed: {result}"
                    )
        except Exception as e:
            logger.exception("Critical error in TradeManager: %s", e)
            await self.telegram_logger.send_telegram_message(
                f"❌ Critical TradeManager error: {e}"
            )
            raise
        finally:
            self.tasks.clear()

    async def stop(self) -> None:
        """Cancel running tasks and shut down Telegram logging."""
        for task in list(self.tasks):
            task.cancel()
        for task in list(self.tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.tasks.clear()
        await TelegramLogger.shutdown()

    def shutdown(self) -> None:
        """Synchronous wrapper for graceful shutdown."""
        if self.loop and self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.stop(), self.loop)
            try:
                fut.result()
            except Exception:
                pass
        else:
            try:
                asyncio.run(self.stop())
            except RuntimeError:
                # event loop already closed
                pass

    async def process_symbol(self, symbol: str):
        while symbol not in self.model_builder.lstm_models:
            logger.debug("Waiting for model for %s", symbol)
            await asyncio.sleep(30)
        while True:
            try:
                signal = await self.evaluate_signal(symbol)
                condition = True
                if "symbol" in self.positions.index.names:
                    condition = symbol not in self.positions.index.get_level_values(
                        "symbol"
                    )
                if signal and condition:
                    ohlcv = self.data_handler.ohlcv
                    if (
                        "symbol" in ohlcv.index.names
                        and symbol in ohlcv.index.get_level_values("symbol")
                    ):
                        df = ohlcv.xs(symbol, level="symbol", drop_level=False)
                    else:
                        df = None
                    empty = await _check_df_async(df, f"process_symbol {symbol}")
                    if empty:
                        continue
                    current_price = df["close"].iloc[-1]
                    params = await self.data_handler.parameter_optimizer.optimize(
                        symbol
                    )
                    await self.open_position(symbol, signal, current_price, params)
                await asyncio.sleep(
                    self.config["check_interval"] / len(self.data_handler.usdt_pairs)
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Error processing %s: %s", symbol, e)
                await asyncio.sleep(1)
                continue


# ----------------------------------------------------------------------
# REST API for minimal integration testing
# ----------------------------------------------------------------------

api_app = Flask(__name__)

# Expose an ASGI-compatible application so Gunicorn's UvicornWorker can run
# this Flask app without raising "Flask.__call__() missing start_response".
try:  # Flask 2.2+ provides ``asgi_app`` for native ASGI support
    asgi_app = api_app.asgi_app
except AttributeError:  # pragma: no cover - older Flask versions
    try:
        from a2wsgi import WSGIMiddleware  # type: ignore
    except Exception:  # pragma: no cover - fallback if a2wsgi isn't installed
        from uvicorn.middleware.wsgi import WSGIMiddleware

    asgi_app = WSGIMiddleware(api_app)

# Track when the TradeManager initialization finishes
_ready_event = threading.Event()

# For simple logging/testing of received orders
POSITIONS = []

trade_manager: TradeManager | None = None


async def create_trade_manager() -> TradeManager:
    """Instantiate the TradeManager using config.json."""
    global trade_manager
    if trade_manager is None:
        logger.info("Loading configuration from config.json")
        try:
            cfg = load_config("config.json")
            logger.info("Configuration loaded successfully")
        except Exception as exc:
            logger.exception("Failed to load configuration: %s", exc)
            raise
        if not ray.is_initialized():
            logger.info("Initializing Ray with num_cpus=%s", cfg["ray_num_cpus"])
            try:
                ray.init(num_cpus=cfg["ray_num_cpus"], ignore_reinit_error=True)
                logger.info("Ray initialized successfully")
            except Exception as exc:
                logger.exception("Ray initialization failed: %s", exc)
                raise
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        telegram_bot = None
        if token:
            try:
                from telegram import Bot

                telegram_bot = Bot(token)
            except Exception as exc:  # pragma: no cover - import/runtime errors
                logger.exception("Failed to create Telegram Bot: %s", exc)
                raise
        from data_handler import DataHandler
        from model_builder import ModelBuilder

        logger.info("Creating DataHandler")
        try:
            dh = DataHandler(cfg, telegram_bot, chat_id)
            logger.info("DataHandler created successfully")
            await dh.load_initial()
            asyncio.create_task(dh.subscribe_to_klines(dh.usdt_pairs))
        except Exception as exc:
            logger.exception("Failed to create DataHandler: %s", exc)
            raise

        logger.info("Creating ModelBuilder")
        try:
            mb = ModelBuilder(cfg, dh, None)
            logger.info("ModelBuilder created successfully")
            asyncio.create_task(mb.train())
        except Exception as exc:
            logger.exception("Failed to create ModelBuilder: %s", exc)
            raise

        trade_manager = TradeManager(cfg, dh, mb, telegram_bot, chat_id)
        logger.info("TradeManager instance created")
        if telegram_bot:
            from utils import TelegramUpdateListener

            listener = TelegramUpdateListener(telegram_bot)

            async def _handle(upd):
                msg = getattr(upd, "message", None)
                if msg and msg.text and msg.text.lower().startswith("/status"):
                    await telegram_bot.send_message(chat_id=msg.chat_id, text="Bot is running")

            threading.Thread(
                target=lambda: asyncio.run(listener.listen(_handle)), daemon=True
            ).start()
        if os.getenv("TEST_MODE") != "1":
            _register_cleanup_handlers(trade_manager)
    return trade_manager

def _initialize_trade_manager() -> None:
    """Background initialization for the TradeManager."""
    global trade_manager
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trade_manager = loop.run_until_complete(create_trade_manager())
        if trade_manager is not None:
            loop.create_task(trade_manager.run())
            _ready_event.set()
            loop.run_forever()
        else:
            _ready_event.set()
    except Exception:
        _ready_event.set()
        raise


# Start initialization when the module is imported
load_dotenv()
if os.getenv("TEST_MODE") == "1":
    _ready_event.set()
else:
    threading.Thread(target=_initialize_trade_manager, daemon=True).start()





@api_app.route("/open_position", methods=["POST"])
async def open_position_route():
    """Open a new trade position."""
    if not _ready_event.is_set() or trade_manager is None:
        return jsonify({"error": "not ready"}), 503
    info = request.get_json(force=True)
    POSITIONS.append(info)
    symbol = info.get("symbol")
    side = info.get("side")
    price = float(info.get("price", 0))
    asyncio.create_task(trade_manager.open_position(symbol, side, price, info))
    return jsonify({"status": "ok"})


@api_app.route("/positions")
def positions_route():
    return jsonify({"positions": POSITIONS})


@api_app.route("/start")
async def start_route():
    if not _ready_event.is_set() or trade_manager is None:
        return jsonify({"error": "not ready"}), 503
    asyncio.create_task(trade_manager.run())
    return jsonify({"status": "started"})


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@api_app.route("/ready")
def ready() -> tuple:
    """Return 200 once the TradeManager is initialized."""
    if _ready_event.is_set() and trade_manager is not None:
        return jsonify({"status": "ok"})
    return jsonify({"status": "initializing"}), 503


if __name__ == "__main__":
    load_dotenv()
    port = int(os.environ.get("PORT", "8002"))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info("Starting TradeManager service on %s:%s", host, port)
    api_app.run(host=host, port=port)
