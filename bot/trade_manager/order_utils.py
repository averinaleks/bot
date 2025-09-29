"""Utility helpers for order sizing, stop calculations and retry handling."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from typing import Any, Awaitable, Callable, Iterable, Sequence


def calculate_position_size(
    *,
    risk_amount: float,
    price: float,
    stop_loss_distance: float | None,
    leverage: float,
    max_position_value: float | None = None,
) -> float:
    """Return position size based on risk and stop loss distance."""
    if not math.isfinite(risk_amount) or risk_amount <= 0:
        return 0.0
    if not math.isfinite(price) or price <= 0:
        return 0.0

    size = 0.0
    if stop_loss_distance is not None and stop_loss_distance > 0:
        size = risk_amount / (stop_loss_distance * leverage)
    else:
        size = risk_amount / price

    if max_position_value is not None and max_position_value > 0:
        max_size = max_position_value / price
        size = min(size, max_size)

    return max(size, 0.0)


def calculate_stop_prices(
    side: str,
    price: float,
    atr: float,
    sl_multiplier: float,
    tp_multiplier: float,
) -> tuple[float, float]:
    """Return stop loss and take profit prices based on multipliers."""
    stop_loss_price = (
        price - sl_multiplier * atr if side == "buy" else price + sl_multiplier * atr
    )
    take_profit_price = (
        price + tp_multiplier * atr if side == "buy" else price - tp_multiplier * atr
    )
    return stop_loss_price, take_profit_price


def _log(loggers: Iterable[logging.Logger], level: str, message: str) -> None:
    for logger in loggers:
        getattr(logger, level)(message)


def _ensure_logger_sequence(loggers: Iterable[logging.Logger] | None) -> Sequence[logging.Logger]:
    if not loggers:
        return (logging.getLogger(__name__),)
    return tuple(loggers)


def _default_success_checker(result: Any) -> bool:
    return bool(result)


async def retry_async(
    operation: Callable[[], Awaitable[Any] | Any],
    *,
    attempts: int,
    delay: float,
    label: str,
    loggers: Iterable[logging.Logger] | None = None,
    success_checker: Callable[[Any], bool] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    backoff_factor: float = 1.0,
) -> Any | None:
    """Execute an async operation with retry logic."""
    logger_sequence = _ensure_logger_sequence(loggers)
    checker = success_checker or _default_success_checker
    current_delay = delay

    for attempt in range(attempts):
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # pragma: no cover - network failures in prod
            _log(
                logger_sequence,
                "error",
                f"{label} attempt {attempt + 1} failed ({type(exc).__name__}): {exc}",
            )
            result = None
        else:
            if checker(result):
                return result
            _log(
                logger_sequence,
                "warning",
                f"{label} attempt {attempt + 1} unsuccessful: {result}",
            )
        if attempt < attempts - 1:
            _log(
                logger_sequence,
                "info",
                f"Retrying {label} (attempt {attempt + 2}/{attempts})",
            )
            if current_delay > 0:
                await sleep(current_delay)
            current_delay *= backoff_factor
    return None


def retry_sync(
    operation: Callable[[], Any],
    *,
    attempts: int,
    delay: float,
    label: str,
    loggers: Iterable[logging.Logger] | None = None,
    success_checker: Callable[[Any], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    backoff_factor: float = 1.0,
) -> Any | None:
    """Execute a synchronous operation with retry logic."""
    logger_sequence = _ensure_logger_sequence(loggers)
    checker = success_checker or _default_success_checker
    current_delay = delay

    for attempt in range(attempts):
        try:
            result = operation()
        except Exception as exc:  # pragma: no cover - network failures in prod
            _log(
                logger_sequence,
                "error",
                f"{label} attempt {attempt + 1} failed ({type(exc).__name__}): {exc}",
            )
            result = None
        else:
            if checker(result):
                return result
            _log(
                logger_sequence,
                "warning",
                f"{label} attempt {attempt + 1} unsuccessful: {result}",
            )
        if attempt < attempts - 1:
            _log(
                logger_sequence,
                "info",
                f"Retrying {label} (attempt {attempt + 2}/{attempts})",
            )
            if current_delay > 0:
                sleep(current_delay)
            current_delay *= backoff_factor
    return None


def order_has_identifier(order: Any) -> bool:
    """Return True if order has an identifier."""
    if not isinstance(order, dict):
        return False
    return (
        order.get("id") is not None
        or order.get("orderId") is not None
        or bool(order.get("result"))
    )


def order_retcode_success(order: Any) -> bool:
    """Return True if exchange response indicates success."""
    if not isinstance(order, dict):
        return False
    ret_code = order.get("retCode") or order.get("ret_code")
    return ret_code in (None, 0)


def place_protective_orders(
    exchange: Any,
    symbol: str,
    side: str,
    amount: float,
    *,
    price: float,
    stop_loss: float | None,
    take_profit: float | None,
    trailing_stop: float | None,
    attempts: int,
    delay: float,
    loggers: Iterable[logging.Logger] | None,
    ccxt_error: type[Exception],
    sanitize: Callable[[str], str],
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, str]]:
    """Place protective orders (SL/TP/trailing) with retries."""
    logger_sequence = _ensure_logger_sequence(loggers)
    protective_failures: list[dict[str, str]] = []
    opposite_side = "sell" if side == "buy" else "buy"
    safe_symbol = sanitize(symbol)
    base_label = f"protective order for {safe_symbol}"

    if stop_loss is not None:
        def _create_stop_loss() -> Any | None:
            try:
                return exchange.create_order(symbol, "stop", opposite_side, amount, stop_loss)
            except ccxt_error as exc:
                _log(logger_sequence, "debug", f"stop order failed: {exc}")
                try:
                    return exchange.create_order(
                        symbol, "stop_market", opposite_side, amount, stop_loss
                    )
                except ccxt_error as second_exc:
                    _log(
                        logger_sequence,
                        "debug",
                        f"stop_market order failed: {second_exc}",
                    )
                    return None

        stop_order = retry_sync(
            _create_stop_loss,
            attempts=attempts,
            delay=delay,
            label=f"stop loss {base_label}",
            loggers=logger_sequence,
            success_checker=order_has_identifier,
            sleep=sleep,
            backoff_factor=2.0,
        )
        if not order_has_identifier(stop_order):
            warn_msg = f"не удалось создать stop loss ордер для {safe_symbol}"
            _log(logger_sequence, "warning", warn_msg)
            protective_failures.append({"type": "stop_loss", "message": warn_msg})

    if take_profit is not None:
        def _create_take_profit() -> Any | None:
            try:
                return exchange.create_order(
                    symbol, "limit", opposite_side, amount, take_profit
                )
            except ccxt_error as exc:
                _log(logger_sequence, "debug", f"take profit order failed: {exc}")
                return None

        tp_order = retry_sync(
            _create_take_profit,
            attempts=attempts,
            delay=delay,
            label=f"take profit {base_label}",
            loggers=logger_sequence,
            success_checker=order_has_identifier,
            sleep=sleep,
            backoff_factor=2.0,
        )
        if not order_has_identifier(tp_order):
            warn_msg = f"не удалось создать take profit ордер для {safe_symbol}"
            _log(logger_sequence, "warning", warn_msg)
            protective_failures.append({"type": "take_profit", "message": warn_msg})

    if trailing_stop is not None and price > 0:
        stop_price, _ = calculate_stop_prices(
            side, price, trailing_stop, 1.0, 0.0
        )

        def _create_trailing() -> Any | None:
            try:
                return exchange.create_order(
                    symbol, "stop", opposite_side, amount, stop_price
                )
            except ccxt_error as exc:
                _log(logger_sequence, "debug", f"trailing stop order failed: {exc}")
                try:
                    return exchange.create_order(
                        symbol, "stop_market", opposite_side, amount, stop_price
                    )
                except ccxt_error as second_exc:
                    _log(
                        logger_sequence,
                        "debug",
                        f"trailing stop_market failed: {second_exc}",
                    )
                    return None

        trailing_order = retry_sync(
            _create_trailing,
            attempts=attempts,
            delay=delay,
            label=f"trailing stop {base_label}",
            loggers=logger_sequence,
            success_checker=order_has_identifier,
            sleep=sleep,
            backoff_factor=2.0,
        )
        if not order_has_identifier(trailing_order):
            warn_msg = f"не удалось создать trailing stop ордер для {safe_symbol}"
            _log(logger_sequence, "warning", warn_msg)
            protective_failures.append({"type": "trailing_stop", "message": warn_msg})

    return protective_failures
