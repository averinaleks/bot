"""Вспомогательные функции для расчёта и размещения ордеров."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable


@dataclass(slots=True)
class ProtectiveOrderPlan:
    """Набор защитных уровней для позиции."""

    opposite_side: str
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    trailing_stop_price: float | None = None

    def as_order_params(self, leverage: float, *, tpsl_mode: str = "full") -> dict[str, Any]:
        """Сформировать параметры основного ордера Bybit/ccxt."""

        params: dict[str, Any] = {"leverage": leverage}
        if self.stop_loss_price is not None:
            params["stopLossPrice"] = self.stop_loss_price
        if self.take_profit_price is not None:
            params["takeProfitPrice"] = self.take_profit_price
        if self.stop_loss_price is not None or self.take_profit_price is not None:
            params["tpslMode"] = tpsl_mode
        return params


def calculate_position_size(
    *,
    equity: float,
    risk_per_trade: float,
    atr: float,
    sl_multiplier: float,
    leverage: float,
    price: float,
    max_position_pct: float,
) -> float:
    """Рассчитать размер позиции исходя из риска и ATR."""

    if any(not math.isfinite(value) or value <= 0 for value in (equity, atr, price, leverage)):
        return 0.0
    if not math.isfinite(risk_per_trade) or risk_per_trade <= 0:
        return 0.0
    stop_loss_distance = atr * sl_multiplier
    if not math.isfinite(stop_loss_distance) or stop_loss_distance <= 0:
        return 0.0
    risk_amount = equity * risk_per_trade
    if risk_amount <= 0:
        return 0.0
    position_size = risk_amount / (stop_loss_distance * leverage)
    cap = equity * leverage / price * max(0.0, max_position_pct)
    return max(0.0, min(position_size, cap))


def calculate_stop_loss_take_profit(
    side: str,
    price: float,
    atr: float,
    sl_multiplier: float,
    tp_multiplier: float,
) -> tuple[float, float]:
    """Вычислить цены стоп-лосса и тейк-профита."""

    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported side: {side}")
    stop_loss_price = (
        price - sl_multiplier * atr if side == "buy" else price + sl_multiplier * atr
    )
    take_profit_price = (
        price + tp_multiplier * atr if side == "buy" else price - tp_multiplier * atr
    )
    return stop_loss_price, take_profit_price


def build_protective_order_plan(
    side: str,
    *,
    entry_price: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    trailing_offset: float | None = None,
) -> ProtectiveOrderPlan:
    """Построить план защитных ордеров для позиции."""

    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported side: {side}")
    opposite_side = "sell" if side == "buy" else "buy"
    trailing_price: float | None = None
    if (
        trailing_offset is not None
        and entry_price is not None
        and math.isfinite(trailing_offset)
        and math.isfinite(entry_price)
        and trailing_offset > 0
    ):
        trailing_price = (
            entry_price - trailing_offset if side == "buy" else entry_price + trailing_offset
        )
    return ProtectiveOrderPlan(
        opposite_side=opposite_side,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        trailing_stop_price=trailing_price,
    )


def order_needs_retry(order: Any) -> bool:
    """Проверить, требуется ли повторное размещение ордера."""

    if not order:
        return True
    if isinstance(order, dict):
        ret_code = order.get("retCode") or order.get("ret_code")
        if ret_code not in (None, 0):
            return True
        if not (order.get("id") or order.get("orderId") or order.get("result")):
            return True
    return False


async def _maybe_await(result: Awaitable[Any] | Any) -> Any:
    return await result if inspect.isawaitable(result) else result


async def _maybe_sleep(
    sleep: Callable[[float], Awaitable[None] | None],
    delay: float,
) -> None:
    if delay <= 0:
        return
    result = sleep(delay)
    if inspect.isawaitable(result):
        await result


async def execute_with_retries(
    call: Callable[[], Awaitable[Any] | Any],
    *,
    attempts: int,
    delay: float | Callable[[int], float],
    sleep: Callable[[float], Awaitable[None] | None],
    logger: logging.Logger,
    description: str,
    exceptions: Iterable[type[BaseException]] = (),
    should_retry: Callable[[Any], bool] | None = None,
    on_exception: Callable[[int, BaseException], None] | None = None,
    on_failed_result: Callable[[int, Any], None] | None = None,
) -> Any:
    """Выполнить вызов с повторами и журналированием."""

    total_attempts = max(1, int(attempts))
    caught_exceptions: tuple[type[BaseException], ...]
    if exceptions:
        caught_exceptions = tuple(exceptions)
    else:
        caught_exceptions = tuple()
    last_result: Any = None
    for attempt in range(total_attempts):
        if attempt > 0:
            if callable(delay):
                wait_for = float(delay(attempt - 1))
            else:
                wait_for = float(delay)
            await _maybe_sleep(sleep, max(0.0, wait_for))
        try:
            result = await _maybe_await(call())
        except caught_exceptions as exc:
            if on_exception is not None:
                on_exception(attempt, exc)
            last_result = None
            continue
        except BaseException:
            raise
        last_result = result
        if should_retry is not None and should_retry(result):
            if on_failed_result is not None:
                on_failed_result(attempt, result)
            continue
        return result
    if total_attempts > 1 and last_result is None:
        logger.error("%s failed after %s attempts", description, total_attempts)
    return last_result


def execute_with_retries_sync(
    call: Callable[[], Any],
    *,
    attempts: int,
    delay: float | Callable[[int], float],
    sleep: Callable[[float], None] = time.sleep,
    logger: logging.Logger,
    description: str,
    exceptions: Iterable[type[BaseException]] = (),
    should_retry: Callable[[Any], bool] | None = None,
    on_exception: Callable[[int, BaseException], None] | None = None,
    on_failed_result: Callable[[int, Any], None] | None = None,
) -> Any:
    """Синхронная обёртка над :func:`execute_with_retries`."""

    async def _runner() -> Any:
        return await execute_with_retries(
            call,
            attempts=attempts,
            delay=delay,
            sleep=sleep,  # type: ignore[arg-type]
            logger=logger,
            description=description,
            exceptions=exceptions,
            should_retry=should_retry,
            on_exception=on_exception,
            on_failed_result=on_failed_result,
        )

    return asyncio.run(_runner())

