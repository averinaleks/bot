"""Вспомогательные функции для расчёта ордеров и защитных заявок."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple, TypeVar

_T = TypeVar("_T")


def calculate_position_size(
    *,
    risk_amount: float,
    stop_loss_distance: float,
    leverage: float,
    price: float,
    equity: float,
    max_position_pct: float,
) -> float:
    """Рассчитать объём позиции с учётом риска и плеча."""
    if price <= 0 or stop_loss_distance <= 0 or leverage <= 0 or equity <= 0:
        return 0.0
    size = risk_amount / (stop_loss_distance * leverage)
    max_size = equity * leverage / price * max_position_pct
    return max(0.0, min(size, max_size))


def amount_from_risk(risk_amount: float, price: float) -> float:
    """Перевести денежный риск в количество лотов."""
    if risk_amount <= 0 or price <= 0:
        return 0.0
    return risk_amount / price


def calculate_stop_loss_take_profit(
    side: str,
    price: float,
    atr: float,
    sl_multiplier: float,
    tp_multiplier: float,
) -> Tuple[float, float]:
    """Посчитать цены стоп-лосса и тейк-профита."""
    direction = 1 if side.lower() == "buy" else -1
    stop_loss_price = price - direction * sl_multiplier * atr
    take_profit_price = price + direction * tp_multiplier * atr
    return stop_loss_price, take_profit_price


def calculate_trailing_stop_price(side: str, entry_price: float, trailing_distance: float) -> float:
    """Рассчитать цену для трейлинг-стопа относительно цены входа."""
    direction = 1 if side.lower() == "buy" else -1
    return entry_price - direction * trailing_distance


def order_has_id(order: Any) -> bool:
    """Проверить, что ответ биржи содержит идентификатор ордера."""
    return bool(getattr(order, "get", lambda _k, _d=None: None)("id"))


def is_successful_exchange_response(order: Any) -> bool:
    """Определить, считается ли ответ от биржи успешным."""
    if not order:
        return False
    if isinstance(order, dict):
        if order.get("id") or order.get("orderId") or order.get("result"):
            return True
        ret_code = order.get("retCode") or order.get("ret_code")
        if ret_code is not None:
            return ret_code == 0
    return False


async def retry_async(
    operation: Callable[[], Awaitable[_T]],
    *,
    attempts: int,
    delay: float,
    success: Callable[[Optional[_T]], bool],
    logger: Optional[Any] = None,
    log_context: str = "operation",
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Optional[_T]:
    """Выполнить асинхронную операцию с повторами."""
    result: Optional[_T] = None
    for attempt in range(attempts):
        if attempt > 0 and logger:
            logger.info(
                "Retrying %s (attempt %s/%s)",
                log_context,
                attempt + 1,
                attempts,
            )
            await asyncio.sleep(delay)
        try:
            result = await operation()
        except exceptions as exc:  # type: ignore[arg-type]
            if logger:
                logger.error(
                    "Attempt %s for %s failed with %s: %s",
                    attempt + 1,
                    log_context,
                    type(exc).__name__,
                    exc,
                )
            result = None
        if success(result):
            return result
        if logger:
            logger.warning(
                "Attempt %s for %s returned unsuccessful response: %s",
                attempt + 1,
                log_context,
                result,
            )
    return result


def retry_sync(
    operation: Callable[[], _T],
    *,
    attempts: int,
    delay: float,
    success: Callable[[Optional[_T]], bool],
    sleep: Callable[[float], None] = time.sleep,
    logger: Optional[Any] = None,
    log_context: str = "operation",
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Optional[_T]:
    """Выполнить синхронную операцию с повторами."""
    result: Optional[_T] = None
    current_delay = delay
    for attempt in range(attempts):
        if attempt > 0:
            if logger:
                logger.info(
                    "Retrying %s (attempt %s/%s)",
                    log_context,
                    attempt + 1,
                    attempts,
                )
            sleep(current_delay)
        try:
            result = operation()
        except exceptions as exc:  # type: ignore[arg-type]
            if logger:
                logger.debug(
                    "Attempt %s for %s raised %s: %s",
                    attempt + 1,
                    log_context,
                    type(exc).__name__,
                    exc,
                )
            result = None
        if success(result):
            return result
        if logger:
            logger.warning(
                "Attempt %s for %s returned unsuccessful response: %s",
                attempt + 1,
                log_context,
                result,
            )
        current_delay *= 2
    return result


def place_protective_order_with_fallback(
    order_factories: Sequence[Callable[[], Optional[dict]]],
    *,
    attempts: int,
    initial_delay: float,
    success: Callable[[Optional[dict]], bool] = order_has_id,
    sleep: Callable[[float], None] = time.sleep,
    logger: Optional[Any] = None,
    log_context: str = "protective order",
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Optional[dict]:
    """Разместить защитный ордер, перебирая варианты и повторы."""
    def _operation() -> Optional[dict]:
        for factory in order_factories:
            try:
                order = factory()
            except exceptions as exc:  # type: ignore[arg-type]
                if logger:
                    logger.debug(
                        "Variant for %s raised %s: %s",
                        log_context,
                        type(exc).__name__,
                        exc,
                    )
                continue
            if success(order):
                return order
        return None

    return retry_sync(
        _operation,
        attempts=attempts,
        delay=initial_delay,
        success=success,
        sleep=sleep,
        logger=logger,
        log_context=log_context,
        exceptions=exceptions,
    )
