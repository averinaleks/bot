import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from bot.gpt_client import GPTClientError, GPTClientJSONError, query_gpt_json_async
from bot.pydantic_compat import BaseModel, ConfigDict, ValidationError
from services.logging_utils import sanitize_log_value
from telegram_utils import send_telegram_alert
from utils import safe_int

logger = logging.getLogger(__name__)

_OFFLINE_ENV = os.getenv("OFFLINE_MODE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _build_gpt_advice_model_class() -> type[BaseModel]:
    """Return the GPT advice model class with offline stub metadata."""

    offline_stub = bool(_OFFLINE_ENV or getattr(BaseModel, "__offline_stub__", False))

    class _GPTAdviceModel(BaseModel):
        """Model for parsing GPT advice responses."""

        __offline_stub__ = offline_stub

        signal: float | str | None = None
        tp_mult: float | None = None
        sl_mult: float | None = None
        model_config = ConfigDict(validate_assignment=False)

        def __setattr__(self, name, value):  # pragma: no cover - simple assignment logic
            super().__setattr__(name, value)
            if name == "signal" and value is None:
                super().__setattr__("tp_mult", None)
                super().__setattr__("sl_mult", None)

        def predict(self, *args, **kwargs):  # pragma: no cover - offline helper only
            """Return a neutral advice value in offline environments."""

            if not self.__offline_stub__:
                raise RuntimeError("predict is only available in offline stub mode")
            return 0.5

    return _GPTAdviceModel


GPTAdviceModel = _build_gpt_advice_model_class()
GPT_ADVICE = GPTAdviceModel()
GPT_ADVICE_MAX_ATTEMPTS = safe_int(
    os.getenv("GPT_ADVICE_MAX_ATTEMPTS"),
    3,
    env_var="GPT_ADVICE_MAX_ATTEMPTS",
)
_GPT_ADVICE_ERROR_COUNT = 0
_GPT_SAFE_MODE = False


async def _enter_gpt_safe_mode(
    set_trading_enabled: Callable[[bool], Awaitable[None]],
    reason: str | None = None,
) -> None:
    """Disable trading and freeze GPT advice after repeated failures."""

    global _GPT_SAFE_MODE, GPT_ADVICE
    if _GPT_SAFE_MODE:
        return

    _GPT_SAFE_MODE = True
    GPT_ADVICE = GPTAdviceModel(signal="hold")
    message = "GPT advice unavailable; entering safe mode"
    if reason:
        message = f"{message}: {reason}"

    logger.error(message)
    try:
        await set_trading_enabled(False)
    except Exception as exc:  # pragma: no cover - unexpected failure to toggle trading
        logger.exception("Failed to disable trading during safe mode: %s", exc)
    await send_telegram_alert(message)


async def _record_gpt_failure(
    set_trading_enabled: Callable[[bool], Awaitable[None]],
    reason: str | None = None,
) -> None:
    """Increment failure counter and enter safe mode when the limit is reached."""

    global _GPT_ADVICE_ERROR_COUNT
    _GPT_ADVICE_ERROR_COUNT += 1
    if reason:
        logger.warning(
            "GPT advice refresh failure (%s/%s): %s",
            _GPT_ADVICE_ERROR_COUNT,
            GPT_ADVICE_MAX_ATTEMPTS,
            reason,
        )
    else:
        logger.warning(
            "GPT advice refresh failure (%s/%s)",
            _GPT_ADVICE_ERROR_COUNT,
            GPT_ADVICE_MAX_ATTEMPTS,
        )
    if _GPT_ADVICE_ERROR_COUNT >= GPT_ADVICE_MAX_ATTEMPTS:
        await _enter_gpt_safe_mode(set_trading_enabled, reason)


async def refresh_gpt_advice(
    symbols: list[str],
    price_history: dict[str, Any],
    build_feature_vector: Callable[[str, float], Awaitable[list[float]]],
    compute_ema: Callable[[list[float]], float],
    set_trading_enabled: Callable[[bool], Awaitable[None]],
) -> None:
    """Fetch GPT analysis and update ``GPT_ADVICE``."""

    global GPT_ADVICE, _GPT_ADVICE_ERROR_COUNT
    if _GPT_SAFE_MODE:
        GPT_ADVICE = GPTAdviceModel(signal="hold")
        logger.warning("GPT safe mode active; skipping advice refresh")
        return

    GPT_ADVICE = GPTAdviceModel()
    try:
        symbol = symbols[0]
        hist = price_history[symbol]
        price = hist[-1] if hist else 0.0
        features = await build_feature_vector(symbol, price)
        rsi = features[-1]
        ema = compute_ema(list(hist))
        prompt = (
            "Using market data:\n"
            f"price={price}, EMA={ema}, RSI={rsi}.\n"
            'Return JSON {"signal": "buy|sell|hold", "tp_mult": float, "sl_mult": float}.'
        )
        gpt_result = await query_gpt_json_async(prompt)
        try:
            advice = GPTAdviceModel.model_validate(gpt_result)
        except ValidationError as exc:
            logger.warning("Invalid GPT advice: %s", sanitize_log_value(str(exc)))
            advice = GPTAdviceModel(signal="hold")
        GPT_ADVICE = advice
        _GPT_ADVICE_ERROR_COUNT = 0
        logger.info("GPT analysis: %s", advice.model_dump())
    except GPTClientJSONError as exc:
        GPT_ADVICE = GPTAdviceModel(signal="hold")
        await send_telegram_alert(f"Invalid JSON from GPT: {exc}")
        logger.debug("GPT analysis failed: %s", exc)
        await _record_gpt_failure(set_trading_enabled, str(exc))
    except GPTClientError as exc:  # pragma: no cover - non-critical
        GPT_ADVICE = GPTAdviceModel(signal="hold")
        logger.debug("GPT analysis failed: %s", exc)
        await _record_gpt_failure(set_trading_enabled, str(exc))


async def gpt_advice_loop(
    symbols: list[str],
    price_history: dict[str, Any],
    build_feature_vector: Callable[[str, float], Awaitable[list[float]]],
    compute_ema: Callable[[list[float]], float],
    set_trading_enabled: Callable[[bool], Awaitable[None]],
) -> None:
    while True:
        try:
            await refresh_gpt_advice(
                symbols,
                price_history,
                build_feature_vector,
                compute_ema,
                set_trading_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - keep loop running
            logger.warning("GPT advice loop error: %s", sanitize_log_value(str(exc)))
            await send_telegram_alert(f"GPT advice loop error: {exc}")
        await asyncio.sleep(3600)
