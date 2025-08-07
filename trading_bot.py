"""Main entry point for the trading bot."""

import asyncio
import os
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Awaitable

import httpx
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from bot.config import BotConfig
from bot.gpt_client import query_gpt
from bot.utils import logger

load_dotenv()

CFG = BotConfig()


def safe_int(env_var: str, default: int) -> int:
    """Return int value of ``env_var`` or ``default`` on failure."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Invalid %s value '%s', using default %s", env_var, value, default
        )
        return default


def safe_float(env_var: str, default: float) -> float:
    """Return float value of ``env_var`` or ``default`` on failure."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Invalid %s value '%s', using default %s", env_var, value, default
        )
        return default


async def send_telegram_alert(message: str) -> None:
    """Send a Telegram notification if credentials are configured."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                url, data={"chat_id": chat_id, "text": message}, timeout=5
            )
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Failed to send Telegram alert: %s", exc)


def run_async(coro: Awaitable[None]) -> None:
    """Run or schedule ``coro`` depending on event loop state."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
    else:
        asyncio.create_task(coro)

# Threshold for slow trade confirmations
CONFIRMATION_TIMEOUT = safe_float("ORDER_CONFIRMATION_TIMEOUT", 5.0)

# Keep a short history of prices to derive simple features such as
# price change (used as a lightweight volume proxy) and a moving
# average.  This avoids additional service calls while still allowing
# us to build a small feature vector for the prediction service.
_PRICE_HISTORY: deque[float] = deque(maxlen=50)
PRICE_HISTORY_LOCK = threading.Lock()
PRICE_HISTORY_ASYNC_LOCK = asyncio.Lock()


# Default trading symbol. Override with the SYMBOL environment variable.
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = safe_float("INTERVAL", 5.0)

# Default retry values for service availability checks
DEFAULT_SERVICE_CHECK_RETRIES = 30
DEFAULT_SERVICE_CHECK_DELAY = 2.0


def _load_env() -> dict:
    """Load service URLs from environment variables.

    If explicit ``*_URL`` variables are not provided, fall back to the ``HOST``
    value when constructing defaults. This allows running the bot locally by
    specifying only ``HOST`` without overriding every service URL.
    """

    host = os.getenv("HOST")
    data_handler = os.getenv("DATA_HANDLER_URL")
    model_builder = os.getenv("MODEL_BUILDER_URL")
    trade_manager = os.getenv("TRADE_MANAGER_URL")
    gptoss_api = os.getenv("GPT_OSS_API")

    if data_handler is None:
        data_handler = f"http://{host}:8000" if host else "http://data_handler:8000"
    if model_builder is None:
        model_builder = f"http://{host}:8001" if host else "http://model_builder:8001"
    if trade_manager is None:
        trade_manager = f"http://{host}:8002" if host else "http://trade_manager:8002"

    return {
        "data_handler_url": data_handler,
        "model_builder_url": model_builder,
        "trade_manager_url": trade_manager,
        "gptoss_api": gptoss_api,
    }


async def check_services() -> None:
    """Ensure dependent services are responsive."""
    env = _load_env()
    retries = safe_int("SERVICE_CHECK_RETRIES", DEFAULT_SERVICE_CHECK_RETRIES)
    delay = safe_float("SERVICE_CHECK_DELAY", DEFAULT_SERVICE_CHECK_DELAY)
    services = {
        "data_handler": (env["data_handler_url"], "ping"),
        "model_builder": (env["model_builder_url"], "ping"),
        "trade_manager": (env["trade_manager_url"], "ready"),
    }
    if env.get("gptoss_api"):
        services["gptoss"] = (env["gptoss_api"], "v1/health")
    async with httpx.AsyncClient() as client:
        for name, (url, endpoint) in services.items():
            for attempt in range(retries):
                try:
                    resp = await client.get(f"{url}/{endpoint}", timeout=5)
                    if resp.status_code == 200:
                        break
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Ping failed for %s (%s): %s", name, attempt + 1, exc
                    )
                await asyncio.sleep(delay)
            else:
                logger.error("Service %s unreachable at %s", name, url)
                raise SystemExit(f"{name} service is not available")




@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
async def fetch_price(symbol: str, env: dict) -> float | None:
    """Return current price or ``None`` if the request fails."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{env['data_handler_url']}/price/{symbol}", timeout=5
            )
        try:
            data = resp.json()
        except ValueError:
            logger.error("Invalid JSON from price service")
            return None
        if resp.status_code != 200 or "error" in data:
            err = data.get("error", f"HTTP {resp.status_code}")
            logger.error("Failed to fetch price: %s", err)
            return None
        price = data.get("price")
        if price is None or price <= 0:
            logger.error("Invalid price for %s: %s", symbol, price)
            return None
        return price
    except httpx.HTTPError as exc:
        logger.error("Price request error: %s", exc)
        return None


def build_feature_vector(price: float) -> list[float]:
    """Derive a simple feature vector from the latest price.

    The feature vector consists of:

    1. ``price`` – the latest price value.
    2. ``volume`` – approximated by the price change since the previous
       observation.  This acts as a very lightweight proxy when real
       volume data is unavailable.
    3. ``sma`` – a simple moving average of recent prices acting as a
       basic technical indicator.
    """

    with PRICE_HISTORY_LOCK:
        _PRICE_HISTORY.append(price)
        if len(_PRICE_HISTORY) > 1:
            volume = _PRICE_HISTORY[-1] - _PRICE_HISTORY[-2]
        else:
            volume = 0.0
        sma = statistics.fmean(_PRICE_HISTORY)
    return [price, volume, sma]


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
async def get_prediction(symbol: str, features: list[float], env: dict) -> dict | None:
    """Return raw model prediction output for the given ``features``."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{env['model_builder_url']}/predict",
                json={"symbol": symbol, "features": features},
                timeout=5,
            )
        if resp.status_code != 200:
            logger.error("Model prediction failed: HTTP %s", resp.status_code)
            return None
        try:
            return resp.json()
        except ValueError:
            logger.error("Invalid JSON from model prediction")
            return None
    except httpx.HTTPError as exc:
        logger.error("Model request error: %s", exc)
        return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
def send_trade(
    symbol: str,
    side: str,
    price: float,
    env: dict,
    tp: float | None = None,
    sl: float | None = None,
    trailing_stop: float | None = None,
) -> bool:
    """Send trade request to trade manager.

    Returns ``True`` when the trade manager reports success, otherwise ``False``.
    """
    try:
        timeout = safe_float("TRADE_MANAGER_TIMEOUT", 5.0)
        start = time.time()
        payload = {"symbol": symbol, "side": side, "price": price}
        if tp is not None:
            payload["tp"] = tp
        if sl is not None:
            payload["sl"] = sl
        if trailing_stop is not None:
            payload["trailing_stop"] = trailing_stop
        headers = {}
        token = os.getenv("TRADE_MANAGER_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(
            f"{env['trade_manager_url']}/open_position",
            json=payload,
            timeout=timeout,
            headers=headers or None,
        )
        elapsed = time.time() - start
        if elapsed > CONFIRMATION_TIMEOUT:
            run_async(
                send_telegram_alert(
                    f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
                )
            )
        try:
            data = resp.json()
        except ValueError:
            data = {}
            logger.error("Trade manager returned invalid JSON")
            run_async(
                send_telegram_alert(
                    f"Trade manager invalid response for {symbol}"
                )
            )
            return False
        error: str | None = None
        if resp.status_code != 200:
            error = f"HTTP {resp.status_code}"
        elif data.get("error"):
            error = str(data.get("error"))
        elif data.get("status") not in (None, "ok", "success"):
            error = str(data.get("status"))
        if error:
            logger.error("Trade manager error: %s", error)
            run_async(
                send_telegram_alert(
                    f"Trade manager responded with {error} for {symbol}"
                )
            )
            return False
        return True
    except requests.RequestException as exc:
        logger.error("Trade manager request error: %s", exc)
        run_async(
            send_telegram_alert(
                f"Trade manager request failed for {symbol}: {exc}"
            )
        )
        return False


def _parse_trade_params(
    tp: float | str | None = None,
    sl: float | str | None = None,
    trailing_stop: float | str | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Safely convert trade parameters to floats when possible."""

    def _parse(value: float | str | None) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return _parse(tp), _parse(sl), _parse(trailing_stop)


def _resolve_trade_params(
    tp: float | None,
    sl: float | None,
    trailing_stop: float | None,
    price: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Return TP/SL/trailing-stop values.

    Starting from ``tp``, ``sl`` and ``trailing_stop`` (already parsed if
    provided), fill in missing values from environment variables or fall back
    to config multipliers using ``price``.
    """

    if tp is None:
        env_tp, _, _ = _parse_trade_params(os.getenv("TP"))
        if env_tp is not None:
            tp = env_tp
        elif price is not None:
            tp = price * CFG.tp_multiplier

    if sl is None:
        _, env_sl, _ = _parse_trade_params(None, os.getenv("SL"))
        if env_sl is not None:
            sl = env_sl
        elif price is not None:
            sl = price * CFG.sl_multiplier

    if trailing_stop is None:
        _, _, env_ts = _parse_trade_params(None, None, os.getenv("TRAILING_STOP"))
        if env_ts is not None:
            trailing_stop = env_ts
        elif price is not None:
            trailing_stop = price * CFG.trailing_stop_multiplier

    return tp, sl, trailing_stop


async def reactive_trade(symbol: str, env: dict | None = None) -> None:
    """Asynchronously fetch prediction and open position if signaled."""
    env = env or _load_env()
    timeout = safe_float("TRADE_MANAGER_TIMEOUT", 5.0)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{env['data_handler_url']}/price/{symbol}", timeout=5.0
            )
            if resp.status_code != 200:
                logger.error("Failed to fetch price: HTTP %s", resp.status_code)
                return
            try:
                price = resp.json().get("price", 0)
            except ValueError:
                logger.error("Invalid JSON from price service")
                return
            if price is None or price <= 0:
                logger.warning("Invalid price for %s: %s", symbol, price)
                return
            async with PRICE_HISTORY_ASYNC_LOCK:
                features = build_feature_vector(price)
            pred = await client.post(
                f"{env['model_builder_url']}/predict",
                json={"symbol": symbol, "features": features},
                timeout=5.0,
            )
            if pred.status_code != 200:
                logger.error("Model prediction failed: HTTP %s", pred.status_code)
                return
            try:
                pdata = pred.json()
            except ValueError:
                logger.error("Invalid JSON from model prediction")
                return
            signal = pdata.get("signal")
            if not signal:
                return
            tp, sl, trailing_stop = _parse_trade_params(
                pdata.get("tp"), pdata.get("sl"), pdata.get("trailing_stop")
            )
            tp, sl, trailing_stop = _resolve_trade_params(tp, sl, trailing_stop, price)
            payload = {"symbol": symbol, "side": signal, "price": price}
            if tp is not None:
                payload["tp"] = tp
            if sl is not None:
                payload["sl"] = sl
            if trailing_stop is not None:
                payload["trailing_stop"] = trailing_stop
            start = time.time()
            trade_resp = await client.post(
                f"{env['trade_manager_url']}/open_position",
                json=payload,
                timeout=timeout,
            )
            elapsed = time.time() - start
            if elapsed > CONFIRMATION_TIMEOUT:
                await send_telegram_alert(
                    f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
                )
            try:
                data = trade_resp.json()
            except ValueError:
                logger.error("Trade manager returned invalid JSON")
                await send_telegram_alert(
                    f"Trade manager invalid response for {symbol}"
                )
                return
            error: str | None = None
            if trade_resp.status_code != 200:
                error = f"HTTP {trade_resp.status_code}"
            elif data.get("error"):
                error = str(data.get("error"))
            elif data.get("status") not in (None, "ok", "success"):
                error = str(data.get("status"))
            if error:
                logger.error("Trade manager error: %s", error)
                await send_telegram_alert(
                    f"Trade manager responded with {error} for {symbol}"
                )
                return
        except httpx.HTTPError as exc:
            logger.error("Reactive trade request error: %s", exc)


def run_once() -> None:
    """Execute a single trading cycle."""
    env = _load_env()
    price = asyncio.run(fetch_price(SYMBOL, env))
    if price is None or price <= 0:
        logger.warning("Invalid price for %s: %s", SYMBOL, price)
        return
    logger.info("Price for %s: %s", SYMBOL, price)
    features = build_feature_vector(price)
    pdata = asyncio.run(get_prediction(SYMBOL, features, env))
    signal = pdata.get("signal") if pdata else None
    logger.info("Prediction: %s", signal)
    if signal:
        tp, sl, trailing_stop = _parse_trade_params(
            pdata.get("tp") if pdata else None,
            pdata.get("sl") if pdata else None,
            pdata.get("trailing_stop") if pdata else None,
        )
        tp, sl, trailing_stop = _resolve_trade_params(tp, sl, trailing_stop, price)
        logger.info("Sending trade: %s %s @ %s", SYMBOL, signal, price)
        send_trade(
            SYMBOL,
            signal,
            price,
            env,
            tp=tp,
            sl=sl,
            trailing_stop=trailing_stop,
        )


def main():
    """Run the trading bot until interrupted."""
    try:
        asyncio.run(check_services())
        try:
            strategy_code = (
                Path(__file__).with_name("strategy_optimizer.py").read_text(encoding="utf-8")
            )
            gpt_result = query_gpt(
                "Что ты видишь в этом коде:\n" + strategy_code
            )
            logger.info("GPT analysis: %s", gpt_result)
        except Exception as exc:  # pragma: no cover - non-critical
            logger.debug("GPT analysis failed: %s", exc)
        while True:
            run_once()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        logger.info('Stopping trading bot')


if __name__ == '__main__':
    main()
