"""Main entry point for the trading bot."""

import asyncio
import os
import statistics
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Awaitable

from model_builder_client import schedule_retrain

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from bot.config import BotConfig
from bot.gpt_client import GPTClientError, query_gpt_async
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
        async with httpx.AsyncClient(trust_env=False) as client:
            await client.post(
                url, data={"chat_id": chat_id, "text": message}, timeout=5
            )
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Failed to send Telegram alert: %s", exc)


_TASKS: set[asyncio.Task[None]] = set()


def _task_done(task: asyncio.Task[None]) -> None:
    """Remove completed ``task`` and log any unhandled exception."""
    _TASKS.discard(task)
    with suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc:
            logger.error("run_async task failed", exc_info=exc)


def run_async(coro: Awaitable[None]) -> None:
    """Run or schedule ``coro`` depending on event loop state.

    When scheduled, keep a reference to the task and log exceptions on
    completion.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
    else:
        task = asyncio.create_task(coro)
        _TASKS.add(task)
        task.add_done_callback(_task_done)

# Threshold for slow trade confirmations
CONFIRMATION_TIMEOUT = safe_float("ORDER_CONFIRMATION_TIMEOUT", 5.0)

# Keep a short history of prices to derive simple features such as
# price change (used as a lightweight volume proxy) and a moving
# average.  This avoids additional service calls while still allowing
# us to build a small feature vector for the prediction service.
_PRICE_HISTORY: deque[float] = deque(maxlen=50)
PRICE_HISTORY_LOCK = asyncio.Lock()


# Default trading symbol. Override with the SYMBOL environment variable.
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = safe_float("INTERVAL", 5.0)
# How often to retrain the reference model (seconds)
TRAIN_INTERVAL = safe_float("TRAIN_INTERVAL", 24 * 60 * 60)

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
        services["gptoss"] = (env["gptoss_api"], "health")
    async with httpx.AsyncClient(trust_env=False) as client:
        async def _probe(name: str, url: str, endpoint: str) -> str | None:
            for attempt in range(retries):
                try:
                    resp = await client.get(f"{url}/{endpoint}", timeout=5)
                    if resp.status_code == 200:
                        return None
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Ping failed for %s (%s): %s", name, attempt + 1, exc
                    )
                await asyncio.sleep(delay)
            logger.error("Service %s unreachable at %s", name, url)
            return f"{name} service is not available"

        tasks = [
            _probe(name, url, endpoint)
            for name, (url, endpoint) in services.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    errors: list[str] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("Service check raised: %s", result)
            errors.append(str(result))
        elif result:
            errors.append(result)
    if errors:
        raise SystemExit("; ".join(errors))




@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
async def fetch_price(symbol: str, env: dict) -> float | None:
    """Return current price or ``None`` if the request fails."""
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
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


async def build_feature_vector(price: float) -> list[float]:
    """Derive a feature vector from recent price history.

    The vector includes:

    1. ``price`` – latest price.
    2. ``volume`` – price change since last observation as a proxy for volume.
    3. ``sma`` – simple moving average of recent prices.
    4. ``volatility`` – standard deviation of recent price changes.
    5. ``rsi`` – Relative Strength Index over the recent window.
    """

    async with PRICE_HISTORY_LOCK:
        _PRICE_HISTORY.append(price)

        if len(_PRICE_HISTORY) > 1:
            volume = _PRICE_HISTORY[-1] - _PRICE_HISTORY[-2]
            deltas = [
                _PRICE_HISTORY[i] - _PRICE_HISTORY[i - 1]
                for i in range(1, len(_PRICE_HISTORY))
            ]
            volatility = (
                statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            )
        else:
            volume = 0.0
            volatility = 0.0

        sma = statistics.fmean(_PRICE_HISTORY)

        rsi_period = 14
        if len(_PRICE_HISTORY) > rsi_period:
            gains: list[float] = []
            losses: list[float] = []
            for i in range(len(_PRICE_HISTORY) - rsi_period, len(_PRICE_HISTORY)):
                change = _PRICE_HISTORY[i] - _PRICE_HISTORY[i - 1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(-change)
            avg_gain = statistics.fmean(gains) if gains else 0.0
            avg_loss = statistics.fmean(losses) if losses else 0.0
            if avg_loss == 0:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - 100 / (1 + rs)
        else:
            rsi = 50.0

    return [price, volume, sma, volatility, rsi]


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
async def get_prediction(symbol: str, features: list[float], env: dict) -> dict | None:
    """Return raw model prediction output for the given ``features``."""
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
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


def _build_trade_payload(
    symbol: str,
    side: str,
    price: float,
    tp: float | None,
    sl: float | None,
    trailing_stop: float | None,
) -> tuple[dict, dict, float]:
    """Return payload, headers and timeout for trade requests."""

    payload = {"symbol": symbol, "side": side, "price": price}
    if tp is not None:
        payload["tp"] = tp
    if sl is not None:
        payload["sl"] = sl
    if trailing_stop is not None:
        payload["trailing_stop"] = trailing_stop

    headers: dict[str, str] = {}
    token = os.getenv("TRADE_MANAGER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = safe_float("TRADE_MANAGER_TIMEOUT", 5.0)
    return payload, headers, timeout


def _handle_trade_response(
    resp: httpx.Response, symbol: str, start: float
) -> tuple[bool, float, str | None]:
    """Return success flag, elapsed time and error message."""

    elapsed = time.time() - start
    try:
        data = resp.json()
    except ValueError:
        data = {}
        error = "invalid JSON"
        return False, elapsed, error

    error: str | None = None
    if resp.status_code != 200:
        error = f"HTTP {resp.status_code}"
    elif data.get("error"):
        error = str(data.get("error"))
    elif data.get("status") not in (None, "ok", "success"):
        error = str(data.get("status"))

    return error is None, elapsed, error


async def _post_trade(
    client: httpx.Client | httpx.AsyncClient,
    symbol: str,
    side: str,
    price: float,
    env: dict,
    tp: float | None,
    sl: float | None,
    trailing_stop: float | None,
) -> tuple[bool, float, str | None]:
    """Execute trade request using ``client`` and return result details."""

    start = time.time()
    payload, headers, timeout = _build_trade_payload(
        symbol, side, price, tp, sl, trailing_stop
    )
    url = f"{env['trade_manager_url']}/open_position"
    post = client.post
    if asyncio.iscoroutinefunction(post):
        resp = await post(url, json=payload, timeout=timeout, headers=headers or None)
    else:
        resp = post(url, json=payload, timeout=timeout, headers=headers or None)
    return _handle_trade_response(resp, symbol, start)


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
        with httpx.Client(trust_env=False) as client:
            ok, elapsed, error = asyncio.run(
                _post_trade(
                    client, symbol, side, price, env, tp, sl, trailing_stop
                )
            )
        if elapsed > CONFIRMATION_TIMEOUT:
            run_async(
                send_telegram_alert(
                    f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
                )
            )
        if not ok:
            logger.error("Trade manager error: %s", error)
            run_async(
                send_telegram_alert(
                    f"Trade manager responded with {error} for {symbol}"
                )
            )
        return ok
    except httpx.HTTPError as exc:
        logger.error("Trade manager request error: %s", exc)
        run_async(
            send_telegram_alert(
                f"Trade manager request failed for {symbol}: {exc}"
            )
        )
        return False


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
async def send_trade_async(
    symbol: str,
    side: str,
    price: float,
    env: dict,
    tp: float | None = None,
    sl: float | None = None,
    trailing_stop: float | None = None,
) -> bool:
    """Asynchronously send trade request to trade manager."""

    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            ok, elapsed, error = await _post_trade(
                client, symbol, side, price, env, tp, sl, trailing_stop
            )
        if elapsed > CONFIRMATION_TIMEOUT:
            await send_telegram_alert(
                f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
            )
        if not ok:
            logger.error("Trade manager error: %s", error)
            await send_telegram_alert(
                f"Trade manager responded with {error} for {symbol}"
            )
        return ok
    except httpx.HTTPError as exc:
        logger.error("Trade manager request error: %s", exc)
        await send_telegram_alert(
            f"Trade manager request failed for {symbol}: {exc}"
        )
        return False


async def monitor_positions(env: dict, interval: float = INTERVAL) -> None:
    """Poll open positions and close them when exit conditions are met."""
    trail_state: dict[str, float] = {}
    async with httpx.AsyncClient(trust_env=False) as client:
        while True:
            try:
                resp = await client.get(
                    f"{env['trade_manager_url']}/positions", timeout=5
                )
                positions = resp.json().get("positions", [])
            except (httpx.HTTPError, ValueError) as exc:
                logger.error("Failed to fetch positions: %s", exc)
                positions = []
            symbols: list[str] = []
            for pos in positions:
                sym = pos.get("symbol")
                if sym and sym not in symbols:
                    symbols.append(sym)

            prices: dict[str, float] = {}
            if symbols:
                tasks = [fetch_price(sym, env) for sym in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sym, result in zip(symbols, results):
                    if isinstance(result, Exception) or result is None:
                        logger.error("Price task failed for %s: %s", sym, result)
                    else:
                        prices[sym] = result

            for pos in positions:
                order_id = pos.get("id")
                symbol = pos.get("symbol")
                side = pos.get("side")
                tp = pos.get("tp")
                sl = pos.get("sl")
                trailing = pos.get("trailing_stop")
                entry = pos.get("entry_price")
                if not order_id or not symbol or not side:
                    continue
                price = prices.get(symbol)
                if price is None:
                    continue
                reason = None
                if side == "buy":
                    if tp is not None and price >= tp:
                        reason = "tp"
                    elif sl is not None and price <= sl:
                        reason = "sl"
                    else:
                        base = trail_state.get(order_id, entry or price)
                        base = max(base, price)
                        trail_state[order_id] = base
                        if trailing is not None and price <= base - trailing:
                            reason = "trailing_stop"
                    close_side = "sell"
                else:
                    if tp is not None and price <= tp:
                        reason = "tp"
                    elif sl is not None and price >= sl:
                        reason = "sl"
                    else:
                        base = trail_state.get(order_id, entry or price)
                        base = min(base, price)
                        trail_state[order_id] = base
                        if trailing is not None and price >= base + trailing:
                            reason = "trailing_stop"
                    close_side = "buy"

                if reason:
                    try:
                        await client.post(
                            f"{env['trade_manager_url']}/close_position",
                            json={"order_id": order_id, "side": close_side},
                            timeout=5,
                        )
                    except httpx.HTTPError as exc:
                        logger.error("Failed to close position %s: %s", order_id, exc)
                    trail_state.pop(order_id, None)
            await asyncio.sleep(interval)


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

    env_tp, env_sl, env_ts = _parse_trade_params(
        os.getenv("TP"), os.getenv("SL"), os.getenv("TRAILING_STOP")
    )

    def _resolve(
        value: float | None, env_value: float | None, multiplier: float
    ) -> float | None:
        if value is not None:
            return value
        if env_value is not None:
            return env_value
        if price is not None:
            return price * multiplier
        return None

    tp = _resolve(tp, env_tp, CFG.tp_multiplier)
    sl = _resolve(sl, env_sl, CFG.sl_multiplier)
    trailing_stop = _resolve(
        trailing_stop, env_ts, CFG.trailing_stop_multiplier
    )

    return tp, sl, trailing_stop


async def reactive_trade(symbol: str, env: dict | None = None) -> None:
    """Asynchronously fetch prediction and open position if signaled."""
    env = env or _load_env()
    async with httpx.AsyncClient(trust_env=False) as client:
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
            features = await build_feature_vector(price)
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
            await send_trade_async(
                symbol,
                signal,
                price,
                env,
                tp=tp,
                sl=sl,
                trailing_stop=trailing_stop,
            )
        except httpx.HTTPError as exc:
            logger.error("Reactive trade request error: %s", exc)


async def run_once_async() -> None:
    """Execute a single trading cycle."""
    env = _load_env()
    price = await fetch_price(SYMBOL, env)
    if price is None or price <= 0:
        logger.warning("Invalid price for %s: %s", SYMBOL, price)
        return
    logger.info("Price for %s: %s", SYMBOL, price)
    features = await build_feature_vector(price)
    pdata = await get_prediction(SYMBOL, features, env)
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
        await send_trade_async(
            SYMBOL,
            signal,
            price,
            env,
            tp=tp,
            sl=sl,
            trailing_stop=trailing_stop,
        )




async def main_async() -> None:
    """Run the trading bot until interrupted."""
    train_task = None
    try:
        await check_services()
        env = _load_env()
        train_task = schedule_retrain(env["model_builder_url"], TRAIN_INTERVAL)
        try:
            strategy_code = (
                Path(__file__).with_name("strategy_optimizer.py").read_text(encoding="utf-8")
            )
            gpt_result = await query_gpt_async(
                "Что ты видишь в этом коде:\n" + strategy_code
            )
            logger.info("GPT analysis: %s", gpt_result)
        except GPTClientError as exc:  # pragma: no cover - non-critical
            logger.debug("GPT analysis failed: %s", exc)
        while True:
            await run_once_async()
            await asyncio.sleep(INTERVAL)
    except KeyboardInterrupt:
        logger.info('Stopping trading bot')
    finally:
        if train_task:
            train_task.cancel()
            with suppress(asyncio.CancelledError):
                await train_task


def main() -> None:
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
