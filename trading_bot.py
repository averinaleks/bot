"""Main entry point for the trading bot."""

import atexit
import asyncio
import json
import logging
import math
import os
import statistics
import time
from collections import defaultdict, deque
from contextlib import suppress
from typing import Awaitable, Callable, TypeVar

from services.stubs import create_httpx_stub, create_pydantic_stub, is_offline_env

_OFFLINE_ENV = is_offline_env()

try:  # pragma: no cover - fallback executed in offline/testing scenarios
    if _OFFLINE_ENV:
        raise ImportError("offline mode uses httpx stub")
    import httpx as _httpx  # type: ignore
except Exception:  # noqa: BLE001 - ensure stubs are used when dependencies missing
    httpx = create_httpx_stub()
else:
    httpx = _httpx

try:  # pragma: no cover - fallback executed in offline/testing scenarios
    if _OFFLINE_ENV:
        raise ImportError("offline mode uses pydantic stub")
    from bot.pydantic_compat import BaseModel, ConfigDict, ValidationError
except Exception:  # noqa: BLE001 - ensure stubs are used when dependencies missing
    BaseModel, ConfigDict, ValidationError = create_pydantic_stub()

from bot.config import BotConfig, OFFLINE_MODE
from bot.dotenv_utils import load_dotenv
from bot.gpt_client import GPTClientError, GPTClientJSONError, query_gpt_json_async
from telegram_logger import TelegramLogger
from utils import retry, suppress_tf_logs
from services.logging_utils import sanitize_log_value

try:  # pragma: no cover - optional dependency
    import ccxt  # type: ignore
except Exception:  # noqa: BLE001 - broad to avoid test import errors
    ccxt = type("ccxt_stub", (), {})()

BybitError = getattr(ccxt, "BaseError", Exception)
NetworkError = getattr(ccxt, "NetworkError", BybitError)

CFG = BotConfig()

logger = logging.getLogger("TradingBot")


class GPTAdviceModel(BaseModel):
    """Model for parsing GPT advice responses."""

    signal: float | str | None = None
    tp_mult: float | None = None
    sl_mult: float | None = None
    model_config = ConfigDict(validate_assignment=False)

    def __setattr__(self, name, value):  # pragma: no cover - simple assignment logic
        super().__setattr__(name, value)
        if name == "signal" and value is None:
            super().__setattr__("tp_mult", None)
            super().__setattr__("sl_mult", None)


GPT_ADVICE = GPTAdviceModel()


class ServiceUnavailableError(Exception):
    """Raised when required services are not reachable."""


T = TypeVar("T", int, float)


def safe_number(env_var: str, default: T, cast: Callable[[str], T]) -> T:
    """Return ``env_var`` cast by ``cast`` or ``default`` on failure or invalid value."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        result = cast(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s value '%s', using default %s",
            sanitize_log_value(env_var),
            sanitize_log_value(value),
            default,
        )
        return default
    if isinstance(result, float) and not math.isfinite(result):
        logger.warning(
            "Invalid %s value '%s', using default %s",
            sanitize_log_value(env_var),
            sanitize_log_value(value),
            default,
        )
        return default
    if result <= 0:
        logger.warning(
            "Non-positive %s value '%s', using default %s",
            sanitize_log_value(env_var),
            sanitize_log_value(value),
            default,
        )
        return default
    return result


def safe_int(env_var: str, default: int) -> int:
    """Return int value of ``env_var`` or ``default`` on failure or non-positive value."""
    return safe_number(env_var, default, int)


def safe_float(env_var: str, default: float) -> float:
    """Return float value of ``env_var`` or ``default`` on failure or non-positive value."""
    return safe_number(env_var, default, float)


GPT_ADVICE_MAX_ATTEMPTS = safe_int("GPT_ADVICE_MAX_ATTEMPTS", 3)
_GPT_ADVICE_ERROR_COUNT = 0
_GPT_SAFE_MODE = False


async def send_telegram_alert(message: str) -> None:
    """Send a Telegram notification if credentials are configured."""
    if OFFLINE_MODE:
        logger.debug("Offline mode enabled, Telegram alert suppressed: %s", message)
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram inactive, message not sent: %s", message)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    client = await get_http_client()
    max_attempts = safe_int("TELEGRAM_ALERT_RETRIES", 3)
    delay = 1
    payload = {"chat_id": chat_id, "text": message}
    for attempt in range(1, max_attempts + 1):
        try:
                try:
                    resp = await client.post(url, json=payload, timeout=10)
                except TypeError:
                    resp = await client.post(url, data=payload, timeout=10)
                raise_for_status = getattr(resp, "raise_for_status", None)
                if callable(raise_for_status):
                    raise_for_status()
                return
        except httpx.HTTPError as exc:  # pragma: no cover - network errors
            req_url = getattr(getattr(exc, "request", None), "url", url)
            redacted_url = str(req_url).replace(token, "***")
            logger.warning(
                "Failed to send Telegram alert (attempt %s/%s): %s (%s) %s",
                attempt,
                max_attempts,
                redacted_url,
                exc.__class__.__name__,
                str(exc).replace(token, "***"),
            )
            if attempt == max_attempts:
                logger.error(
                    "Failed to send Telegram alert after %s attempts: %s",
                    max_attempts,
                    message,
                )
                if CFG.save_unsent_telegram:
                    _logger = type("_TL", (), {"unsent_path": CFG.unsent_telegram_path})()
                    TelegramLogger._save_unsent(_logger, chat_id, message)
                return
            await asyncio.sleep(delay)
            delay *= 2


_TASKS: set[asyncio.Task[None]] = set()
_TASKS_LOCK = asyncio.Lock()


def _task_done(task: asyncio.Task[None]) -> None:
    """Remove completed ``task`` and log any unhandled exception."""

    async def _remove() -> None:
        async with _TASKS_LOCK:
            _TASKS.discard(task)

    asyncio.create_task(_remove())
    with suppress(asyncio.CancelledError):
        exc = task.exception()
        if exc:
            logger.error("run_async task failed", exc_info=exc)


def run_async(coro: Awaitable[None], timeout: float | None = None) -> None:
    """Run or schedule ``coro`` depending on event loop state.

    When scheduled, keep a reference to the task and log exceptions on
    completion.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(asyncio.wait_for(coro, timeout))
    else:
        task = asyncio.create_task(asyncio.wait_for(coro, timeout))
        task.add_done_callback(_task_done)

        async def _add() -> None:
            async with _TASKS_LOCK:
                _TASKS.add(task)

        asyncio.create_task(_add())


async def shutdown_async_tasks(timeout: float = 5.0) -> None:
    """Wait for all scheduled tasks to complete.

    Tasks that do not finish within ``timeout`` seconds are logged and
    cancelled.
    """
    async with _TASKS_LOCK:
        tasks = set(_TASKS)

    if not tasks:
        return

    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        logger.warning("Cancelling pending tasks: %s", [repr(t) for t in pending])
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async with _TASKS_LOCK:
        _TASKS.clear()

# Threshold for slow trade confirmations
CONFIRMATION_TIMEOUT = safe_float("ORDER_CONFIRMATION_TIMEOUT", 5.0)

# Keep a short history of prices per symbol to derive simple features such as
# price change (used as a lightweight volume proxy) and a moving average.
# Use a larger window to accommodate EMA/RSI calculations.
_PRICE_HISTORY: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=200))
PRICE_HISTORY_LOCK = asyncio.Lock()

# Track model performance
_PRED_RESULTS: deque[int] = deque(maxlen=CFG.prediction_history_size)
_LAST_PREDICTION: int | None = None


# Default trading symbols; overridden from configuration at runtime.
SYMBOLS: list[str] = ["BTCUSDT"]

# Maximum allowed capital exposure across all open positions. When not set, no
# limit is enforced.
CAPITAL_LIMIT = safe_float("CAPITAL_LIMIT", float("inf"))


def _compute_ema(prices: list[float], period: int = 10) -> float:
    """Return exponential moving average for ``prices``."""
    if not prices:
        return 0.0
    alpha = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema
INTERVAL = safe_float("INTERVAL", 5.0)
# How often to retrain the reference model (seconds)
TRAIN_INTERVAL = safe_float("TRAIN_INTERVAL", 24 * 60 * 60)

# Default retry values for service availability checks
DEFAULT_SERVICE_CHECK_RETRIES = 30
DEFAULT_SERVICE_CHECK_DELAY = 2.0

# Global flag toggled via Telegram commands to enable/disable trading
_TRADING_ENABLED: bool = True
_TRADING_ENABLED_LOCK = asyncio.Lock()


async def get_trading_enabled() -> bool:
    """Return the current trading enabled state."""
    async with _TRADING_ENABLED_LOCK:
        return _TRADING_ENABLED


async def set_trading_enabled(value: bool) -> None:
    """Set the trading enabled state to ``value``."""
    global _TRADING_ENABLED
    async with _TRADING_ENABLED_LOCK:
        _TRADING_ENABLED = value

# Shared HTTP client for outgoing requests
HTTP_CLIENT: httpx.AsyncClient | None = None
HTTP_CLIENT_LOCK = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    """Return a shared HTTP client instance.

    Timeout for requests can be configured via the ``HTTP_CLIENT_TIMEOUT``
    environment variable (default 5 seconds).
    """
    global HTTP_CLIENT
    async with HTTP_CLIENT_LOCK:
        if HTTP_CLIENT is None:
            timeout = safe_float("HTTP_CLIENT_TIMEOUT", 5.0)
            kwargs = {"trust_env": False, "timeout": timeout}
            if OFFLINE_MODE or getattr(httpx, "__offline_stub__", False):
                logger.debug("Offline HTTP client instantiated")
            try:
                HTTP_CLIENT = httpx.AsyncClient(**kwargs)
            except TypeError:  # pragma: no cover - stub may ignore kwargs
                HTTP_CLIENT = httpx.AsyncClient()
    return HTTP_CLIENT


async def close_http_client() -> None:
    """Close the module-level HTTP client if it exists."""
    global HTTP_CLIENT
    await shutdown_async_tasks()
    if HTTP_CLIENT is not None:
        close = getattr(HTTP_CLIENT, "aclose", None)
        if callable(close):
            await close()
        HTTP_CLIENT = None



def _cleanup_http_client() -> None:
    """Synchronously close the shared HTTP client.

    If an event loop is already running (e.g. in environments like Jupyter),
    ``asyncio.run`` will raise ``RuntimeError``. In that case schedule the
    cleanup on the existing loop and return immediately.
    """

    coro = close_http_client()
    try:
        asyncio.run(coro)
    except RuntimeError:
        coro.close()
        with suppress(RuntimeError):
            loop = asyncio.get_running_loop()
            loop.create_task(close_http_client())


atexit.register(_cleanup_http_client)


def _load_env() -> dict:
    """Load service URLs from environment variables.

    If explicit ``*_URL`` variables are not provided, fall back to the ``HOST``
    value when constructing defaults. This allows running the bot locally by
    specifying only ``HOST`` without overriding every service URL.
    """

    host = os.getenv("HOST")
    scheme = os.getenv("SERVICE_SCHEME", "http")
    data_handler = os.getenv("DATA_HANDLER_URL")
    model_builder = os.getenv("MODEL_BUILDER_URL")
    trade_manager = os.getenv("TRADE_MANAGER_URL")
    gptoss_api = os.getenv("GPT_OSS_API")

    if data_handler is None:
        data_handler = (
            f"{scheme}://{host}:8000" if host else f"{scheme}://data_handler:8000"
        )
    if model_builder is None:
        model_builder = (
            f"{scheme}://{host}:8001" if host else f"{scheme}://model_builder:8001"
        )
    if trade_manager is None:
        trade_manager = (
            f"{scheme}://{host}:8002" if host else f"{scheme}://trade_manager:8002"
        )

    return {
        "data_handler_url": data_handler,
        "model_builder_url": model_builder,
        "trade_manager_url": trade_manager,
        "gptoss_api": gptoss_api,
    }


async def check_services() -> None:
    """Ensure dependent services are responsive."""
    if OFFLINE_MODE:
        logger.info("Offline mode enabled, skipping service availability check")
        return
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
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        async def _probe(name: str, url: str, endpoint: str) -> str | None:
            for attempt in range(retries):
                try:
                    resp = await client.get(f"{url}/{endpoint}", timeout=5)
                    if resp.status_code == 200:
                        return None
                    if resp.status_code != 200:
                        logger.warning(
                            "Ping failed for %s (%s): HTTP %s",
                            name,
                            attempt + 1,
                            resp.status_code,
                        )
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
        raise ServiceUnavailableError("; ".join(errors))




@retry(3, lambda attempt: min(2 ** attempt, 5))
async def fetch_price(symbol: str, env: dict) -> float | None:
    """Return current price or ``None`` if the request fails."""
    if OFFLINE_MODE:
        logger.debug("Offline mode: price fetch skipped for %s", symbol)
        return None
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
        if not isinstance(price, (int, float)):
            logger.error("Invalid price type for %s: %r", symbol, price)
            return None
        if price <= 0:
            logger.error("Invalid price for %s: %s", symbol, price)
            return None
        return price
    except httpx.HTTPError as exc:
        logger.error("Price request error: %s", exc)
        return None


async def fetch_initial_history(symbol: str, env: dict) -> None:
    """Populate ``_PRICE_HISTORY`` with initial OHLCV data."""
    if OFFLINE_MODE:
        logger.debug("Offline mode: initial history fetch skipped for %s", symbol)
        return
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            resp = await client.get(
                f"{env['data_handler_url']}/history/{symbol}", timeout=5
            )
            data = resp.json() if resp.status_code == 200 else {}
            candles = data.get("history", [])
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch initial history: %s", exc)
            candles = []
    async with PRICE_HISTORY_LOCK:
        hist = _PRICE_HISTORY[symbol]
        hist.clear()
        for candle in candles:
            if len(candle) > 4:
                try:
                    hist.append(float(candle[4]))
                except (TypeError, ValueError):
                    continue


async def build_feature_vector(symbol: str, price: float) -> list[float]:
    """Derive a feature vector from recent price history.

    The vector includes:

    1. ``price`` - latest price.
    2. ``volume`` - price change since last observation as a proxy for volume.
    3. ``sma`` - simple moving average of recent prices.
    4. ``volatility`` - standard deviation of recent price changes.
    5. ``rsi`` - Relative Strength Index over the recent window.
    """

    async with PRICE_HISTORY_LOCK:
        hist = _PRICE_HISTORY[symbol]
        hist.append(price)

        if len(hist) > 1:
            volume = hist[-1] - hist[-2]
            deltas = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
            volatility = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        else:
            volume = 0.0
            volatility = 0.0

        sma = statistics.fmean(hist)

        rsi_period = 14
        if len(hist) > rsi_period:
            gains: list[float] = []
            losses: list[float] = []
            for i in range(len(hist) - rsi_period, len(hist)):
                change = hist[i] - hist[i - 1]
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


@retry(3, lambda attempt: min(2 ** attempt, 5))
async def get_prediction(symbol: str, features: list[float], env: dict) -> dict | None:
    """Return raw model prediction output for the given ``features``."""
    if OFFLINE_MODE:
        logger.debug("Offline mode: prediction request skipped for %s", symbol)
        return None
    try:
        payload = json.dumps({"symbol": symbol, "features": features}).encode()
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        }
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                resp = await client.post(
                    f"{env['model_builder_url']}/predict",
                    content=payload,
                    headers=headers,
                    timeout=5,
                )
            except TypeError:  # pragma: no cover - fallback for stub clients
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

    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

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
    client: httpx.AsyncClient,
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
    body = json.dumps(payload).encode()
    headers["Content-Type"] = "application/json"
    headers["Content-Length"] = str(len(body))
    try:
        try:
            resp = await client.post(
                url,
                content=body,
                timeout=timeout,
                headers=headers or None,
            )
        except TypeError:  # pragma: no cover - fallback for stub clients
            resp = await client.post(
                url,
                json=payload,
                timeout=timeout,
                headers=headers or None,
            )
        return _handle_trade_response(resp, symbol, start)
    except httpx.TimeoutException:
        return False, time.time() - start, "request timed out"
    except httpx.ConnectError:
        return False, time.time() - start, "connection error"
    except httpx.HTTPError as exc:  # pragma: no cover - other network errors
        return False, time.time() - start, str(exc)


async def send_trade_async(
    client: httpx.AsyncClient,
    symbol: str,
    side: str,
    price: float,
    env: dict,
    tp: float | None = None,
    sl: float | None = None,
    trailing_stop: float | None = None,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
) -> tuple[bool, str | None]:
    """Asynchronously send trade request to trade manager.

    Returns a tuple ``(ok, error)`` where ``error`` is ``None`` on success.
    """

    if OFFLINE_MODE:
        logger.info(
            "Offline mode: trade request suppressed for %s (%s)",
            symbol,
            side,
        )
        return True, None

    for attempt in range(1, max_attempts + 1):
        ok, elapsed, error = await _post_trade(
            client, symbol, side, price, env, tp, sl, trailing_stop
        )
        if ok:
            if elapsed > CONFIRMATION_TIMEOUT:
                await send_telegram_alert(
                    f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
                )
            return True, None
        msg = error or "unknown error"
        if attempt < max_attempts:
            logger.warning(
                "Retrying order for %s (attempt %s/%s): %s",
                symbol,
                attempt,
                max_attempts,
                msg,
            )
            await asyncio.sleep(retry_delay)
            continue
        logger.error("Failed to place order for %s: %s", symbol, msg)
        await send_telegram_alert(
            f"Trade manager request failed for {symbol}: {msg}"
        )
        return False, msg


async def close_position_async(
    client: httpx.AsyncClient,
    env: dict,
    order_id: str,
    side: str,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
) -> tuple[bool, str | None]:
    """Close an existing position via the trade manager.

    Returns a tuple ``(ok, error)`` similar to :func:`send_trade_async`.
    """

    if OFFLINE_MODE:
        logger.info(
            "Offline mode: close position request suppressed for %s (%s)",
            order_id,
            side,
        )
        return True, None

    url = f"{env['trade_manager_url']}/close_position"
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(
                url,
                json={"order_id": order_id, "side": side},
                timeout=5,
            )
            if response.status_code == 200:
                return True, None
            msg = f"HTTP {response.status_code} - {response.text}"
        except httpx.TimeoutException:
            msg = "request timed out"
        except httpx.ConnectError:
            msg = "connection error"
        except httpx.HTTPError as exc:  # pragma: no cover - other network errors
            msg = str(exc)
        if attempt < max_attempts:
            logger.warning(
                "Retrying close for %s (attempt %s/%s): %s",
                order_id,
                attempt,
                max_attempts,
                msg,
            )
            await asyncio.sleep(retry_delay)
            continue
        logger.error("Failed to close position %s: %s", order_id, msg)
        return False, msg


async def monitor_positions(env: dict, interval: float = INTERVAL) -> None:
    """Poll open positions and close them when exit conditions are met."""
    if OFFLINE_MODE:
        logger.info("Offline mode: position monitoring disabled")
        return
    trail_state: dict[str, float] = {}
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        while True:
            try:
                resp = await client.get(
                    f"{env['trade_manager_url']}/positions", timeout=5
                )
                if resp.status_code != 200:
                    logger.error(
                        "Failed to fetch positions: status code %s", resp.status_code
                    )
                    await asyncio.sleep(interval)
                    continue
                positions = resp.json().get("positions", [])
            except (httpx.HTTPError, ValueError) as exc:
                logger.error("Failed to fetch positions: %s", exc)
                await asyncio.sleep(interval)
                continue
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
                    ok, err = await close_position_async(
                        client, env, order_id, close_side
                    )
                    if not ok and err:
                        await send_telegram_alert(
                            f"Failed to close position {order_id}: {err}"
                        )
                    trail_state.pop(order_id, None)
            await asyncio.sleep(interval)


async def _total_position_notional(env: dict) -> float:
    """Return total notional value of all open positions."""
    if OFFLINE_MODE:
        logger.debug("Offline mode: assuming zero position notional")
        return 0.0
    client = await get_http_client()
    try:
        resp = await client.get(f"{env['trade_manager_url']}/positions", timeout=5)
        if resp.status_code != 200:
            logger.error(
                "Failed to fetch positions: status code %s", resp.status_code
            )
            return 0.0
        positions = resp.json().get("positions", [])
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Failed to fetch positions: %s", exc)
        return 0.0

    total = 0.0
    missing: list[tuple[str, float]] = []
    for pos in positions:
        symbol = pos.get("symbol")
        try:
            amount = float(pos.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        price = pos.get("entry_price")
        if amount <= 0 or not symbol:
            continue
        if price is None:
            missing.append((symbol, amount))
        else:
            try:
                total += amount * float(price)
            except (TypeError, ValueError):
                continue

    if missing:
        tasks = [fetch_price(sym, env) for sym, _ in missing]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (sym, amt), res in zip(missing, results):
            if isinstance(res, Exception) or res is None:
                logger.error("Price task failed for %s: %s", sym, res)
            else:
                total += amt * res
    return total


async def capital_under_limit(env: dict) -> bool:
    """Return ``True`` if total exposure is below ``CAPITAL_LIMIT``."""
    if not math.isfinite(CAPITAL_LIMIT):
        return True
    exposure = await _total_position_notional(env)
    return exposure < CAPITAL_LIMIT


def _parse_trade_params(
    tp: float | str | None = None,
    sl: float | str | None = None,
    trailing_stop: float | str | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Safely convert trade parameters to floats when possible."""

    def _parse(value: float | str | None) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid trade parameter %r: %s", value, exc)
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

    tp_mult = CFG.tp_multiplier
    sl_mult = CFG.sl_multiplier
    if GPT_ADVICE.signal in {"buy", "sell"}:
        if GPT_ADVICE.tp_mult is not None:
            tp_mult *= float(GPT_ADVICE.tp_mult)
        if GPT_ADVICE.sl_mult is not None:
            sl_mult *= float(GPT_ADVICE.sl_mult)

    tp = _resolve(tp, env_tp, tp_mult)
    sl = _resolve(sl, env_sl, sl_mult)
    trailing_stop = _resolve(
        trailing_stop, env_ts, CFG.trailing_stop_multiplier
    )

    return tp, sl, trailing_stop


def _is_trade_allowed(
    symbol: str | None,
    model_signal: str,
    prob: float | None,
    threshold: float | None,
) -> bool:
    if symbol is None:
        symbol = SYMBOLS[0]

    gpt_signal = GPT_ADVICE.signal
    if gpt_signal and gpt_signal not in {"buy", "sell", "hold"}:
        logger.info("Invalid GPT advice %s", gpt_signal)
        return False

    prices = list(_PRICE_HISTORY.get(symbol, []))
    ema_signal = None
    if prices:
        ema = _compute_ema(prices)
        if prices[-1] > ema:
            ema_signal = "buy"
        elif prices[-1] < ema:
            ema_signal = "sell"

    weights = {"model": CFG.transformer_weight, "ema": CFG.ema_weight}
    scores = {"buy": 0.0, "sell": 0.0}
    if model_signal == "buy":
        scores["buy"] += weights["model"]
    else:
        scores["sell"] += weights["model"]
    if ema_signal == "buy":
        scores["buy"] += weights["ema"]
    elif ema_signal == "sell":
        scores["sell"] += weights["ema"]
    if gpt_signal == "buy":
        weights["gpt"] = CFG.gpt_weight
        scores["buy"] += weights["gpt"]
    elif gpt_signal == "sell":
        weights["gpt"] = CFG.gpt_weight
        scores["sell"] += weights["gpt"]

    total_weight = sum(weights.values())
    final: str | None = None
    if scores["buy"] > scores["sell"] and scores["buy"] >= total_weight / 2:
        final = "buy"
    elif scores["sell"] > scores["buy"] and scores["sell"] >= total_weight / 2:
        final = "sell"

    if final and final != model_signal:
        logger.info(
            "Weighted advice %s conflicts with model signal %s", final, model_signal
        )
        return False

    if prob is None or threshold is None:
        return final == model_signal

    if model_signal == "buy":
        if prob is not None and threshold is not None:
            return final == model_signal and prob >= threshold
        return final == model_signal
    else:
        if prob is not None and threshold is not None:
            return final == model_signal and prob <= 1 - threshold
        return final == model_signal


def should_trade(
    model_signal: str,
    prob: float = 1.0,
    threshold: float = 0.5,
    symbol: str | None = None,
) -> bool:
    """Return ``True`` if the weighted advice supports the model signal."""
    return _is_trade_allowed(symbol, model_signal, prob, threshold)


async def _enter_gpt_safe_mode(reason: str | None = None) -> None:
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


async def _record_gpt_failure(reason: str | None = None) -> None:
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
        await _enter_gpt_safe_mode(reason)


async def refresh_gpt_advice() -> None:
    """Fetch GPT analysis and update ``GPT_ADVICE``."""
    global GPT_ADVICE, _GPT_ADVICE_ERROR_COUNT
    if _GPT_SAFE_MODE:
        GPT_ADVICE = GPTAdviceModel(signal="hold")
        logger.warning("GPT safe mode active; skipping advice refresh")
        return

    GPT_ADVICE = GPTAdviceModel()
    try:
        symbol = SYMBOLS[0]
        hist = _PRICE_HISTORY[symbol]
        price = hist[-1] if hist else 0.0
        features = await build_feature_vector(symbol, price)
        rsi = features[-1]
        ema = _compute_ema(list(hist))
        prompt = (
            "На основании рыночных данных:\n"
            f"price={price}, EMA={ema}, RSI={rsi}.\n"
            "Дай JSON {\"signal\": 'buy'|'sell'|'hold', \"tp_mult\": float, \"sl_mult\": float}."
        )
        gpt_result = await query_gpt_json_async(prompt)
        try:
            advice = GPTAdviceModel.model_validate(gpt_result)
        except ValidationError as exc:
            logger.warning("Invalid GPT advice: %s", exc)
            advice = GPTAdviceModel(signal="hold")
        GPT_ADVICE = advice
        _GPT_ADVICE_ERROR_COUNT = 0
        logger.info("GPT analysis: %s", advice.model_dump())
    except GPTClientJSONError as exc:
        GPT_ADVICE = GPTAdviceModel(signal="hold")
        await send_telegram_alert(f"Некорректный JSON от GPT: {exc}")
        logger.debug("GPT analysis failed: %s", exc)
        await _record_gpt_failure(str(exc))
    except GPTClientError as exc:  # pragma: no cover - non-critical
        GPT_ADVICE = GPTAdviceModel(signal="hold")
        logger.debug("GPT analysis failed: %s", exc)
        await _record_gpt_failure(str(exc))


async def _gpt_advice_loop() -> None:
    while True:
        try:
            await refresh_gpt_advice()
        except Exception as exc:  # noqa: BLE001 - keep loop running
            logger.warning("GPT advice loop error: %s", exc)
            await send_telegram_alert(f"GPT advice loop error: {exc}")
        await asyncio.sleep(3600)


async def reactive_trade(symbol: str, env: dict | None = None) -> None:
    """Asynchronously fetch prediction and open position if signaled."""
    if OFFLINE_MODE:
        logger.info("Offline mode: reactive trade skipped for %s", symbol)
        return
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
            features = await build_feature_vector(symbol, price)
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
                client,
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


async def run_once_async(symbol: str | None = None) -> None:
    """Execute a single trading cycle for ``symbol``."""

    env = _load_env()
    if symbol is None:
        symbol = SYMBOLS[0]

    if not await get_trading_enabled():
        logger.info("Trading disabled")
        return

    if not await capital_under_limit(env):
        logger.warning("Capital limit reached, skipping %s", symbol)
        await send_telegram_alert(f"Capital limit reached for {symbol}")
        return

    price = await fetch_price(symbol, env)
    if price is None or price <= 0:
        return

    features = await build_feature_vector(symbol, price)
    prediction = await get_prediction(symbol, features, env)
    if not prediction:
        return

    signal = prediction.get("signal")
    if not signal:
        return
    if (
        CFG.get("gpt_weight", 0) >= 1.0
        and GPT_ADVICE.signal in {"buy", "sell"}
        and GPT_ADVICE.signal != signal
    ):
        logger.info(
            "GPT advice %s conflicts with model signal %s for %s, skipping",
            GPT_ADVICE.signal,
            signal,
            symbol,
        )
        return
    prob = prediction.get("prob")
    threshold = float(prediction.get("threshold", 0.5))

    logger.info("Prediction for %s: %s", symbol, signal)

    if not should_trade(signal, prob, threshold, symbol):
        logger.info("Trade for %s vetoed by weighted advice", symbol)
        return

    if prob is not None and prob < threshold:
        logger.info(
            "Probability %.3f below threshold %.3f for %s",
            prob,
            threshold,
            symbol,
        )
        return

    tp, sl, trailing_stop = _parse_trade_params(
        prediction.get("tp"), prediction.get("sl"), prediction.get("trailing_stop")
    )
    tp, sl, trailing_stop = _resolve_trade_params(tp, sl, trailing_stop, price)

    client = await get_http_client()
    await send_trade_async(
        client,
        symbol,
        signal,
        price,
        env,
        tp=tp,
        sl=sl,
        trailing_stop=trailing_stop,
    )

async def main_async() -> None:
    # Run the trading bot until interrupted.
    if OFFLINE_MODE:
        logger.info("Offline mode enabled, trading loop not started")
        return
    train_task = None
    monitor_task = None
    gpt_task = None
    try:
        await check_services()
        gpt_task = asyncio.create_task(_gpt_advice_loop())
        while True:
            try:
                for symbol in SYMBOLS:
                    await run_once_async(symbol)
            except ServiceUnavailableError as exc:
                logger.error("Service availability check failed: %s", exc)
                await send_telegram_alert(
                    f"Service availability check failed: {exc}"
                )
            except (NetworkError, httpx.HTTPError) as exc:
                logger.exception("Network error in main loop: %s", exc)
                await send_telegram_alert(f"Network error in main loop: {exc}")
            except GPTClientError as exc:
                logger.exception("GPT client error in main loop: %s", exc)
                await send_telegram_alert(f"GPT client error in main loop: {exc}")
            except BybitError as exc:
                logger.exception("Bybit error in main loop: %s", exc)
                await send_telegram_alert(f"Bybit error in main loop: {exc}")
            await asyncio.sleep(INTERVAL)
    except ServiceUnavailableError as exc:
        logger.error("Service availability check failed: %s", exc)
        await send_telegram_alert(f"Service availability check failed: {exc}")
    except (NetworkError, httpx.HTTPError) as exc:  # pragma: no cover - startup network
        logger.exception("Network error in main_async: %s", exc)
        await send_telegram_alert(f"Network error in main_async: {exc}")
    except GPTClientError as exc:  # pragma: no cover - startup GPT errors
        logger.exception("GPT client error in main_async: %s", exc)
        await send_telegram_alert(f"GPT client error in main_async: {exc}")
    except BybitError as exc:  # pragma: no cover - startup Bybit errors
        logger.exception("Bybit error in main_async: %s", exc)
        await send_telegram_alert(f"Bybit error in main_async: {exc}")
    except KeyboardInterrupt:
        logger.info('Stopping trading bot')
    finally:
        if monitor_task:
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task
        if train_task:
            train_task.cancel()
            with suppress(asyncio.CancelledError):
                await train_task
        if gpt_task:
            gpt_task.cancel()
            with suppress(asyncio.CancelledError):
                await gpt_task


def main() -> None:
    from data_handler import get_settings  # local import to avoid circular dependency

    load_dotenv()
    try:
        cfg = get_settings()
    except ValidationError as exc:  # pragma: no cover - config errors
        logger.error("Invalid environment configuration: %s", exc)
        raise SystemExit(1)
    suppress_tf_logs()
    global SYMBOLS
    SYMBOLS = cfg.symbols
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        logger.warning(
            "Telegram inactive: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set"
        )
    asyncio.run(main_async())


if __name__ == '__main__':
    from bot.utils import configure_logging

    configure_logging()
    main()
