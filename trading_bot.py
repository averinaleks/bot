"""Main entry point for the trading bot."""

import os
import time

import httpx
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from bot.utils import logger


def send_telegram_alert(message: str) -> None:
    """Send a Telegram notification if credentials are configured."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to send Telegram alert: %s", exc)

# Threshold for slow trade confirmations
CONFIRMATION_TIMEOUT = float(os.getenv("ORDER_CONFIRMATION_TIMEOUT", "5"))


# Default trading symbol. Override with the SYMBOL environment variable.
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = float(os.getenv("INTERVAL", "5"))

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
    }


def check_services() -> None:
    """Ensure dependent services are responsive."""
    env = _load_env()
    retries = int(
        os.getenv("SERVICE_CHECK_RETRIES", str(DEFAULT_SERVICE_CHECK_RETRIES))
    )
    delay = float(
        os.getenv("SERVICE_CHECK_DELAY", str(DEFAULT_SERVICE_CHECK_DELAY))
    )
    services = {
        "data_handler": (env["data_handler_url"], "ping"),
        "model_builder": (env["model_builder_url"], "ping"),
        "trade_manager": (env["trade_manager_url"], "ready"),
    }
    for name, (url, endpoint) in services.items():
        for attempt in range(retries):
            try:
                resp = requests.get(f"{url}/{endpoint}", timeout=5)
                if resp.status_code == 200:
                    break
            except requests.RequestException as exc:
                logger.warning(
                    "Ping failed for %s (%s): %s", name, attempt + 1, exc
                )
            time.sleep(delay)
        else:
            logger.error("Service %s unreachable at %s", name, url)
            raise SystemExit(f"{name} service is not available")




@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
def fetch_price(symbol: str, env: dict) -> float | None:
    """Return current price or None if request failed."""
    try:
        resp = requests.get(f"{env['data_handler_url']}/price/{symbol}", timeout=5)
        data = resp.json()
        if resp.status_code != 200 or "error" in data:
            err = data.get("error", f"HTTP {resp.status_code}")
            logger.error("Failed to fetch price: %s", err)
            return None
        price = data.get("price")
        if price is None or price <= 0:
            logger.error("Invalid price for %s: %s", symbol, price)
            return None
        return price
    except requests.RequestException as exc:
        logger.error("Price request error: %s", exc)
        return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
def get_prediction(symbol: str, price: float, env: dict) -> dict | None:
    """Return raw model prediction output if available."""
    try:
        resp = requests.post(
            f"{env['model_builder_url']}/predict",
            json={"symbol": symbol, "features": [price]},
            timeout=5,
        )
        if resp.status_code != 200:
            logger.error("Model prediction failed: HTTP %s", resp.status_code)
            return None
        return resp.json()
    except requests.RequestException as exc:
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
) -> None:
    """Send trade request to trade manager."""
    try:
        timeout = float(os.getenv("TRADE_MANAGER_TIMEOUT", "5"))
        start = time.time()
        payload = {"symbol": symbol, "side": side, "price": price}
        if tp is not None:
            payload["tp"] = tp
        if sl is not None:
            payload["sl"] = sl
        if trailing_stop is not None:
            payload["trailing_stop"] = trailing_stop
        resp = requests.post(
            f"{env['trade_manager_url']}/open_position",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - start
        if elapsed > CONFIRMATION_TIMEOUT:
            send_telegram_alert(
                f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
            )
        if resp.status_code != 200:
            logger.error("Trade manager error: HTTP %s", resp.status_code)
            send_telegram_alert(
                f"Trade manager responded with HTTP {resp.status_code} for {symbol}"
            )
    except requests.RequestException as exc:
        logger.error("Trade manager request error: %s", exc)
        send_telegram_alert(f"Trade manager request failed for {symbol}: {exc}")


async def reactive_trade(symbol: str, env: dict | None = None) -> None:
    """Asynchronously fetch prediction and open position if signaled."""
    env = env or _load_env()
    timeout = float(os.getenv("TRADE_MANAGER_TIMEOUT", "5"))
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{env['data_handler_url']}/price/{symbol}", timeout=5.0
            )
            if resp.status_code != 200:
                logger.error("Failed to fetch price: HTTP %s", resp.status_code)
                return
            price = resp.json().get("price", 0)
            if price is None or price <= 0:
                logger.warning("Invalid price for %s: %s", symbol, price)
                return
            pred = await client.post(
                f"{env['model_builder_url']}/predict",
                json={"symbol": symbol, "features": [price]},
                timeout=5.0,
            )
            if pred.status_code != 200:
                logger.error("Model prediction failed: HTTP %s", pred.status_code)
                return
            pdata = pred.json()
            signal = pdata.get("signal")
            if not signal:
                return
            tp = pdata.get("tp")
            sl = pdata.get("sl")
            trailing_stop = pdata.get("trailing_stop")
            if tp is None:
                t = os.getenv("TP")
                tp = float(t) if t else None
            if sl is None:
                s = os.getenv("SL")
                sl = float(s) if s else None
            if trailing_stop is None:
                ts = os.getenv("TRAILING_STOP")
                trailing_stop = float(ts) if ts else None
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
                send_telegram_alert(
                    f"⚠️ Slow TradeManager response {elapsed:.2f}s for {symbol}"
                )
            if trade_resp.status_code != 200:
                logger.error(
                    "Trade manager error: HTTP %s", trade_resp.status_code
                )
        except httpx.HTTPError as exc:
            logger.error("Reactive trade request error: %s", exc)


def run_once() -> None:
    """Execute a single trading cycle."""
    env = _load_env()
    price = fetch_price(SYMBOL, env)
    if price is None or price <= 0:
        logger.warning("Invalid price for %s: %s", SYMBOL, price)
        return
    logger.info("Price for %s: %s", SYMBOL, price)
    pdata = get_prediction(SYMBOL, price, env)
    signal = pdata.get("signal") if pdata else None
    logger.info("Prediction: %s", signal)
    if signal:
        tp = pdata.get("tp") if pdata else None
        sl = pdata.get("sl") if pdata else None
        trailing_stop = pdata.get("trailing_stop") if pdata else None
        if tp is None:
            t = os.getenv("TP")
            tp = float(t) if t else None
        if sl is None:
            s = os.getenv("SL")
            sl = float(s) if s else None
        if trailing_stop is None:
            ts = os.getenv("TRAILING_STOP")
            trailing_stop = float(ts) if ts else None
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
    # Load environment variables from a .env file when running as a script
    load_dotenv()
    try:
        check_services()
        while True:
            run_once()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        logger.info('Stopping trading bot')


if __name__ == '__main__':
    main()
