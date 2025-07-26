"""Main entry point for the trading bot."""

import os
import time
import requests
from dotenv import load_dotenv
from utils import logger
from tenacity import retry, wait_exponential, stop_after_attempt


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
        if resp.status_code != 200:
            logger.error("Failed to fetch price: HTTP %s", resp.status_code)
            return None
        return resp.json().get("price", 0)
    except requests.RequestException as exc:
        logger.error("Price request error: %s", exc)
        return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
def get_prediction(symbol: str, price: float, env: dict) -> str | None:
    """Return model signal if available."""
    try:
        resp = requests.post(
            f"{env['model_builder_url']}/predict",
            json={"symbol": symbol, "features": [price]},
            timeout=5,
        )
        if resp.status_code != 200:
            logger.error("Model prediction failed: HTTP %s", resp.status_code)
            return None
        return resp.json().get("signal")
    except requests.RequestException as exc:
        logger.error("Model request error: %s", exc)
        return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(3))
def send_trade(symbol: str, side: str, price: float, env: dict) -> None:
    """Send trade request to trade manager."""
    try:
        timeout = float(os.getenv("TRADE_MANAGER_TIMEOUT", "5"))
        resp = requests.post(
            f"{env['trade_manager_url']}/open_position",
            json={"symbol": symbol, "side": side, "price": price},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.error("Trade manager error: HTTP %s", resp.status_code)
    except requests.RequestException as exc:
        logger.error("Trade manager request error: %s", exc)


def run_once() -> None:
    """Execute a single trading cycle."""
    env = _load_env()
    price = fetch_price(SYMBOL, env)
    if price is None:
        return
    logger.info("Price for %s: %s", SYMBOL, price)
    signal = get_prediction(SYMBOL, price, env)
    logger.info("Prediction: %s", signal)
    if signal:
        logger.info("Sending trade: %s %s @ %s", SYMBOL, signal, price)
        send_trade(SYMBOL, signal, price, env)


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
