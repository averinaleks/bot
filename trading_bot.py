import os
import time
import asyncio
import requests
from dotenv import load_dotenv
from utils import logger
from config import load_config, BotConfig
from tenacity import retry, wait_exponential, stop_after_attempt

# Automatically load environment variables from a .env file if present
load_dotenv()

SYMBOL = os.getenv("SYMBOL", "TEST")
INTERVAL = float(os.getenv("INTERVAL", "5"))
CONFIG: BotConfig = load_config("config.json")


def _load_env() -> dict:
    """Load service URLs from environment variables."""
    return {
        "data_handler_url": os.getenv("DATA_HANDLER_URL", "http://data_handler:8000"),
        "model_builder_url": os.getenv("MODEL_BUILDER_URL", "http://model_builder:8001"),
        "trade_manager_url": os.getenv("TRADE_MANAGER_URL", "http://trade_manager:8002"),
    }


def check_services(
    retries: int | None = None,
    delay: float | None = None,
) -> bool:
    """Return True if all dependent services respond to /ping.

    The ``SERVICE_CHECK_RETRIES`` and ``SERVICE_CHECK_DELAY`` environment
    variables can override the default attempt count (5) and delay between
    attempts (2 seconds).

    Each service is queried several times with a small delay between attempts
    to allow containers time to start accepting connections.
    """
    if retries is None:
        retries = int(os.getenv("SERVICE_CHECK_RETRIES", "5"))
    if delay is None:
        delay = float(os.getenv("SERVICE_CHECK_DELAY", "2.0"))

    env = _load_env()
    services = {
        "data_handler": env["data_handler_url"],
        "model_builder": env["model_builder_url"],
        "trade_manager": env["trade_manager_url"],
    }

    for name, url in services.items():
        for _ in range(retries):
            try:
                resp = requests.get(f"{url}/ping", timeout=5)
                if resp.status_code == 200:
                    break
                logger.error(
                    "Service %s at %s returned %s", name, url, resp.status_code
                )
            except requests.RequestException as exc:
                logger.error("Service %s at %s unavailable: %s", name, url, exc)
            time.sleep(delay)
        else:
            logger.error(
                "Service %s at %s did not respond after %s retries."
                " Increase SERVICE_CHECK_RETRIES or SERVICE_CHECK_DELAY if"
                " your containers need more time to start.",
                name,
                url,
                retries,
            )
            return False
    return True


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
            json={"symbol": symbol, "price": price},
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
        resp = requests.post(
            f"{env['trade_manager_url']}/open_position",
            json={"symbol": symbol, "side": side, "price": price},
            timeout=5,
        )
        if resp.status_code != 200:
            logger.error("Trade manager error: HTTP %s", resp.status_code)
    except requests.RequestException as exc:
        logger.error("Trade manager request error: %s", exc)


def run_once() -> None:
    env = _load_env()
    price = fetch_price(SYMBOL, env)
    if price is None:
        return
    signal = get_prediction(SYMBOL, price, env)
    if signal:
        send_trade(SYMBOL, signal, price, env)


def main():
    if not check_services():
        logger.error(
            "Dependent services are unavailable. Adjust SERVICE_CHECK_RETRIES "
            "and SERVICE_CHECK_DELAY in your .env if startup is slow."
        )
        raise SystemExit('dependent services are unavailable')
    try:
        while True:
            run_once()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        logger.info('Stopping trading bot')
    finally:
        dh = globals().get('data_handler')
        if dh is not None and hasattr(dh, 'stop'):
            asyncio.run(dh.stop())


if __name__ == '__main__':
    main()
