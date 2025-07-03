import os
import time
import requests
from dotenv import load_dotenv
from utils import logger
from tenacity import retry, wait_exponential, stop_after_attempt


# Default trading symbol. Override with the SYMBOL environment variable.
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = float(os.getenv("INTERVAL", "5"))


def _load_env() -> dict:
    """Load service URLs from environment variables."""
    return {
        "data_handler_url": os.getenv("DATA_HANDLER_URL", "http://data_handler:8000"),
        "model_builder_url": os.getenv("MODEL_BUILDER_URL", "http://model_builder:8001"),
        "trade_manager_url": os.getenv("TRADE_MANAGER_URL", "http://trade_manager:8002"),
    }


def check_services() -> None:
    """Ensure dependent services respond to `/ping`."""
    env = _load_env()
    retries = int(os.getenv("SERVICE_CHECK_RETRIES", "5"))
    delay = float(os.getenv("SERVICE_CHECK_DELAY", "1"))
    services = {
        "data_handler": env["data_handler_url"],
        "model_builder": env["model_builder_url"],
        "trade_manager": env["trade_manager_url"],
    }
    for name, url in services.items():
        for attempt in range(retries):
            try:
                resp = requests.get(f"{url}/ping", timeout=5)
                if resp.status_code == 200:
                    break
            except requests.RequestException as exc:
                logger.warning("Ping failed for %s (%s): %s", name, attempt + 1, exc)
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
    env = _load_env()
    price = fetch_price(SYMBOL, env)
    if price is None:
        return
    signal = get_prediction(SYMBOL, price, env)
    if signal:
        send_trade(SYMBOL, signal, price, env)


def main():
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
