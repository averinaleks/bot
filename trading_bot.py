import os
import time
import requests

DEFAULT_DATA_HANDLER_URL = 'http://data_handler:8000'
DEFAULT_MODEL_BUILDER_URL = 'http://model_builder:8001'
DEFAULT_TRADE_MANAGER_URL = 'http://trade_manager:8002'
DEFAULT_SYMBOL = 'TEST'
DEFAULT_INTERVAL = 5.0


def _load_env():
    """Return runtime configuration from environment variables."""
    return {
        'data_handler_url': os.getenv('DATA_HANDLER_URL', DEFAULT_DATA_HANDLER_URL),
        'model_builder_url': os.getenv('MODEL_BUILDER_URL', DEFAULT_MODEL_BUILDER_URL),
        'trade_manager_url': os.getenv('TRADE_MANAGER_URL', DEFAULT_TRADE_MANAGER_URL),
        'symbol': os.getenv('SYMBOL', DEFAULT_SYMBOL),
        'interval': float(os.getenv('INTERVAL', str(DEFAULT_INTERVAL))),
    }


def check_services() -> bool:
    """Return True if all dependent services respond to /ping."""
    env = _load_env()
    services = {
        'data_handler': env['data_handler_url'],
        'model_builder': env['model_builder_url'],
        'trade_manager': env['trade_manager_url'],
    }
    for name, url in services.items():
        try:
            resp = requests.get(f"{url}/ping", timeout=5)
            if resp.status_code != 200:
                print(f"Service {name} at {url} returned {resp.status_code}")
                return False
        except requests.RequestException as e:
            print(f"Service {name} at {url} unavailable: {e}")
            return False
    return True


def run_once():
    env = _load_env()
    price_resp = requests.get(f"{env['data_handler_url']}/price/{env['symbol']}", timeout=5)
    price = price_resp.json().get('price', 0)

    model_resp = requests.post(
        f"{env['model_builder_url']}/predict",
        json={'symbol': env['symbol'], 'price': price},
        timeout=5,
    )
    signal = model_resp.json().get('signal')

    if signal:
        requests.post(
            f"{env['trade_manager_url']}/open_position",
            json={'symbol': env['symbol'], 'side': signal, 'price': price},
            timeout=5,
        )


def main():
    if not check_services():
        raise SystemExit('dependent services are unavailable')
    while True:
        env = _load_env()
        run_once()
        time.sleep(env['interval'])


if __name__ == '__main__':
    main()
