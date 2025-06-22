import os
import time
import requests

DATA_HANDLER_URL = os.getenv('DATA_HANDLER_URL', 'http://data_handler:8000')
MODEL_BUILDER_URL = os.getenv('MODEL_BUILDER_URL', 'http://model_builder:8001')
TRADE_MANAGER_URL = os.getenv('TRADE_MANAGER_URL', 'http://trade_manager:8002')
SYMBOL = os.getenv('SYMBOL', 'TEST')
INTERVAL = float(os.getenv('INTERVAL', '5'))


def run_once():
    price_resp = requests.get(f"{DATA_HANDLER_URL}/price/{SYMBOL}", timeout=5)
    price = price_resp.json().get('price', 0)

    model_resp = requests.post(
        f"{MODEL_BUILDER_URL}/predict",
        json={'symbol': SYMBOL, 'price': price},
        timeout=5,
    )
    signal = model_resp.json().get('signal')

    if signal:
        requests.post(
            f"{TRADE_MANAGER_URL}/open_position",
            json={'symbol': SYMBOL, 'side': signal, 'price': price},
            timeout=5,
        )


def main():
    while True:
        run_once()
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
