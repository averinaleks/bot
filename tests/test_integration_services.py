import os
import time
import requests
import multiprocessing
from data_handler import api_app as dh_app
from model_builder import api_app as mb_app
from trade_manager import api_app as tm_app
import trading_bot


def _run(app, port):
    app.run(port=port)


def test_services_communicate():
    processes = [
        multiprocessing.Process(target=_run, args=(dh_app, 8000)),
        multiprocessing.Process(target=_run, args=(mb_app, 8001)),
        multiprocessing.Process(target=_run, args=(tm_app, 8002)),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    os.environ.update({
        'DATA_HANDLER_URL': 'http://localhost:8000',
        'MODEL_BUILDER_URL': 'http://localhost:8001',
        'TRADE_MANAGER_URL': 'http://localhost:8002',
    })
    try:
        trading_bot.run_once()
        resp = requests.get('http://localhost:8002/positions', timeout=5)
        data = resp.json()
        assert data['positions'], 'position was not created'
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
