import multiprocessing
import os
import time
import types
import requests

ctx = multiprocessing.get_context("spawn")


def _run_dh():
    class DummyExchange:
        def fetch_ticker(self, symbol):
            return {'last': 42.0}
        def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
            return [[1, 1, 1, 1, 1, 1]]
    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    os.environ['STREAM_SYMBOLS'] = ''
    os.environ['HOST'] = '127.0.0.1'
    from bot.services import data_handler_service
    data_handler_service.app.run(host='127.0.0.1', port=8000)


def test_data_handler_service_price():
    p = ctx.Process(target=_run_dh)
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get('http://127.0.0.1:8000/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.get('http://127.0.0.1:8000/price/BTCUSDT', timeout=5)
        assert resp.status_code == 200
        assert resp.json()['price'] == 42.0
    finally:
        p.terminate()
        p.join()


def _run_mb(model_dir: str):
    os.environ['MODEL_DIR'] = model_dir
    os.environ['HOST'] = '127.0.0.1'
    from bot.services import model_builder_service
    model_builder_service.app.run(host='127.0.0.1', port=8001)


def test_model_builder_service_train_predict(tmp_path):
    p = ctx.Process(target=_run_mb, args=(str(tmp_path),))
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get('http://127.0.0.1:8001/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.post(
            'http://127.0.0.1:8001/train',
            json={'symbol': 'SYM', 'features': [[0], [1]], 'labels': [0, 1]},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.post(
            'http://127.0.0.1:8001/predict',
            json={'symbol': 'SYM', 'features': [1]},
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json()['signal'] in {'buy', 'sell'}
    finally:
        p.terminate()
        p.join()


def _run_tm():
    class DummyExchange:
        def create_order(self, symbol, typ, side, amount, params=None):
            return {'id': '1'}
    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    os.environ['HOST'] = '127.0.0.1'
    os.environ.setdefault('TRADE_RISK_USD', '10')
    from bot.services import trade_manager_service
    trade_manager_service.app.run(host='127.0.0.1', port=8002)


def test_trade_manager_service_endpoints():
    p = ctx.Process(target=_run_tm)
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get('http://127.0.0.1:8002/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.post(
            'http://127.0.0.1:8002/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.post(
            'http://127.0.0.1:8002/close_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get('http://127.0.0.1:8002/positions', timeout=5)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 2
    finally:
        p.terminate()
        p.join()


def test_trade_manager_service_price_only():
    p = ctx.Process(target=_run_tm)
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get('http://127.0.0.1:8002/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.post(
            'http://127.0.0.1:8002/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'price': 5},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get('http://127.0.0.1:8002/positions', timeout=5)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1
    finally:
        p.terminate()
        p.join()


def test_trade_manager_ready_route():
    p = ctx.Process(target=_run_tm)
    p.start()
    try:
        resp = None
        for _ in range(50):
            try:
                resp = requests.get('http://127.0.0.1:8002/ready', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        assert resp is not None and resp.status_code == 200
    finally:
        p.terminate()
        p.join()
