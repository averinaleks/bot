import multiprocessing
import os
import time
import types
import socket
import requests

ctx = multiprocessing.get_context("spawn")


def _get_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_dh(port: int):
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
    data_handler_service.app.run(host='127.0.0.1', port=port)


def test_data_handler_service_price():
    port = _get_free_port()
    p = ctx.Process(target=_run_dh, args=(port,))
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get(f'http://127.0.0.1:{port}/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5)
        assert resp.status_code == 200
        assert resp.json()['price'] == 42.0
    finally:
        p.terminate()
        p.join()


def _run_mb(model_dir: str, port: int):
    os.environ['MODEL_DIR'] = model_dir
    os.environ['HOST'] = '127.0.0.1'
    from bot.services import model_builder_service
    model_builder_service.app.run(host='127.0.0.1', port=port)


def test_model_builder_service_train_predict(tmp_path):
    port = _get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get(f'http://127.0.0.1:{port}/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.post(
            f'http://127.0.0.1:{port}/train',
            json={'symbol': 'SYM', 'features': [[0], [1]], 'labels': [0, 1]},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.post(
            f'http://127.0.0.1:{port}/predict',
            json={'symbol': 'SYM', 'features': [1]},
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json()['signal'] in {'buy', 'sell'}
    finally:
        p.terminate()
        p.join()


def _run_tm(port: int):
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
    trade_manager_service.app.run(host='127.0.0.1', port=port)


def test_trade_manager_service_endpoints():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get(f'http://127.0.0.1:{port}/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.post(
            f'http://127.0.0.1:{port}/close_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get(f'http://127.0.0.1:{port}/positions', timeout=5)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 2
    finally:
        p.terminate()
        p.join()


def test_trade_manager_service_price_only():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    p.start()
    try:
        for _ in range(50):
            try:
                resp = requests.get(f'http://127.0.0.1:{port}/ping', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        resp = requests.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'price': 5},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get(f'http://127.0.0.1:{port}/positions', timeout=5)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1
    finally:
        p.terminate()
        p.join()


def test_trade_manager_ready_route():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    p.start()
    try:
        resp = None
        for _ in range(50):
            try:
                resp = requests.get(f'http://127.0.0.1:{port}/ready', timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        assert resp is not None and resp.status_code == 200
    finally:
        p.terminate()
        p.join()
