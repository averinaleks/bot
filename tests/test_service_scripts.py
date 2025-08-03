import multiprocessing
import os
import types
import socket
import requests
import pytest

from tests.helpers import wait_for_service

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


@pytest.mark.integration
def test_data_handler_service_price():
    port = _get_free_port()
    p = ctx.Process(target=_run_dh, args=(port,))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5)
        assert resp.status_code == 200
        assert resp.json()['price'] == 42.0
    finally:
        p.terminate()
        p.join()


def _run_dh_fail(port: int):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            raise RuntimeError("exchange down")

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    os.environ['STREAM_SYMBOLS'] = ''
    os.environ['HOST'] = '127.0.0.1'
    from bot.services import data_handler_service
    data_handler_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_data_handler_service_price_error():
    port = _get_free_port()
    p = ctx.Process(target=_run_dh_fail, args=(port,))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5)
        assert resp.status_code == 503
        assert 'error' in resp.json()
    finally:
        p.terminate()
        p.join()


def _run_mb(model_dir: str, port: int):
    class DummyLR:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            import numpy as np

            prob = np.zeros((X.shape[0], 2), dtype=float)
            prob[:, 1] = (X[:, 0] > 0).astype(float)
            prob[:, 0] = 1 - prob[:, 1]
            return prob

    DummyLR.__module__ = 'sklearn.linear_model'
    DummyLR.__name__ = 'LogisticRegression'
    DummyLR.__qualname__ = 'LogisticRegression'
    sklearn = types.ModuleType('sklearn')
    linear_model = types.ModuleType('sklearn.linear_model')
    linear_model.LogisticRegression = DummyLR
    sklearn.linear_model = linear_model
    import sys

    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.linear_model'] = linear_model

    os.environ['MODEL_DIR'] = model_dir
    os.environ['HOST'] = '127.0.0.1'
    from bot.services import model_builder_service
    model_builder_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_model_builder_service_train_predict(tmp_path):
    port = _get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
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


@pytest.mark.integration
def test_model_builder_service_train_predict_multi_class(tmp_path):
    port = _get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.post(
            f'http://127.0.0.1:{port}/train',
            json={'symbol': 'SYM', 'features': [[0], [1], [2]], 'labels': [0, 1, 2]},
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


@pytest.mark.integration
def test_model_builder_service_rejects_single_class_labels(tmp_path):
    port = _get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.post(
            f'http://127.0.0.1:{port}/train',
            json={'features': [[0], [1]], 'labels': [0, 0]},
            timeout=5,
        )
        assert resp.status_code == 400
    finally:
        p.terminate()
        p.join()


def _run_mb_fail(model_file: str, port: int):
    class DummyLR:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            import numpy as np

            prob = np.zeros((X.shape[0], 2), dtype=float)
            prob[:, 1] = (X[:, 0] > 0).astype(float)
            prob[:, 0] = 1 - prob[:, 1]
            return prob

    DummyLR.__module__ = 'sklearn.linear_model'
    DummyLR.__name__ = 'LogisticRegression'
    DummyLR.__qualname__ = 'LogisticRegression'
    sklearn = types.ModuleType('sklearn')
    linear_model = types.ModuleType('sklearn.linear_model')
    linear_model.LogisticRegression = DummyLR
    sklearn.linear_model = linear_model
    import sys

    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.linear_model'] = linear_model

    os.environ['MODEL_FILE'] = model_file
    os.environ['HOST'] = '127.0.0.1'
    from bot.services import model_builder_service
    model_builder_service._load_model()
    model_builder_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_model_builder_service_load_failure(tmp_path):
    port = _get_free_port()
    bad_file = tmp_path / 'model.pkl'
    bad_file.write_text('broken')
    p = ctx.Process(target=_run_mb_fail, args=(str(bad_file), port))
    p.start()
    try:
        resp = wait_for_service(f'http://127.0.0.1:{port}/ping')
        assert resp.status_code == 200
    finally:
        p.terminate()
        p.join()


def _run_tm(port: int, with_tp_sl: bool = True, fail_after_market: bool = False):
    class DummyExchange:
        def __init__(self):
            self.calls = 0

        def create_order(self, symbol, typ, side, amount, price=None, params=None):
            self.calls += 1
            if fail_after_market and self.calls > 1:
                return None
            return {'id': str(self.calls), 'type': typ, 'side': side, 'price': price}

        if with_tp_sl:
            def create_order_with_take_profit_and_stop_loss(
                self, symbol, typ, side, amount, price, tp, sl, params=None
            ):
                self.calls += 1
                if fail_after_market:
                    return None
                return {'id': 'tp-sl', 'tp': tp, 'sl': sl}

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    os.environ['HOST'] = '127.0.0.1'
    os.environ.setdefault('TRADE_RISK_USD', '10')
    from bot.services import trade_manager_service
    trade_manager_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_trade_manager_service_endpoints():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get(f'http://127.0.0.1:{port}/positions', timeout=5)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1
        resp = requests.post(
            f'http://127.0.0.1:{port}/close_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1},
            timeout=5,
        )
        assert resp.status_code == 200
        resp = requests.get(f'http://127.0.0.1:{port}/positions', timeout=5)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 0
    finally:
        p.terminate()
        p.join()


@pytest.mark.integration
def test_trade_manager_service_price_only():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
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


@pytest.mark.integration
def test_trade_manager_service_fallback_orders():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port, False))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5},
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


@pytest.mark.integration
def test_trade_manager_service_fallback_failure():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port, False, True))
    p.start()
    try:
        wait_for_service(f'http://127.0.0.1:{port}/ping')
        resp = requests.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5},
            timeout=5,
        )
        assert resp.status_code == 500
    finally:
        p.terminate()
        p.join()


@pytest.mark.integration
def test_trade_manager_ready_route():
    port = _get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    p.start()
    try:
        resp = wait_for_service(f'http://127.0.0.1:{port}/ready')
        assert resp.status_code == 200
    finally:
        p.terminate()
        p.join()
