import multiprocessing
import os
import types
import httpx
import pytest
from unittest.mock import patch

from tests.helpers import get_free_port, service_process

pytestmark = pytest.mark.integration

TOKEN_HEADERS = {"Authorization": "Bearer test-token"}

ctx = multiprocessing.get_context("spawn")


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
    with patch.dict(os.environ, {'STREAM_SYMBOLS': '', 'HOST': '127.0.0.1'}):
        from bot.services import data_handler_service
        data_handler_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_data_handler_service_price():
    port = get_free_port()
    p = ctx.Process(target=_run_dh, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5, trust_env=False)
        assert resp.status_code == 200
        assert resp.json()['price'] == 42.0


def _run_dh_fail(port: int):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            raise RuntimeError("exchange down")

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    with patch.dict(os.environ, {'STREAM_SYMBOLS': '', 'HOST': '127.0.0.1'}):
        from bot.services import data_handler_service
        data_handler_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_data_handler_service_price_error():
    port = get_free_port()
    p = ctx.Process(target=_run_dh_fail, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5, trust_env=False)
        assert resp.status_code == 503
        assert 'error' in resp.json()


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

    with patch.dict(os.environ, {'MODEL_DIR': model_dir}):
        from bot.services import model_builder_service
        model_builder_service.app.run(port=port)


@pytest.mark.integration
def test_model_builder_service_train_predict(tmp_path):
    port = get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/train',
            json={'symbol': 'SYM', 'features': [[0], [1]], 'labels': [0, 1]},
            timeout=5, trust_env=False,
        )
        assert resp.status_code == 200
        resp = httpx.post(
            f'http://127.0.0.1:{port}/predict',
            json={'symbol': 'SYM', 'features': [1]},
            timeout=5, trust_env=False,
        )
        assert resp.status_code == 200
        assert resp.json()['signal'] in {'buy', 'sell'}


@pytest.mark.integration
def test_model_builder_service_train_predict_multi_class(tmp_path):
    port = get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/train',
            json={'symbol': 'SYM', 'features': [[0], [1], [2]], 'labels': [0, 1, 2]},
            timeout=5, trust_env=False,
        )
        assert resp.status_code == 200
        resp = httpx.post(
            f'http://127.0.0.1:{port}/predict',
            json={'symbol': 'SYM', 'features': [1]},
            timeout=5, trust_env=False,
        )
        assert resp.status_code == 200
        assert resp.json()['signal'] in {'buy', 'sell'}


@pytest.mark.integration
def test_model_builder_service_rejects_single_class_labels(tmp_path):
    port = get_free_port()
    p = ctx.Process(target=_run_mb, args=(str(tmp_path), port))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/train',
            json={'features': [[0], [1]], 'labels': [0, 0]},
            timeout=5, trust_env=False,
        )
        assert resp.status_code == 400


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

    with patch.dict(os.environ, {'MODEL_FILE': model_file}):
        from bot.services import model_builder_service
        model_builder_service._load_model()
        model_builder_service.app.run(port=port)


@pytest.mark.integration
def test_model_builder_service_load_failure(tmp_path):
    port = get_free_port()
    bad_file = tmp_path / 'model.pkl'
    bad_file.write_text('broken')
    p = ctx.Process(target=_run_mb_fail, args=(str(bad_file), port))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping') as resp:
        assert resp.status_code == 200


def _run_tm(
    port: int,
    with_tp_sl: bool = True,
    fail_after_market: bool = False,
    with_trailing: bool = True,
):
    class DummyExchange:
        def __init__(self):
            self.calls = 0

        def create_order(self, symbol, typ, side, amount, price=None, params=None):
            self.calls += 1
            if fail_after_market and self.calls > 1:
                return None
            return {
                'id': str(self.calls),
                'type': typ,
                'side': side,
                'price': price,
            }

        if with_tp_sl:
            def create_order_with_take_profit_and_stop_loss(
                self, symbol, typ, side, amount, price, tp, sl, params=None
            ):
                self.calls += 1
                if fail_after_market:
                    return None
                return {'id': 'tp-sl', 'tp': tp, 'sl': sl}

        if with_trailing:
            def create_order_with_trailing_stop(
                self, symbol, typ, side, amount, price, trailing, params=None
            ):
                self.calls += 1
                if fail_after_market:
                    return None
                return {'id': 'trailing', 'trailing': trailing}

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    env = {
        'HOST': '127.0.0.1',
        'TRADE_MANAGER_TOKEN': 'test-token',
        'TRADE_RISK_USD': os.environ.get('TRADE_RISK_USD', '10'),
    }
    with patch.dict(os.environ, env):
        from bot.services import trade_manager_service
        trade_manager_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_trade_manager_service_endpoints():
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5, 'trailing_stop': 1},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        order_id = resp.json()['order_id']
        resp = httpx.get(f'http://127.0.0.1:{port}/positions', timeout=5, trust_env=False, headers=TOKEN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1
        assert data[0]['trailing_stop'] == 1
        resp = httpx.post(
            f'http://127.0.0.1:{port}/close_position',
            json={'order_id': order_id, 'side': 'sell'},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        resp = httpx.get(f'http://127.0.0.1:{port}/positions', timeout=5, trust_env=False, headers=TOKEN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 0


@pytest.mark.integration
def test_trade_manager_service_partial_close():
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        order_id = resp.json()['order_id']
        # close half the position
        resp = httpx.post(
            f'http://127.0.0.1:{port}/close_position',
            json={'order_id': order_id, 'side': 'sell', 'close_amount': 0.4},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        resp = httpx.get(f'http://127.0.0.1:{port}/positions', timeout=5, trust_env=False, headers=TOKEN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1
        assert data[0]['amount'] == pytest.approx(0.6, rel=1e-3)
        # close the remainder
        resp = httpx.post(
            f'http://127.0.0.1:{port}/close_position',
            json={'order_id': order_id, 'side': 'sell'},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        resp = httpx.get(f'http://127.0.0.1:{port}/positions', timeout=5, trust_env=False, headers=TOKEN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 0


@pytest.mark.integration
def test_trade_manager_service_price_only():
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'price': 5},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        resp = httpx.get(f'http://127.0.0.1:{port}/positions', timeout=5, trust_env=False, headers=TOKEN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1


@pytest.mark.integration
def test_trade_manager_service_fallback_orders():
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port, False, False, False))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5, 'price': 100, 'trailing_stop': 1},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        resp = httpx.get(f'http://127.0.0.1:{port}/positions', timeout=5, trust_env=False, headers=TOKEN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()['positions']
        assert len(data) == 1
        assert data[0]['trailing_stop'] == 1


@pytest.mark.integration
def test_trade_manager_service_fallback_failure():
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port, False, True))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 500


@pytest.mark.integration
def test_trade_manager_ready_route():
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ready') as resp:
        assert resp.status_code == 200
