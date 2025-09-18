import multiprocessing
import os
import types
import httpx
import pytest
from pathlib import Path
from unittest.mock import patch

from tests.helpers import get_free_port, service_process

pytestmark = pytest.mark.integration

TOKEN_HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture
def ctx():
    return multiprocessing.get_context("spawn")


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
    with patch.dict(os.environ, {'STREAM_SYMBOLS': '', 'HOST': '127.0.0.1', 'TEST_MODE': '1'}):
        from bot.services import data_handler_service
        data_handler_service.app.run(host=data_handler_service.get_bind_host(), port=port)


@pytest.mark.integration
def test_data_handler_service_price(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_dh, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5, trust_env=False)
        assert resp.status_code == 200
        assert resp.json()['price'] == 42.0


@pytest.mark.integration
def test_data_handler_service_history(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_dh, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.get(
            f'http://127.0.0.1:{port}/history/BTCUSDT', timeout=5, trust_env=False
        )
        assert resp.status_code == 200
        assert resp.json()['history'] == [[1, 1, 1, 1, 1, 1]]


def _run_dh_fail(port: int):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            raise RuntimeError("exchange down")

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys
    sys.modules['ccxt'] = ccxt
    with patch.dict(os.environ, {'STREAM_SYMBOLS': '', 'HOST': '127.0.0.1', 'TEST_MODE': '1'}):
        from bot.services import data_handler_service
        data_handler_service.app.run(host=data_handler_service.get_bind_host(), port=port)


def _run_dh_token(port: int):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            return {'last': 42.0}

        def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
            return [[1, 1, 1, 1, 1, 1]]

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys

    sys.modules['ccxt'] = ccxt
    with patch.dict(
        os.environ,
        {'STREAM_SYMBOLS': '', 'HOST': '127.0.0.1', 'TEST_MODE': '1', 'DATA_HANDLER_API_KEY': 'secret'},
    ):
        from bot.services import data_handler_service

        data_handler_service.app.run(host=data_handler_service.get_bind_host(), port=port)


@pytest.mark.integration
def test_data_handler_service_price_error(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_dh_fail, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5, trust_env=False)
        assert resp.status_code == 503
        assert 'error' in resp.json()


def test_data_handler_service_rejects_non_local_host(monkeypatch):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            return {'last': 42.0}

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: DummyExchange()
    import sys

    monkeypatch.setitem(sys.modules, 'ccxt', ccxt)
    monkeypatch.setenv('TEST_MODE', '1')
    monkeypatch.setenv('HOST', '8.8.8.8')
    monkeypatch.delitem(sys.modules, 'bot.services.data_handler_service', raising=False)
    import importlib

    module = importlib.import_module('bot.services.data_handler_service')

    with pytest.raises(ValueError):
        module.get_bind_host()


@pytest.mark.integration
def test_data_handler_service_requires_api_key(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_dh_token, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.get(f'http://127.0.0.1:{port}/price/BTCUSDT', timeout=5, trust_env=False)
        assert resp.status_code == 401
        assert resp.json()['error'] == 'unauthorized'

        resp = httpx.get(
            f'http://127.0.0.1:{port}/price/BTCUSDT',
            timeout=5,
            trust_env=False,
            headers={'X-API-KEY': 'secret'},
        )
        assert resp.status_code == 200
        assert resp.json()['price'] == 42.0

        resp = httpx.get(
            f'http://127.0.0.1:{port}/history/BTCUSDT',
            timeout=5,
            trust_env=False,
            headers={'X-API-KEY': 'secret'},
        )
        assert resp.status_code == 200
        assert resp.json()['history'] == [[1, 1, 1, 1, 1, 1]]


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

    with patch.dict(os.environ, {'MODEL_DIR': model_dir, 'TEST_MODE': '1'}):
        from bot.services import model_builder_service
        model_builder_service.app.run(port=port)


@pytest.mark.integration
def test_model_builder_service_train_predict(tmp_path, ctx):
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
def test_model_builder_service_train_predict_multi_class(tmp_path, ctx):
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
def test_model_builder_service_rejects_single_class_labels(tmp_path, ctx):
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

    with patch.dict(os.environ, {'MODEL_FILE': model_file, 'TEST_MODE': '1'}):
        from bot.services import model_builder_service
        model_builder_service._load_model()
        model_builder_service.app.run(port=port)


@pytest.mark.integration
def test_model_builder_service_load_failure(tmp_path, ctx):
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
    positions_file = Path('cache/positions.json')
    try:
        positions_file.unlink()
    except FileNotFoundError:
        pass
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
        'TEST_MODE': '1',
    }
    with patch.dict(os.environ, env):
        from bot.services import trade_manager_service
        trade_manager_service.app.run(host='127.0.0.1', port=port)


@pytest.mark.integration
def test_trade_manager_service_endpoints(ctx):
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
def test_trade_manager_service_partial_close(ctx):
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
def test_trade_manager_service_price_only(ctx):
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
def test_trade_manager_service_fallback_orders(ctx):
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
def test_trade_manager_service_fallback_failure(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port, False, True))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            json={'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'tp': 10, 'sl': 5},
            timeout=5, trust_env=False,
            headers=TOKEN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert 'warnings' in data
        assert data['warnings']['protective_orders_failed']


@pytest.mark.integration
def test_trade_manager_service_invalid_json(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ping'):
        resp = httpx.post(
            f'http://127.0.0.1:{port}/open_position',
            content='{',
            timeout=5,
            trust_env=False,
            headers={**TOKEN_HEADERS, 'Content-Type': 'application/json'},
        )
        assert resp.status_code == 400
        assert resp.json() == {'error': 'invalid json'}


@pytest.mark.integration
def test_trade_manager_ready_route(ctx):
    port = get_free_port()
    p = ctx.Process(target=_run_tm, args=(port,))
    with service_process(p, url=f'http://127.0.0.1:{port}/ready') as resp:
        assert resp.status_code == 200
