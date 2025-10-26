import importlib
import sys
import types
from urllib.parse import quote


def _load_data_handler_module(monkeypatch, observed_limits):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            return {'last': 1.0}

        def fetch_ohlcv(self, symbol, timeframe='1m', limit=200):
            observed_limits.append(limit)
            return [[0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *args, **kwargs: DummyExchange()
    ccxt.BaseError = Exception
    ccxt.NetworkError = Exception
    monkeypatch.setitem(sys.modules, 'ccxt', ccxt)

    monkeypatch.setenv('TEST_MODE', '1')
    monkeypatch.setenv('DATA_HANDLER_ALLOW_ANONYMOUS', '1')
    monkeypatch.delenv('DATA_HANDLER_API_KEY', raising=False)

    monkeypatch.delitem(sys.modules, 'bot.services.data_handler_service', raising=False)
    monkeypatch.delitem(sys.modules, 'services.data_handler_service', raising=False)

    module = importlib.import_module('bot.services.data_handler_service')
    module.app.testing = True
    return module


def test_history_limit_is_capped_and_warns(monkeypatch):
    requested_limit = 5000
    observed_limits: list[int] = []
    symbol = 'BTC/USDT'
    encoded_symbol = quote(symbol, safe='')

    module = _load_data_handler_module(monkeypatch, observed_limits)

    with module.app.test_client() as client:
        response = client.get(f'/history/{encoded_symbol}?limit={requested_limit}')

    assert response.status_code == 200

    payload = response.get_json()
    assert payload is not None
    assert 'history' in payload
    assert 'warnings' in payload
    assert 'limit' in payload['warnings']
    warning = payload['warnings']['limit']
    assert warning['requested'] == requested_limit
    assert warning['applied'] == module.MAX_HISTORY_LIMIT
    assert observed_limits == [module.MAX_HISTORY_LIMIT]


def test_history_limit_zero_is_raised_to_minimum(monkeypatch):
    requested_limit = 0
    observed_limits: list[int] = []

    symbol = 'BTC/USDT'
    encoded_symbol = quote(symbol, safe='')

    module = _load_data_handler_module(monkeypatch, observed_limits)

    with module.app.test_client() as client:
        response = client.get(f'/history/{encoded_symbol}?limit={requested_limit}')

    assert response.status_code == 200

    payload = response.get_json()
    assert payload is not None
    assert 'history' in payload
    assert 'warnings' in payload
    assert 'limit' in payload['warnings']
    warning = payload['warnings']['limit']
    assert warning['requested'] == requested_limit
    assert warning['applied'] == module.MIN_HISTORY_LIMIT
    assert warning['message'] == f'limit raised to minimum {module.MIN_HISTORY_LIMIT}'
    assert observed_limits == [module.MIN_HISTORY_LIMIT]


def test_history_limit_negative_is_raised_to_minimum(monkeypatch):
    requested_limit = -100
    observed_limits: list[int] = []

    symbol = 'BTC/USDT'
    encoded_symbol = quote(symbol, safe='')

    module = _load_data_handler_module(monkeypatch, observed_limits)

    with module.app.test_client() as client:
        response = client.get(f'/history/{encoded_symbol}?limit={requested_limit}')

    assert response.status_code == 200

    payload = response.get_json()
    assert payload is not None
    assert 'history' in payload
    assert 'warnings' in payload
    assert 'limit' in payload['warnings']
    warning = payload['warnings']['limit']
    assert warning['requested'] == requested_limit
    assert warning['applied'] == module.MIN_HISTORY_LIMIT
    assert warning['message'] == f'limit raised to minimum {module.MIN_HISTORY_LIMIT}'
    assert observed_limits == [module.MIN_HISTORY_LIMIT]
