import importlib
import sys
import types


def test_history_limit_is_capped_and_warns(monkeypatch):
    requested_limit = 5000
    observed_limits: list[int] = []

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

    with module.app.test_client() as client:
        response = client.get(f'/history/BTCUSDT?limit={requested_limit}')

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
