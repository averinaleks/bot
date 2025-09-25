import importlib
import logging
import sys
import types


def _import_service(monkeypatch):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            return {'last': 1.0}

    ccxt = types.ModuleType("ccxt")
    ccxt.bybit = lambda *args, **kwargs: DummyExchange()
    ccxt.BaseError = Exception
    ccxt.NetworkError = Exception
    monkeypatch.setitem(sys.modules, "ccxt", ccxt)

    monkeypatch.setenv("TEST_MODE", "1")

    monkeypatch.delitem(sys.modules, 'bot.services.data_handler_service', raising=False)
    monkeypatch.delitem(sys.modules, 'services.data_handler_service', raising=False)

    return importlib.import_module('bot.services.data_handler_service')


def test_price_requires_api_token(monkeypatch, caplog):
    monkeypatch.delenv('DATA_HANDLER_API_KEY', raising=False)
    monkeypatch.delenv('DATA_HANDLER_ALLOW_ANONYMOUS', raising=False)

    module = _import_service(monkeypatch)

    client = module.app.test_client()
    with caplog.at_level(logging.WARNING):
        response = client.get('/price/BTCUSDT')

    assert response.status_code == 401
    assert 'DATA_HANDLER_API_KEY' in caplog.text


def test_price_allows_anonymous_when_flag(monkeypatch, caplog):
    monkeypatch.delenv('DATA_HANDLER_API_KEY', raising=False)
    monkeypatch.setenv('DATA_HANDLER_ALLOW_ANONYMOUS', '1')

    module = _import_service(monkeypatch)

    client = module.app.test_client()
    with caplog.at_level(logging.WARNING):
        response = client.get('/price/BTCUSDT')

    assert response.status_code == 200
    assert 'DATA_HANDLER_ALLOW_ANONYMOUS' in caplog.text
