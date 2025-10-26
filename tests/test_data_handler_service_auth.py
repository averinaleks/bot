import importlib
import logging
import sys
import types

import pytest


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
        response = client.get('/price/BTC/USDT')

    assert response.status_code == 401
    assert 'DATA_HANDLER_API_KEY' in caplog.text


def test_price_allows_anonymous_when_flag(monkeypatch, caplog):
    monkeypatch.delenv('DATA_HANDLER_API_KEY', raising=False)
    monkeypatch.setenv('DATA_HANDLER_ALLOW_ANONYMOUS', '1')

    module = _import_service(monkeypatch)

    client = module.app.test_client()
    with caplog.at_level(logging.WARNING):
        response = client.get('/price/BTC/USDT')

    assert response.status_code == 200
    assert 'DATA_HANDLER_ALLOW_ANONYMOUS' in caplog.text


def test_price_rejects_invalid_symbol_format(monkeypatch):
    monkeypatch.delenv('DATA_HANDLER_API_KEY', raising=False)
    monkeypatch.setenv('DATA_HANDLER_ALLOW_ANONYMOUS', '1')

    module = _import_service(monkeypatch)

    class DummyProvider:
        def create(self):  # pragma: no cover - defensive fail-safe
            pytest.fail('exchange should not be created for invalid symbol')

        def close_instance(self, exchange):  # pragma: no cover - defensive no-op
            pass

    module.exchange_provider = DummyProvider()
    client = module.app.test_client()
    resp = client.get('/price/BTCUSDT')

    assert resp.status_code == 400
    assert resp.get_json() == {'error': 'invalid symbol format'}


def test_history_rejects_invalid_symbol_format(monkeypatch):
    monkeypatch.delenv('DATA_HANDLER_API_KEY', raising=False)
    monkeypatch.setenv('DATA_HANDLER_ALLOW_ANONYMOUS', '1')

    module = _import_service(monkeypatch)

    class DummyProvider:
        def create(self):  # pragma: no cover - defensive fail-safe
            pytest.fail('exchange should not be created for invalid symbol')

        def close_instance(self, exchange):  # pragma: no cover - defensive no-op
            pass

    module.exchange_provider = DummyProvider()
    client = module.app.test_client()
    resp = client.get('/history/BTCUSDT')

    assert resp.status_code == 400
    assert resp.get_json() == {'error': 'invalid symbol format'}
