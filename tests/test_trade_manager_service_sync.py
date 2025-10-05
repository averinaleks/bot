import importlib
import json
import secrets
import sys
import types


def _reload_service(monkeypatch, tmp_path, exchange):
    fake_ccxt = types.SimpleNamespace(
        bybit=lambda *a, **k: exchange,
        BaseError=Exception,
        NetworkError=Exception,
        BadRequest=Exception,
    )
    monkeypatch.setitem(sys.modules, 'ccxt', fake_ccxt)
    monkeypatch.setenv('TRADE_MANAGER_TOKEN', 'token')
    service = importlib.reload(importlib.import_module('services.trade_manager_service'))
    service.POSITIONS_FILE = tmp_path / 'positions.json'
    service.POSITIONS[:] = []
    service.exchange_provider.override(exchange)
    # Bandit note - a placeholder token is used solely for fixture initialisation.
    service.API_TOKEN = secrets.token_hex(8)
    return service


def test_sync_removes_closed_positions(monkeypatch, tmp_path):
    class DummyExchange:
        def __init__(self):
            self.calls = 0

        def fetch_positions(self):
            self.calls += 1
            return [
                {'id': '2', 'symbol': 'ETHUSDT', 'side': 'sell', 'contracts': 2},
            ]

    exchange = DummyExchange()
    service = _reload_service(monkeypatch, tmp_path, exchange)
    service.POSITIONS[:] = [
        {'id': '1', 'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'action': 'open'},
        {'id': '2', 'symbol': 'ETHUSDT', 'side': 'sell', 'amount': 2, 'action': 'open'},
    ]
    service._save_positions()

    service._sync_positions_with_exchange(exchange)

    assert [pos['id'] for pos in service.POSITIONS] == ['2']
    with service.POSITIONS_FILE.open('r', encoding='utf-8') as fh:
        cached = json.load(fh)
    assert [pos['id'] for pos in cached] == ['2']


def test_positions_endpoint_triggers_sync(monkeypatch, tmp_path):
    class DummyExchange:
        def __init__(self):
            self.calls = 0

        def fetch_positions(self):
            self.calls += 1
            return []

    exchange = DummyExchange()
    service = _reload_service(monkeypatch, tmp_path, exchange)
    service.POSITIONS[:] = [
        {'id': '1', 'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'action': 'open'},
    ]
    response = service.app.test_client().get(
        '/positions',
        headers={'Authorization': 'Bearer token'},
    )

    assert response.status_code == 200
    assert response.json['positions'] == []
    assert exchange.calls == 1


def test_positions_endpoint_respects_rotated_token(monkeypatch, tmp_path):
    class DummyExchange:
        def __init__(self):
            self.calls = 0

        def fetch_positions(self):
            self.calls += 1
            return []

    exchange = DummyExchange()
    service = _reload_service(monkeypatch, tmp_path, exchange)
    client = service.app.test_client()

    first = client.get('/positions', headers={'Authorization': 'Bearer token'})
    assert first.status_code == 200
    assert exchange.calls == 1

    rotated = secrets.token_hex(8)
    monkeypatch.setenv('TRADE_MANAGER_TOKEN', rotated)

    second = client.get('/positions', headers={'Authorization': f'Bearer {rotated}'})
    assert second.status_code == 200
    assert exchange.calls == 2

    rejected = client.get('/positions', headers={'Authorization': 'Bearer token'})
    assert rejected.status_code == 401
