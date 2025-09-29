import json
import logging

import pytest

from tests import test_trade_manager_routes as tm_routes


@pytest.fixture(autouse=True)
def _reset_positions(tmp_path, monkeypatch):
    from services import trade_manager_service as tms

    cache_file = tmp_path / 'positions.json'
    monkeypatch.setattr(tms, 'POSITIONS_FILE', cache_file)
    monkeypatch.setattr(tms, 'POSITIONS', [])
    monkeypatch.setattr(tms, 'API_TOKEN', 'test-token')
    tms.exchange_provider.override(None)
    yield
    tms.exchange_provider.close()


def _post_open_position(client, payload):
    return client.post(
        '/open_position',
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer test-token'},
    )


def test_post_without_token_rejected_when_not_configured(monkeypatch, caplog):
    from services import trade_manager_service as tms

    monkeypatch.setattr(tms, 'API_TOKEN', '')

    with tms.app.test_client() as client, caplog.at_level(logging.WARNING):
        response = client.post('/open_position', json={'symbol': 'BTCUSDT'})

    assert response.status_code == 401
    assert response.get_json() == {'error': 'unauthorized'}
    assert any(
        'API token is not configured' in message
        for message in (record.getMessage() for record in caplog.records)
    )


def test_open_position_records_even_when_stop_loss_fails(monkeypatch):
    from services import trade_manager_service as tms

    class ExchangeWithCancel:
        def __init__(self):
            self.cancelled = []
            self.created = []

        def create_order(self, symbol, typ, side, amount, price=None, params=None):
            self.created.append({'type': typ, 'params': params})
            if typ == 'market' and params is None:
                return {'id': 'primary'}
            if typ in {'stop', 'stop_market'}:
                return None
            if typ == 'limit':
                return {'id': 'tp'}
            if params and params.get('reduce_only'):
                return {'id': 'close'}
            return {'id': 'other'}

        def cancel_order(self, order_id, symbol):
            self.cancelled.append((order_id, symbol))
            return {'id': order_id}

    exchange = ExchangeWithCancel()
    tms.exchange_provider.override(exchange)

    with tms.app.test_client() as client:
        response = _post_open_position(
            client,
            {'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'sl': 5, 'tp': 10, 'price': 100},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert 'warnings' in payload
    warning = payload['warnings']
    assert warning['protective_orders_failed'][0]['type'] == 'stop_loss'
    assert 'primary_order_cancelled' in warning.get('mitigations', [])
    assert exchange.cancelled == [('primary', 'BTCUSDT')]

    from services import trade_manager_service as tms_reload

    assert len(tms_reload.POSITIONS) == 1


def test_open_position_emergency_close_when_cancel_unavailable(monkeypatch):
    from services import trade_manager_service as tms

    class ExchangeWithoutCancel:
        def __init__(self):
            self.created = []

        def create_order(self, symbol, typ, side, amount, price=None, params=None):
            self.created.append({'type': typ, 'params': params})
            if typ == 'market' and params is None:
                return {'id': 'primary'}
            if typ in {'stop', 'stop_market'}:
                return None
            if typ == 'limit':
                return {'id': 'tp'}
            if params and params.get('reduce_only'):
                return {'id': 'close'}
            return {'id': 'other'}

    exchange = ExchangeWithoutCancel()
    tms.exchange_provider.override(exchange)

    with tms.app.test_client() as client:
        response = _post_open_position(
            client,
            {'symbol': 'ETHUSDT', 'side': 'sell', 'amount': 2, 'sl': 15, 'tp': 25, 'price': 50},
        )

    assert response.status_code == 200
    payload = response.get_json()
    warning = payload['warnings']
    assert any(item['type'] == 'stop_loss' for item in warning['protective_orders_failed'])
    assert 'emergency_close_submitted' in warning.get('mitigations', [])
    assert any(
        entry['params'] and entry['params'].get('reduce_only')
        for entry in exchange.created
    )

    from services import trade_manager_service as tms_reload

    assert len(tms_reload.POSITIONS) == 1


def _setup_trade_manager(monkeypatch):
    tm, loop, stub = tm_routes._setup_module(monkeypatch)
    stub._positions_data = []
    return tm, loop, stub


def test_open_position_route_rejects_missing_symbol(monkeypatch):
    tm, loop, _ = _setup_trade_manager(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.post(
        "/open_position",
        json={"side": "buy", "price": 100.0},
    )
    assert resp.status_code == 400
    assert "symbol" in resp.json["error"]
    assert not loop.calls


def test_open_position_route_rejects_invalid_side(monkeypatch):
    tm, loop, _ = _setup_trade_manager(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.post(
        "/open_position",
        json={"symbol": "BTCUSDT", "side": "hold", "price": 100.0},
    )
    assert resp.status_code == 400
    assert "side" in resp.json["error"]
    assert not loop.calls


def test_open_position_route_rejects_invalid_price(monkeypatch):
    tm, loop, _ = _setup_trade_manager(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.post(
        "/open_position",
        json={"symbol": "BTCUSDT", "side": "buy", "price": -1},
    )
    assert resp.status_code == 400
    assert "price" in resp.json["error"]
    assert not loop.calls


def test_close_position_route_rejects_invalid_price(monkeypatch):
    tm, loop, _ = _setup_trade_manager(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.post(
        "/close_position",
        json={"symbol": "BTCUSDT", "price": "nan"},
    )
    assert resp.status_code == 400
    assert "price" in resp.json["error"]
    assert not loop.calls
