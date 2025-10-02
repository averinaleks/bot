import json
import logging
import threading
import time

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


def test_open_position_missing_symbol_returns_code(monkeypatch, caplog):
    from services import trade_manager_service as tms

    class DummyExchange:
        pass

    tms.exchange_provider.override(DummyExchange())

    with tms.app.test_client() as client, caplog.at_level(logging.WARNING):
        response = _post_open_position(client, {'side': 'buy', 'price': 100})

    assert response.status_code == 400
    assert response.get_json() == {'error': 'symbol is required', 'code': 'missing_symbol'}
    messages = [record.getMessage() for record in caplog.records]
    assert any('open_position_error[missing_symbol]' in message for message in messages)


def test_open_position_invalid_price_returns_code(monkeypatch, caplog):
    from services import trade_manager_service as tms

    class DummyExchange:
        pass

    tms.exchange_provider.override(DummyExchange())

    with tms.app.test_client() as client, caplog.at_level(logging.WARNING):
        response = _post_open_position(
            client,
            {'symbol': 'BTCUSDT', 'side': 'buy', 'price': 'oops'},
        )

    assert response.status_code == 400
    assert response.get_json() == {'error': 'invalid price', 'code': 'invalid_price'}
    messages = [record.getMessage() for record in caplog.records]
    assert any('open_position_error[invalid_price]' in message for message in messages)
    assert any('BTCUSDT' in message for message in messages)


def test_open_position_negative_risk_returns_code(monkeypatch, caplog):
    from services import trade_manager_service as tms

    class DummyExchange:
        pass

    tms.exchange_provider.override(DummyExchange())
    monkeypatch.setenv('TRADE_RISK_USD', '-5')

    with tms.app.test_client() as client, caplog.at_level(logging.WARNING):
        response = _post_open_position(
            client,
            {'symbol': 'BTCUSDT', 'side': 'buy', 'price': 100, 'amount': 0},
        )

    assert response.status_code == 400
    assert response.get_json() == {'error': 'risk must be positive', 'code': 'negative_risk'}
    messages = [record.getMessage() for record in caplog.records]
    assert any('open_position_error[negative_risk]' in message for message in messages)


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


def test_open_position_warns_when_positions_cache_fails(monkeypatch, caplog):
    from services import trade_manager_service as tms

    class Exchange:
        def create_order(self, symbol, typ, side, amount, price=None, params=None):
            return {'id': 'primary'}

    tms.exchange_provider.override(Exchange())

    def failing_replace(*_args, **_kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(tms.os, 'replace', failing_replace)

    with tms.app.test_client() as client, caplog.at_level(logging.WARNING):
        response = _post_open_position(
            client,
            {'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'price': 100},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    warning = payload['warnings']['positions_cache_failed']
    assert warning['message'] == 'не удалось обновить кэш позиций'
    assert 'details' in warning
    assert len(tms.POSITIONS) == 1
    assert any('не удалось обновить кэш позиций' in record.getMessage() for record in caplog.records)


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


def test_close_position_warns_when_positions_cache_fails(monkeypatch, caplog):
    from services import trade_manager_service as tms

    class Exchange:
        def create_order(self, symbol, typ, side, amount, price=None, params=None):
            return {'id': 'close'}

    tms.exchange_provider.override(Exchange())
    tms.POSITIONS.append(
        {'id': 'open-order', 'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 1, 'action': 'open'}
    )

    def failing_dump(*_args, **_kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(tms.json, 'dump', failing_dump)

    with tms.app.test_client() as client, caplog.at_level(logging.WARNING):
        response = client.post(
            '/close_position',
            json={'order_id': 'open-order', 'side': 'sell'},
            headers={'Authorization': 'Bearer test-token'},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    warning = payload['warnings']['positions_cache_failed']
    assert warning['message'] == 'не удалось обновить кэш позиций'
    assert 'details' in warning
    assert not tms.POSITIONS
    assert any('не удалось обновить кэш позиций' in record.getMessage() for record in caplog.records)


def test_exchange_calls_are_serialized(monkeypatch):
    from services import trade_manager_service as tms

    call_sequence: list[int] = []
    in_call = threading.Event()
    first_entered = threading.Event()
    release_first = threading.Event()

    event_timeout = 5.0

    def create_order(*_args, **_kwargs):
        assert not in_call.is_set(), 'create_order re-entered concurrently'
        in_call.set()
        try:
            if not first_entered.is_set():
                first_entered.set()
                assert release_first.wait(timeout=event_timeout), 'release not signalled'
            time.sleep(0.01)
            call_sequence.append(len(call_sequence) + 1)
            return {'id': call_sequence[-1]}
        finally:
            in_call.clear()

    exchange = type('Exchange', (), {'create_order': create_order})()

    results: list[dict[str, int]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result = tms._call_exchange_method(
                exchange, 'create_order', 'BTCUSDT', 'market', 'buy', 1.0
            )
            results.append(result)
        except BaseException as exc:  # pragma: no cover - diagnostic
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)

    first.start()
    assert first_entered.wait(timeout=event_timeout)
    second.start()
    time.sleep(0.05)
    release_first.set()
    first.join()
    second.join()

    assert not errors
    assert [result['id'] for result in results] == [1, 2]
    assert call_sequence == [1, 2]


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
