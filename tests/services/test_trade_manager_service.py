import importlib

import pytest


@pytest.fixture
def trade_manager_service(monkeypatch):
    module = importlib.reload(importlib.import_module('services.trade_manager_service'))
    monkeypatch.setattr(module, 'API_TOKEN', 'test-token')
    monkeypatch.setenv('TRADE_MANAGER_TOKEN', 'test-token')
    monkeypatch.delenv('OFFLINE_MODE', raising=False)
    monkeypatch.delenv('TRADE_MANAGER_USE_STUB', raising=False)
    monkeypatch.setattr(module, 'POSITIONS', [])
    module._reset_exchange_executor()
    try:
        yield module
    finally:
        module._reset_exchange_executor()


def test_positions_requires_token(trade_manager_service):
    with trade_manager_service.app.test_client() as client:
        response = client.get('/positions')

    assert response.status_code == 401
    assert response.get_json() == {'error': 'unauthorized'}


def test_positions_allows_requests_with_valid_token(trade_manager_service):
    with trade_manager_service.app.test_client() as client:
        response = client.get(
            '/positions', headers={'Authorization': 'Bearer test-token'}
        )

    assert response.status_code == 200
    assert response.get_json() == {'positions': []}


def test_close_position_auto_inverts_side_when_not_opposite(trade_manager_service, monkeypatch):
    module = trade_manager_service
    state = module._get_state()
    state.replace_positions([
        {
            'id': 'order-1',
            'action': 'open',
            'symbol': 'BTC/USDT',
            'amount': 1.5,
            'side': 'buy',
        }
    ])

    class DummyExchange:
        def __init__(self):
            self.calls: list[tuple[str, str, str, float, dict[str, object] | None]] = []

        def create_order(self, symbol, order_type, side, amount, params=None):
            self.calls.append((symbol, order_type, side, amount, params))
            return {'id': 'close-1'}

    exchange = DummyExchange()
    state.set_exchange(exchange)
    monkeypatch.setattr(module, '_exchange_runtime', None)
    monkeypatch.setattr(module, '_write_positions_locked', lambda: None)

    with module.app.test_client() as client:
        response = client.post(
            '/close_position',
            json={'order_id': 'order-1', 'side': 'buy'},
            headers={'Authorization': 'Bearer test-token'},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        'status': 'ok',
        'order_id': 'close-1',
        'warnings': {
            'side_adjusted': {
                'message': 'направление заявки скорректировано для закрытия позиции',
                'requested': 'buy',
                'used': 'sell',
            }
        },
    }
    assert exchange.calls == [('BTC/USDT', 'market', 'sell', 1.5, {'reduce_only': True})]

    state.replace_positions([])
