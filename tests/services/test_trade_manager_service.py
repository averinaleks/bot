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
    return module


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
