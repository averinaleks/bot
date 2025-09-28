import pytest

from tests.test_trade_manager_routes import _setup_module


def test_positions_route_returns_positions(monkeypatch):
    tm, _, stub = _setup_module(monkeypatch)
    expected = [
        {
            "symbol": "ETHUSDT",
            "timestamp": "2024-02-02T00:00:00+00:00",
            "position": 1.5,
        }
    ]
    stub.get_positions = lambda: expected
    client = tm.api_app.test_client()
    resp = client.get("/positions")
    assert resp.status_code == 200
    assert resp.json["positions"] == expected


def test_positions_route_returns_not_ready(monkeypatch):
    tm, _, _ = _setup_module(monkeypatch)
    tm.trade_manager_factory.reset()
    tm._ready_event.clear()
    client = tm.api_app.test_client()
    resp = client.get("/positions")
    assert resp.status_code == 503
    assert resp.json == {"error": "not ready"}


def test_positions_route_handles_manager_error(monkeypatch):
    tm, _, stub = _setup_module(monkeypatch)

    def _boom():
        raise RuntimeError("failure")

    stub.get_positions = _boom
    client = tm.api_app.test_client()
    resp = client.get("/positions")
    assert resp.status_code == 500
    assert resp.json == {"error": "internal error"}
