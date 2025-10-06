from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytestmark = pytest.mark.integration


@pytest.fixture()
def offline_trade_manager(monkeypatch, tmp_path):
    """Provide a TradeManager service configured for offline integration tests."""

    # Ensure offline stubs are enabled and clean credentials are generated.
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("TRADE_MANAGER_TOKEN", raising=False)

    module_name = "services.trade_manager_service"
    original_module = sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)

        offline_mod = importlib.import_module("services.offline")
        monkeypatch.setattr(module.ccxt, "bybit", offline_mod.OfflineBybit)

        # Normalise ccxt exception hierarchy for the stub environment.
        class NetworkError(Exception):
            pass

        class BadRequestError(Exception):
            pass

        monkeypatch.setattr(module, "CCXT_BASE_ERROR", Exception)
        monkeypatch.setattr(module, "CCXT_NETWORK_ERROR", NetworkError)
        monkeypatch.setattr(module, "CCXT_BAD_REQUEST", BadRequestError)

        cache_file = tmp_path / "positions.json"
        monkeypatch.setattr(module, "POSITIONS_FILE", cache_file)
        state = module._get_state()
        state.clear_positions()
        if cache_file.exists():
            cache_file.unlink()
        module._load_positions()

        module.init_exchange()

        with module.app.test_client() as client:
            yield module, client, NetworkError, state
    finally:
        module = sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if module is not None:
            module._get_state().set_exchange(None)


def test_offline_open_close_cycle(offline_trade_manager):
    module, client, _, state = offline_trade_manager

    open_payload = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "amount": 1,
        "price": 25_000.0,
    }
    response = client.post("/open_position", json=open_payload)
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["status"] == "ok"
    order_id = payload["order_id"]

    positions_path = Path(module.POSITIONS_FILE)
    assert positions_path.exists(), "position cache should be created"
    cache_data = json.loads(positions_path.read_text("utf-8"))
    assert cache_data and cache_data[0]["id"] == order_id

    close_payload = {"order_id": order_id, "side": "sell"}
    close_response = client.post("/close_position", json=close_payload)
    assert close_response.status_code == 200
    close_data = close_response.get_json()
    assert close_data["status"] == "ok"

    # Positions should be cleared after the successful close request.
    assert not state.snapshot_positions()


def test_offline_open_position_network_degradation(monkeypatch, offline_trade_manager):
    module, client, network_error, state = offline_trade_manager

    exchange = state.get_exchange()
    original_create_order = exchange.create_order

    def flaky_create_order(*args, **kwargs):
        raise network_error("temporary outage")

    monkeypatch.setattr(exchange, "create_order", flaky_create_order)

    response = client.post(
        "/open_position",
        json={"symbol": "ETHUSDT", "side": "buy", "amount": 1, "price": 1_500.0},
    )
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"] == "network error contacting exchange"
    assert not state.snapshot_positions()

    # Restore the original implementation so subsequent tests can reuse the fixture safely.
    monkeypatch.setattr(exchange, "create_order", original_create_order)


def test_offline_close_position_network_degradation(monkeypatch, offline_trade_manager):
    module, client, network_error, state = offline_trade_manager

    open_response = client.post(
        "/open_position",
        json={"symbol": "SOLUSDT", "side": "buy", "amount": 2, "price": 40.0},
    )
    order_id = open_response.get_json()["order_id"]
    positions = state.snapshot_positions()
    assert positions and positions[0]["id"] == order_id

    exchange = state.get_exchange()
    original_create_order = exchange.create_order

    def failing_close(symbol, order_type, side, amount, price=None, params=None):
        raise network_error("connection reset")

    monkeypatch.setattr(exchange, "create_order", failing_close)

    close_response = client.post(
        "/close_position", json={"order_id": order_id, "side": "sell"}
    )
    assert close_response.status_code == 503
    payload = close_response.get_json()
    assert payload["error"] == "network error contacting exchange"

    # Original position should remain intact so operators can retry.
    positions = state.snapshot_positions()
    assert positions and positions[0]["id"] == order_id

    monkeypatch.setattr(exchange, "create_order", original_create_order)


def test_offline_recovers_from_corrupted_cache(offline_trade_manager):
    module, client, _, state = offline_trade_manager

    module.POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    module.POSITIONS_FILE.write_text("{not-valid-json", encoding="utf-8")

    # Corrupted cache should be ignored when reloading state.
    module._load_positions()
    assert not state.snapshot_positions()

    response = client.post(
        "/open_position",
        json={"symbol": "XRPUSDT", "side": "buy", "amount": 3, "price": 0.5},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"

    # The cache file must now contain valid JSON describing the new position.
    cache_data = json.loads(module.POSITIONS_FILE.read_text("utf-8"))
    assert cache_data and cache_data[0]["id"] == payload["order_id"]
