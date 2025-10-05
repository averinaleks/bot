from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor


def _import_service(monkeypatch):
    creation_lock = threading.Lock()
    created_ids: list[int] = []
    closed_ids: list[int] = []
    active_instances: set[int] = set()

    class DummyExchange:
        def __init__(self):
            with creation_lock:
                created_ids.append(id(self))

        def _start_call(self) -> None:
            with creation_lock:
                if id(self) in active_instances:
                    raise AssertionError("exchange reused concurrently")
                active_instances.add(id(self))

        def _end_call(self) -> None:
            with creation_lock:
                active_instances.discard(id(self))

        def fetch_ticker(self, symbol):
            self._start_call()
            try:
                time.sleep(0.01)
                return {"last": 1.0}
            finally:
                self._end_call()

        def fetch_ohlcv(self, symbol, timeframe="1m", limit=200):
            self._start_call()
            try:
                time.sleep(0.01)
                return [[0, 1.0, 1.0, 1.0, 1.0, 1.0]] * min(limit, 1)
            finally:
                self._end_call()

        def close(self):
            with creation_lock:
                closed_ids.append(id(self))

    ccxt = types.ModuleType("ccxt")
    ccxt.bybit = lambda *args, **kwargs: DummyExchange()
    ccxt.BaseError = Exception
    ccxt.NetworkError = Exception
    monkeypatch.setitem(sys.modules, "ccxt", ccxt)

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("DATA_HANDLER_ALLOW_ANONYMOUS", "1")

    monkeypatch.delitem(sys.modules, "bot.services.data_handler_service", raising=False)
    monkeypatch.delitem(sys.modules, "services.data_handler_service", raising=False)

    module = importlib.import_module("bot.services.data_handler_service")
    module.app.testing = True

    return module, created_ids, closed_ids


def test_parallel_requests_receive_isolated_clients(monkeypatch):
    module, created_ids, closed_ids = _import_service(monkeypatch)

    total_requests = 12
    barrier = threading.Barrier(total_requests)

    def request_price():
        with module.app.test_client() as client:
            barrier.wait()
            resp = client.get("/price/BTCUSDT")
            assert resp.status_code == 200
            assert resp.get_json() == {"price": 1.0}

    def request_history():
        with module.app.test_client() as client:
            barrier.wait()
            resp = client.get("/history/BTCUSDT")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "history" in data
            assert data["history"]

    with ThreadPoolExecutor(max_workers=total_requests) as executor:
        futures = []
        for _ in range(total_requests // 2):
            futures.append(executor.submit(request_price))
            futures.append(executor.submit(request_history))
        for future in futures:
            future.result(timeout=10)

    assert len(set(created_ids)) >= total_requests
    assert set(created_ids).issubset(set(closed_ids))
