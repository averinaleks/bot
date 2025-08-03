import importlib
import os
import types
import inspect
import asyncio
import concurrent.futures
import logging


class DummyLoop:
    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, callback, *args):
        self.calls.append((callback, args))


async def dummy_coroutine(*_args, **_kwargs):
    pass


def _setup_module(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    import sys
    # stub heavy deps before import
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", torch)

    ray_mod = types.ModuleType("ray")
    ray_mod.is_initialized = lambda: False
    ray_mod.init = lambda *a, **k: None
    sys.modules.setdefault("ray", ray_mod)

    utils_stub = types.ModuleType("utils")
    class DummyTL:
        def __init__(self, *a, **k):
            pass
        async def send_telegram_message(self, *a, **k):
            pass
        @classmethod
        async def shutdown(cls):
            pass
    utils_stub.TelegramLogger = DummyTL
    utils_stub.logger = logging.getLogger("test")
    async def _cde(*a, **k):
        return False
    utils_stub.check_dataframe_empty = _cde
    utils_stub.check_dataframe_empty_async = _cde
    utils_stub.is_cuda_available = lambda: False
    async def _safe_api_call(exchange, method: str, *args, **kwargs):
        return await getattr(exchange, method)(*args, **kwargs)
    utils_stub.safe_api_call = _safe_api_call
    sys.modules["utils"] = utils_stub

    joblib_mod = types.ModuleType("joblib")
    joblib_mod.dump = lambda *a, **k: None
    joblib_mod.load = lambda *a, **k: {}
    sys.modules.setdefault("joblib", joblib_mod)

    monkeypatch.syspath_prepend(os.getcwd())
    sys.modules.pop("trade_manager", None)
    tm = importlib.import_module("trade_manager")
    loop = DummyLoop()
    stub = types.SimpleNamespace(
        loop=loop,
        open_position=dummy_coroutine,
        run=dummy_coroutine,
        get_stats=lambda: {"win_rate": 1.0},
    )
    tm.trade_manager = stub
    tm._ready_event.set()
    monkeypatch.delenv("TEST_MODE", raising=False)
    return tm, loop, stub


def test_open_position_route_schedules_task(monkeypatch):
    tm, loop, _ = _setup_module(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.post(
        "/open_position",
        json={"symbol": "BTCUSDT", "side": "buy", "price": 100.0},
    )
    assert resp.status_code == 200
    assert loop.calls, "call_soon_threadsafe not called"
    cb, args = loop.calls[0]
    assert cb is asyncio.create_task
    assert inspect.iscoroutine(args[0])
    assert args[0].cr_code is dummy_coroutine.__code__
    args[0].close()


def test_start_route_schedules_run(monkeypatch):
    tm, loop, _ = _setup_module(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.get("/start")
    assert resp.status_code == 200
    assert loop.calls, "call_soon_threadsafe not called"
    cb, args = loop.calls[0]
    assert cb is asyncio.create_task
    assert inspect.iscoroutine(args[0])
    assert args[0].cr_code is dummy_coroutine.__code__
    args[0].close()


def test_stats_route_returns_data(monkeypatch):
    tm, _, _ = _setup_module(monkeypatch)
    client = tm.api_app.test_client()
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json["stats"]["win_rate"] == 1.0


def test_open_position_route_concurrent(monkeypatch):
    tm, loop, _ = _setup_module(monkeypatch)

    def _call():
        client = tm.api_app.test_client()
        resp = client.post(
            "/open_position",
            json={"symbol": "BTCUSDT", "side": "buy", "price": 100.0},
        )
        assert resp.status_code == 200

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_call) for _ in range(5)]
        for fut in futures:
            fut.result()

    assert len(loop.calls) == 5
    for _, call_args in loop.calls:
        call_args[0].close()
