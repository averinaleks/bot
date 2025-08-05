import asyncio
import contextlib
import logging
import types
import os
import sys
import pytest
from bot.config import BotConfig

class DummyTL:
    def __init__(self):
        self.sent = []
    async def send_telegram_message(self, msg):
        self.sent.append(msg)
    @classmethod
    async def shutdown(cls):
        pass

class DummyDH:
    def __init__(self):
        self.usdt_pairs = ["BTCUSDT"]
        self.telegram_logger = DummyTL()

class DummyTM:
    pass

@pytest.mark.asyncio
async def test_backtest_loop_warns(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("TEST_MODE", "1")
    cfg = BotConfig(cache_dir=str(tmp_path), backtest_interval=0, min_sharpe_ratio=0.5)
    dh = DummyDH()
    gym_mod = types.ModuleType("gymnasium")
    gym_mod.Env = object
    spaces_mod = types.ModuleType("gymnasium.spaces")
    class DummyDiscrete:
        def __init__(self, n):
            self.n = n
    spaces_mod.Discrete = DummyDiscrete
    spaces_mod.Box = object
    gym_mod.spaces = spaces_mod
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces_mod)
    import importlib
    model_builder = importlib.import_module("model_builder")
    ModelBuilder = model_builder.ModelBuilder
    monkeypatch.setattr(
        model_builder,
        "_get_torch_modules",
        lambda: {"torch": types.SimpleNamespace(device=lambda *_: "cpu")},
    )
    mb = ModelBuilder(cfg, dh, DummyTM())

    calls = {"n": 0}
    async def fake_backtest_all():
        calls["n"] += 1
        return {"BTCUSDT": 0.4}
    monkeypatch.setattr(mb, "backtest_all", fake_backtest_all)

    caplog.set_level(logging.WARNING)

    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(mb.backtest_loop())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["n"] >= 1
    assert any("Sharpe ratio" in r.getMessage() for r in caplog.records)
    assert dh.telegram_logger.sent
