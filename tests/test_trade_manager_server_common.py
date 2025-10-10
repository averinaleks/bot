import asyncio
import importlib
import sys
import types

import pytest


@pytest.mark.asyncio
async def test_create_trade_manager_uses_shared_config(monkeypatch):
    import bot.trade_manager

    module = importlib.reload(bot.trade_manager.service)
    from bot.trade_manager import server_common

    sentinel_cfg = {"ray_num_cpus": 1}
    called = {}

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    def fake_load(path="config.json"):
        called["path"] = path
        return sentinel_cfg

    monkeypatch.setattr(server_common, "load_trade_manager_config", fake_load)

    class DummyDataHandler:
        def __init__(self, cfg, telegram_bot, chat_id):
            assert cfg is sentinel_cfg
            self.feature_callback = None
            self.usdt_pairs = []

        async def load_initial(self):
            return None

        async def stop(self):
            return None

        async def subscribe_to_klines(self, pairs):
            return None

    class DummyModelBuilder:
        def __init__(self, cfg, data_handler, trade_manager):
            assert cfg is sentinel_cfg
            self.precompute_features = lambda *_, **__: None

        async def train(self):
            return None

        async def backtest_loop(self):
            return None

    class DummyTradeManager:
        def __init__(self, cfg, dh, mb, bot, chat_id):
            assert cfg is sentinel_cfg
            self.loop = asyncio.get_event_loop()

    monkeypatch.setitem(
        sys.modules,
        "bot.data_handler",
        types.SimpleNamespace(DataHandler=DummyDataHandler),
    )
    monkeypatch.setitem(
        sys.modules,
        "bot.model_builder",
        types.SimpleNamespace(ModelBuilder=DummyModelBuilder),
    )
    monkeypatch.setattr(module, "TradeManager", DummyTradeManager)
    monkeypatch.setattr(module.ray, "is_initialized", lambda: True)

    module.trade_manager_factory.reset()
    try:
        manager = await module.create_trade_manager()
    finally:
        module.trade_manager_factory.reset()

    assert manager is not None
    assert called["path"] == "config.json"


def test_package_service_uses_common_token_validator(monkeypatch):
    import bot.trade_manager

    module = importlib.reload(bot.trade_manager.service)
    from bot.trade_manager import server_common

    calls: list[tuple[dict, str | None]] = []

    placeholder_token = "placeholder" + "_value"

    def fake_validate(headers, expected):
        calls.append((dict(headers), expected))
        return "missing placeholder"

    monkeypatch.setattr(server_common, "validate_token", fake_validate)
    module.TRADE_MANAGER_TOKEN = placeholder_token
    module.IS_TEST_MODE = False

    with module.api_app.test_request_context(
        "/open_position", method="POST", json={"symbol": "BTC"}
    ):
        response = module._require_token()

    assert calls
    assert response[1] == 401


def test_flask_service_uses_common_token_validator(monkeypatch):
    module = importlib.reload(importlib.import_module("services.trade_manager_service"))
    from bot.trade_manager import server_common

    module._reset_exchange_executor()

    calls: list[tuple[dict, str | None]] = []

    api_placeholder = "placeholder" + "_value"

    def fake_validate(headers, expected):
        calls.append((dict(headers), expected))
        return "placeholder mismatch"

    monkeypatch.setattr(server_common, "validate_token", fake_validate)
    module.API_TOKEN = api_placeholder
    module.IS_TEST_MODE = False

    try:
        with module.app.test_request_context(
            "/open_position", method="POST", json={"symbol": "BTC"}
        ):
            response = module._require_api_token()
    finally:
        module._reset_exchange_executor()

    assert calls
    assert response[1] == 401
