import hashlib
import logging
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pandas")

from config import BotConfig, MissingEnvError

import run_bot
from run_bot import _instantiate_factory


class DummyConfig(SimpleNamespace):
    pass


def _base_config(**overrides):
    cfg = BotConfig(**{"service_factories": {}, **overrides})
    return cfg


def test_build_components_missing_exchange_factory():
    cfg = _base_config()
    message = (
        r"No service factory configured for 'exchange'.*service_factories.*--offline"
    )
    with pytest.raises(ValueError, match=message):
        run_bot._build_components(cfg, offline=False, symbols=None)


def test_build_components_invalid_exchange_factory():
    cfg = _base_config(service_factories={"exchange": "bot:missing"})
    message = "Failed to load service factory 'exchange' from 'bot:missing'"
    with pytest.raises(ValueError, match=message):
        run_bot._build_components(cfg, offline=False, symbols=None)


def test_missing_env_error_message_contains_hint():
    exc = MissingEnvError(["FOO", "BAR"])
    message = str(exc)
    assert "Run 'python run_bot.py --offline'" in message
    assert "create a .env file with the required variables" in message


def test_incompatible_signature_logs_warning(caplog):
    def factory(other, another):  # noqa: ARG001
        raise AssertionError("Should not be called")

    cfg = DummyConfig()

    caplog.set_level(logging.WARNING, logger="TradingBot")

    result = _instantiate_factory(factory, cfg)

    assert result is None
    assert any(
        "Не удалось создать экземпляр" in message for message in caplog.messages
    ), caplog.records


def test_factory_type_error_propagates():
    def factory(cfg):  # noqa: ARG001
        raise TypeError("inner boom")

    cfg = DummyConfig()

    with pytest.raises(TypeError, match="inner boom"):
        _instantiate_factory(factory, cfg)


def test_factory_with_config_keyword():
    def factory(*, config):
        return config

    cfg = DummyConfig(value=42)

    result = _instantiate_factory(factory, cfg)

    assert result is cfg


def test_build_components_offline_overrides_factories(monkeypatch):
    cfg = BotConfig(
        service_factories={
            "exchange": "services.bybit:RealBybit",
            "telegram_logger": "services.telegram:RealTelegram",
            "gpt_client": "services.gpt:RealGPT",
            "model_builder": "model_builder.core:ModelBuilder",
        }
    )

    import bot.config as config_module

    monkeypatch.setattr(config_module, "OFFLINE_MODE", True, raising=False)

    import data_handler.offline as offline_data_module

    original_blake2s = hashlib.blake2s

    def _safe_blake2s(*args, **kwargs):  # noqa: ANN002, ANN003 - test helper
        if len(args) >= 3:
            args = list(args)
            args[2] = args[2][:8]
        elif "person" in kwargs and kwargs["person"] is not None:
            kwargs = dict(kwargs)
            kwargs["person"] = kwargs["person"][:8]
        return original_blake2s(*args, **kwargs)

    monkeypatch.setattr(hashlib, "blake2s", _safe_blake2s)

    monkeypatch.setattr(offline_data_module.OfflineDataHandler, "refresh", lambda self: None)
    monkeypatch.setattr(offline_data_module, "_SYMBOL_PERSONALISATION", b"offline")

    removed_modules: dict[str, object] = {}
    for module_name in ("bot.data_handler", "bot.model_builder", "bot.trade_manager"):
        removed_modules[module_name] = sys.modules.pop(module_name, None)

    try:
        data_handler, model_builder, trade_manager = run_bot._build_components(
            cfg, offline=True, symbols=None
        )
    finally:
        for module_name, module in removed_modules.items():
            if module is not None:
                sys.modules[module_name] = module

    from services.offline import OfflineBybit, OfflineGPT, OfflineTelegram

    assert isinstance(data_handler.exchange, OfflineBybit)
    assert model_builder.__class__.__module__.endswith("model_builder.offline")
    assert model_builder.__class__.__name__ == "OfflineModelBuilder"
    assert getattr(model_builder, "__offline_stub__", False)
    assert trade_manager.gpt_client_factory is OfflineGPT
    assert isinstance(trade_manager.telegram_logger, OfflineTelegram)
