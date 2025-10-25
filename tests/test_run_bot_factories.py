import logging
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
