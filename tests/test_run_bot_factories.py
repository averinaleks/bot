import pytest

pytest.importorskip("pandas")

from config import BotConfig, MissingEnvError

import run_bot


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
