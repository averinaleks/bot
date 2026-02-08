import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

import run_bot


@pytest.fixture(autouse=True)
def clear_offline_mode(monkeypatch):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)


def _clear_required_env(monkeypatch):
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TRADE_MANAGER_TOKEN",
        "TRADE_RISK_USD",
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)


def _get_dotenv_module(monkeypatch):
    try:
        import dotenv  # type: ignore

        return dotenv
    except ModuleNotFoundError:
        stub = ModuleType("dotenv")
        stub.dotenv_values = lambda *_, **__: {}
        monkeypatch.setitem(sys.modules, "dotenv", stub)
        return stub


def test_configure_environment_requires_explicit_offline(monkeypatch):
    _clear_required_env(monkeypatch)

    dotenv = _get_dotenv_module(monkeypatch)

    monkeypatch.setattr(dotenv, "dotenv_values", lambda *_, **__: {})

    args = SimpleNamespace(offline=False, auto_offline=False)

    with pytest.raises(SystemExit, match=r"Отсутствуют обязательные переменные окружения"):
        run_bot.configure_environment(args)

    assert os.getenv("OFFLINE_MODE") is None


def test_configure_environment_auto_offline_switch(monkeypatch, caplog):
    _clear_required_env(monkeypatch)

    dotenv = _get_dotenv_module(monkeypatch)

    monkeypatch.setattr(dotenv, "dotenv_values", lambda *_, **__: {})

    args = SimpleNamespace(offline=False, auto_offline=True)

    caplog.set_level("WARNING", logger="TradingBot")
    offline_mode = run_bot.configure_environment(args)

    assert offline_mode is True
    assert os.getenv("OFFLINE_MODE") == "1"
    assert any("--auto-offline" in msg for msg in caplog.messages)


def test_configure_environment_tolerates_missing_offline(monkeypatch, caplog):
    _clear_required_env(monkeypatch)

    dotenv = _get_dotenv_module(monkeypatch)

    monkeypatch.setattr(dotenv, "dotenv_values", lambda *_, **__: {})

    def _boom():
        raise ImportError("no stubs")

    monkeypatch.setattr(run_bot, "_import_offline_module", _boom)

    args = SimpleNamespace(offline=True, auto_offline=False)
    caplog.set_level("WARNING", logger="TradingBot")

    offline_mode = run_bot.configure_environment(
        args, allow_missing_offline_stubs=True
    )

    assert offline_mode is True
    assert run_bot.OFFLINE_STUBS_AVAILABLE is False
    assert any("офлайн-заглушек" in message for message in caplog.messages)


def test_offline_placeholders_are_stable(monkeypatch):
    import services.offline as offline

    monkeypatch.setattr(offline, "OFFLINE_MODE", True)
    monkeypatch.setattr(offline, "_OFFLINE_ENV_RESOLVED", {}, raising=False)

    counter = 0

    def generate():
        nonlocal counter
        counter += 1
        return f"value-{counter}"

    mapping = {"FOO": generate}

    monkeypatch.delenv("FOO", raising=False)
    applied_first = offline.ensure_offline_env(mapping)

    assert os.getenv("FOO") == "value-1"
    assert applied_first == ["FOO"]
    assert counter == 1

    monkeypatch.delenv("FOO", raising=False)
    applied_second = offline.ensure_offline_env(mapping)

    assert os.getenv("FOO") == "value-1"
    assert applied_second == ["FOO"]
    assert counter == 1
