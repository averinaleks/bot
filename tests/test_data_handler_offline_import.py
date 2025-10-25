import builtins
import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _ensure_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("OFFLINE_MODE", "0")
    yield
    for name in list(sys.modules):
        if name.startswith("bot.data_handler"):
            sys.modules.pop(name, None)


def _clear_data_handler_modules():
    for name in list(sys.modules):
        if name.startswith("bot.data_handler"):
            sys.modules.pop(name, None)


def test_import_uses_offline_handler_without_numpy(monkeypatch):
    _clear_data_handler_modules()

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    module = importlib.import_module("bot.data_handler")
    assert module.DataHandler.__name__ == "OfflineDataHandler"
    assert module.np is None


def test_import_uses_offline_handler_when_core_dependencies_missing(monkeypatch):
    _clear_data_handler_modules()

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    module = importlib.import_module("bot.data_handler")
    assert module.DataHandler.__name__ == "OfflineDataHandler"
    atr = module.atr_fast([10.0, 11.0], [9.0, 8.5], [9.5, 9.0], 1)
    assert atr[-1] > 0


def test_offline_data_handler_personalisation_is_short(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    _clear_data_handler_modules()

    module = importlib.import_module("bot.data_handler.offline")

    # Instantiation should not raise due to blake2s personalisation length.
    handler = module.OfflineDataHandler()

    seed_first = handler._symbol_seed("BTCUSDT")
    seed_second = handler._symbol_seed("BTCUSDT")
    handler_again = module.OfflineDataHandler()
    seed_third = handler_again._symbol_seed("BTCUSDT")

    assert seed_first == seed_second == seed_third
