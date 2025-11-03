from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _import_service(monkeypatch: pytest.MonkeyPatch):
    class DummyExchange:
        def fetch_ticker(self, symbol):  # pragma: no cover - simple stub
            return {"last": 1.0}

        def fetch_ohlcv(self, symbol, timeframe="1m", limit=200):  # pragma: no cover - simple stub
            return [[0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    ccxt = types.ModuleType("ccxt")
    ccxt.bybit = lambda *args, **kwargs: DummyExchange()
    ccxt.BaseError = Exception
    ccxt.NetworkError = Exception
    monkeypatch.setitem(sys.modules, "ccxt", ccxt)

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delitem(sys.modules, "bot.services.data_handler_service", raising=False)
    monkeypatch.delitem(sys.modules, "services.data_handler_service", raising=False)

    return importlib.import_module("bot.services.data_handler_service")


def test_normalise_cache_dir_accepts_regular_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _import_service(monkeypatch)

    target = tmp_path / "history"
    result = module._normalise_cache_dir(str(target))

    assert result == target.resolve()
    assert result.is_dir()


def test_normalise_cache_dir_rejects_symlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _import_service(monkeypatch)

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)

    assert module._normalise_cache_dir(str(link)) is None
