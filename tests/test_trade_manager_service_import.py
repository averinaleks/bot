import builtins
import importlib
import sys
from types import ModuleType

import pytest


from bot import config as bot_config
from services.stubs import create_httpx_stub


def test_service_imports_without_httpx(monkeypatch):
    """Service module should fall back to the httpx stub when missing."""

    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setattr(bot_config, "OFFLINE_MODE", True, raising=False)
    monkeypatch.delitem(sys.modules, "bot.trade_manager.service", raising=False)
    monkeypatch.delitem(sys.modules, "bot.trade_manager", raising=False)
    monkeypatch.delitem(sys.modules, "httpx", raising=False)

    original_import = builtins.__import__
    import_attempts = {"count": 0}

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            if import_attempts["count"] == 0:
                import_attempts["count"] += 1
                raise ModuleNotFoundError("No module named 'httpx'")
            if "httpx" not in sys.modules:
                stub_namespace = create_httpx_stub()
                stub_module = ModuleType("httpx")
                stub_module.__dict__.update(stub_namespace.__dict__)
                sys.modules["httpx"] = stub_module
            return sys.modules["httpx"]
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    service = importlib.import_module("bot.trade_manager.service")

    stub_httpx = service.httpx

    assert getattr(stub_httpx, "__offline_stub__", False) is True

    with pytest.raises(stub_httpx.HTTPError):
        raise stub_httpx.HTTPError("stub error")
