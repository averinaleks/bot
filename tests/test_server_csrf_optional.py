import importlib
import sys
import types

import pytest


def test_server_requires_csrf(monkeypatch):
    monkeypatch.setenv("API_KEYS", "testkey")
    monkeypatch.setitem(sys.modules, "fastapi_csrf_protect", None)
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: None
    dotenv_stub.dotenv_values = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)
    monkeypatch.delitem(sys.modules, "server", raising=False)
    with pytest.raises(RuntimeError):
        importlib.import_module("server")
