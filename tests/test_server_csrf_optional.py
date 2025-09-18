import importlib
import sys

import pytest


def test_server_requires_csrf(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi_csrf_protect", None)
    monkeypatch.setitem(sys.modules, "dotenv", None)
    monkeypatch.delitem(sys.modules, "server", raising=False)
    with pytest.raises(RuntimeError):
        importlib.import_module("server")
