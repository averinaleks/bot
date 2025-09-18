import importlib
import sys

import pytest


@pytest.mark.usefixtures("csrf_secret")
def test_import_requires_csrf_secret(monkeypatch):
    sys.modules.pop("server", None)
    importlib.import_module("server")
    monkeypatch.delenv("CSRF_SECRET", raising=False)
    sys.modules.pop("server", None)
    with pytest.raises(RuntimeError, match="CSRF_SECRET environment variable is required"):
        importlib.import_module("server")
    sys.modules.pop("server", None)


def test_calling_resolve_without_secret_raises(monkeypatch):
    monkeypatch.setenv("CSRF_SECRET", "static-test-secret")
    sys.modules.pop("server", None)
    server = importlib.import_module("server")
    assert server._resolve_csrf_secret() == "static-test-secret"
    monkeypatch.delenv("CSRF_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="CSRF_SECRET environment variable is required"):
        server._resolve_csrf_secret()
    sys.modules.pop("server", None)
