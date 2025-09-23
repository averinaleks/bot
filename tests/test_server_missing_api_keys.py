import secrets
import sys

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_missing_api_keys_causes_startup_failure(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("CSRF_SECRET", secrets.token_hex(32))
    sys.modules.pop("server", None)
    import server
    server.API_KEYS.clear()
    with pytest.raises(RuntimeError, match="API_KEYS environment variable is required"):
        with TestClient(server.app):
            pass
    sys.modules.pop("server", None)
