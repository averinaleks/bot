import importlib
import sys

from fastapi.testclient import TestClient


def test_server_starts_without_csrf(monkeypatch):
    monkeypatch.setenv("API_KEYS", "testkey")
    monkeypatch.setitem(sys.modules, "fastapi_csrf_protect", None)
    monkeypatch.delitem(sys.modules, "server", raising=False)
    server = importlib.import_module("server")

    def dummy_load_model():
        server.model_manager.tokenizer = object()
        server.model_manager.model = object()

    monkeypatch.setattr(server.model_manager, "load_model", dummy_load_model)
    client = TestClient(server.app)
    with client:
        resp = client.get("/nonexistent", headers={"Authorization": "Bearer testkey"})
    assert resp.status_code == 404
