from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("flask")
pytestmark = pytest.mark.integration


@pytest.fixture()
def model_builder_service(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("NN_FRAMEWORK", "sklearn")

    module_name = "services.model_builder_service"
    original_module = sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        with module.app.test_client() as client:
            yield module, client
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def test_predict_returns_string_signal_without_model(model_builder_service):
    _, client = model_builder_service

    response = client.post("/predict", json={"price": -1.0})
    assert response.status_code == 200

    payload = response.get_json()
    assert isinstance(payload["signal"], str)
    assert payload["signal"] in {"buy", "sell"}
