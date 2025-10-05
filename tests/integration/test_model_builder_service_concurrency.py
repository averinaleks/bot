from __future__ import annotations

import importlib
import sys
import threading
import time

import pytest

pytest.importorskip("flask")
pytestmark = pytest.mark.integration


@pytest.fixture()
def isolated_model_builder_service(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_FILE", raising=False)
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.delenv("NN_FRAMEWORK", raising=False)

    module_name = "services.model_builder_service"
    original_module = sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        module.app.testing = True
        yield module
    finally:
        module = sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        if module is not None:
            with module._state_lock:  # type: ignore[attr-defined]
                module._models.clear()
                module._scalers.clear()


def _train_payload(symbol: str) -> dict[str, list[list[float]] | list[int] | str]:
    return {
        "symbol": symbol,
        "features": [[float(i)] for i in range(4)],
        "labels": [0, 1, 0, 1],
    }


def test_train_predict_concurrency(monkeypatch: pytest.MonkeyPatch, isolated_model_builder_service):
    module = isolated_model_builder_service
    if not getattr(module, "JOBLIB_AVAILABLE", False):
        pytest.skip("joblib is required to exercise persistence race conditions")

    symbol = "concurrency"
    payload = _train_payload(symbol)

    original_dump = module.joblib.dump
    original_load = module.safe_joblib_load

    def slow_dump(*args, **kwargs):
        time.sleep(0.05)
        return original_dump(*args, **kwargs)

    def slow_load(*args, **kwargs):
        time.sleep(0.05)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(module.joblib, "dump", slow_dump)
    monkeypatch.setattr(module, "safe_joblib_load", slow_load)

    with module.app.test_client() as client:
        response = client.post("/train", json=payload)
        assert response.status_code == 200

    errors: list[tuple[str, int]] = []
    errors_lock = threading.Lock()
    start = threading.Event()

    def trainer() -> None:
        start.wait()
        for _ in range(3):
            with module.app.test_client() as client:
                resp = client.post("/train", json=payload)
            if resp.status_code != 200:
                with errors_lock:
                    errors.append(("train", resp.status_code))

    def predictor() -> None:
        start.wait()
        for _ in range(15):
            with module._state_lock:
                module._models.pop(symbol, None)
                module._scalers.pop(symbol, None)
            with module.app.test_client() as client:
                resp = client.post("/predict", json={"symbol": symbol, "features": [0.5]})
            if resp.status_code != 200:
                with errors_lock:
                    errors.append(("predict", resp.status_code))
                continue
            data = resp.get_json()
            assert data["signal"] in ("buy", "sell", None)
            assert 0.0 <= data["prob"] <= 1.0

    threads = [threading.Thread(target=trainer), threading.Thread(target=predictor)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert not errors, f"no concurrent failures expected, got: {errors}"
