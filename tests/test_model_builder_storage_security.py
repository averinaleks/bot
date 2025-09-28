import importlib.util
import sys
from pathlib import Path

import pytest


def _load_storage(monkeypatch, tmp_path, request):
    """Import ``model_builder.storage`` with an isolated MODEL_DIR."""

    import security

    original_model_dir = security.MODEL_DIR
    request.addfinalizer(lambda: setattr(security, "MODEL_DIR", original_model_dir))

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    module_name = "model_builder_storage_test"
    sys.modules.pop(module_name, None)
    storage_path = Path(__file__).resolve().parents[1] / "model_builder" / "storage.py"
    spec = importlib.util.spec_from_file_location(module_name, storage_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_save_artifacts_sanitizes_symbol(monkeypatch, tmp_path, request):
    storage = _load_storage(monkeypatch, tmp_path, request)
    unsafe_symbol = "../..//evil symbol!!"

    target_dir = storage.save_artifacts({"weights": [1, 2, 3]}, unsafe_symbol, {"info": "ok"})

    assert target_dir.is_dir()
    relative = target_dir.relative_to(storage.MODEL_DIR)
    assert len(relative.parts) == 2
    safe_symbol = relative.parts[0]
    assert safe_symbol == "evil-symbol"
    assert (target_dir / "meta.json").is_file()
    if storage.JOBLIB_AVAILABLE:
        assert (target_dir / "model.pkl").is_file()


def test_save_artifacts_rejects_invalid_symbol(monkeypatch, tmp_path, request):
    storage = _load_storage(monkeypatch, tmp_path, request)

    with pytest.raises(ValueError):
        storage.save_artifacts({"weights": []}, "..", {})

    assert list(tmp_path.iterdir()) == []


def test_save_artifacts_rejects_symlink_escape(monkeypatch, tmp_path, request):
    storage = _load_storage(monkeypatch, tmp_path, request)
    malicious = tmp_path / "escape"
    malicious.symlink_to(tmp_path.parent)

    with pytest.raises(ValueError):
        storage.save_artifacts({"weights": []}, "escape", {})


def test_save_artifacts_rejects_model_symlink(monkeypatch, tmp_path, request):
    storage = _load_storage(monkeypatch, tmp_path, request)
    if not storage.JOBLIB_AVAILABLE:
        pytest.skip("joblib is required to persist model artifacts")

    fixed_timestamp = 1_750_000_000
    monkeypatch.setattr(storage.time, "time", lambda: fixed_timestamp)

    symbol = "BTCUSDT"
    target_dir = storage._symbol_directory(symbol) / str(fixed_timestamp)
    target_dir.mkdir(parents=True, exist_ok=True)

    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"sentinel")
    (target_dir / "model.pkl").symlink_to(outside)

    with pytest.raises(RuntimeError, match="символь"):
        storage.save_artifacts({"weights": []}, symbol, {})

