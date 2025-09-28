from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import pytest

import security
from security import _get_model_state_hmac_key


class DummyModel:
    """Minimal estimator stub used for signature validation tests."""

    def predict_proba(self, _features):
        return [[0.4, 0.6]]


@pytest.fixture
def isolated_model_builder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Load :mod:`model_builder` with isolated MODEL_DIR/MODEL_FILE settings."""

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_FILE", "model.pkl")
    package_path = Path(__file__).resolve().parent.parent / "model_builder"
    spec = importlib.util.spec_from_file_location(
        "test_model_builder_module",
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in list(sys.modules):
            if name == spec.name or name.startswith(f"{spec.name}."):
                sys.modules.pop(name, None)


def _model_path(module) -> Path:
    path = module._safe_model_file_path()
    assert path is not None
    return path


def test_load_model_validates_signatures(
    monkeypatch: pytest.MonkeyPatch, isolated_model_builder
) -> None:
    module = isolated_model_builder
    model_path = _model_path(module)
    joblib.dump(DummyModel(), model_path)

    monkeypatch.setenv("MODEL_STATE_HMAC_KEY", "super-secret")
    _get_model_state_hmac_key.cache_clear()

    sig_path = model_path.with_name(model_path.name + ".sig")
    if sig_path.exists() or sig_path.is_symlink():
        sig_path.unlink(missing_ok=True)

    module._model = None
    module._load_model()
    assert module._model is None, "Model must not load without a signature"

    security.write_model_state_signature(model_path)
    module._model = None
    module._load_model()
    assert isinstance(module._model, DummyModel)

    sig_path.write_text("tampered", encoding="utf-8")
    module._model = None
    module._load_model()
    assert module._model is None, "Model must be rejected when signature mismatches"

    sig_path.unlink()
    malicious = model_path.parent / "malicious.sig"
    malicious.write_text("bogus", encoding="utf-8")
    sig_path.symlink_to(malicious)
    module._model = None
    module._load_model()
    assert module._model is None, "Model must be rejected when signature path is a symlink"

