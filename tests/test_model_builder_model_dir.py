import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest


def _load_model_builder(module_name: str) -> object:
    package_path = Path(__file__).resolve().parents[1] / "model_builder"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)
    return module


def test_model_dir_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "1")
    primary = tmp_path / "primary"
    fallback_root = tmp_path / "fallback_root"
    monkeypatch.setenv("MODEL_DIR", str(primary))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback_root))

    primary_resolved = primary.resolve()
    original_mkdir = Path.mkdir

    def fake_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self == primary_resolved:
            raise PermissionError("denied")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    module = _load_model_builder("model_builder_temp_fallback")
    expected = (fallback_root / "trading_bot_models").resolve()

    assert module.MODEL_DIR == expected
    assert expected.exists()


def test_model_dir_raises_when_all_candidates_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "primary"))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "fallback"))

    def fail_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        raise PermissionError("no access")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(PermissionError):
        _load_model_builder("model_builder_temp_failure")
