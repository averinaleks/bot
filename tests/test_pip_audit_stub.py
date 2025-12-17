from __future__ import annotations

import importlib
import sys

import pytest
from importlib import metadata


@pytest.fixture(autouse=True)
def reset_sys_modules():
    original_modules = sys.modules.copy()
    yield
    sys.modules.clear()
    sys.modules.update(original_modules)


def test_load_upstream_missing(monkeypatch):
    stub = importlib.import_module("pip_audit.__main__")

    def _raise_missing(_package: str) -> None:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise_missing)

    with pytest.raises(ModuleNotFoundError):
        stub._load_upstream_main()


def test_loads_real_module(monkeypatch, tmp_path):
    stub = importlib.import_module("pip_audit.__main__")

    module_root = tmp_path / "pip_audit"
    module_root.mkdir()
    (module_root / "__init__.py").write_text("\n")
    module_main = module_root / "__main__.py"
    module_main.write_text(
        """
def main(argv=None):
    return 0
"""
    )

    monkeypatch.setattr(metadata, "version", lambda _pkg: "1.0")
    monkeypatch.syspath_prepend(str(tmp_path))

    loaded = stub._load_upstream_main()

    assert loaded is not None
    assert loaded([]) == 0
