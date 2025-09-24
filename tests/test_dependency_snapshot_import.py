from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_dependency_snapshot_import_succeeds_without_requests(monkeypatch) -> None:
    """Ensure the dependency snapshot script has no hard dependency on ``requests``."""

    monkeypatch.setitem(sys.modules, "requests", None)

    module_name = "dependency_snapshot_no_requests"
    spec = importlib.util.spec_from_file_location(
        module_name, Path("scripts/submit_dependency_snapshot.py")
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert hasattr(module, "submit_dependency_snapshot")
    assert callable(module.submit_dependency_snapshot)
