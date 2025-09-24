from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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


def test_submit_dependency_snapshot_skips_when_requests_missing(monkeypatch, capsys):
    """The script should report a friendly message when ``requests`` is unavailable."""

    from scripts import submit_dependency_snapshot as snapshot

    monkeypatch.setattr(snapshot, "requests", None, raising=False)
    monkeypatch.setattr(
        snapshot,
        "_REQUESTS_IMPORT_ERROR",
        ImportError("requests package is not installed"),
        raising=False,
    )

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "requests" in captured.err
    assert "Skipping submission" in captured.err


def test_submit_with_headers_requires_requests(monkeypatch):
    """Submitting without ``requests`` should raise a descriptive error."""

    from scripts import submit_dependency_snapshot as snapshot

    monkeypatch.setattr(snapshot, "requests", None, raising=False)
    missing_error = ImportError("dependency missing")
    monkeypatch.setattr(
        snapshot,
        "_REQUESTS_IMPORT_ERROR",
        missing_error,
        raising=False,
    )

    with pytest.raises(snapshot.DependencySubmissionError) as excinfo:
        snapshot._submit_with_headers("https://example.com/api", b"{}", {})

    message = str(excinfo.value)
    assert "requests" in message
    assert excinfo.value.__cause__ is missing_error
