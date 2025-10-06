from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bot import utils_loader


@pytest.fixture(autouse=True)
def _reset_utils_loader_cache():
    original_cache = utils_loader._UTILS_CACHE
    utils_loader._UTILS_CACHE = None
    try:
        yield
    finally:
        utils_loader._UTILS_CACHE = original_cache


def test_resolve_utils_path_rejects_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1", encoding="utf-8")
    target = project_root / "utils.py"
    target.symlink_to(outside)

    with pytest.raises(ImportError, match="symlink"):
        utils_loader._resolve_utils_path(project_root)


def test_resolve_utils_path_requires_regular_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    utils_dir = project_root / "utils.py"
    utils_dir.mkdir(parents=True)

    with pytest.raises(ImportError, match="regular file"):
        utils_loader._resolve_utils_path(project_root)


def test_load_from_source_uses_secure_path(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    utils_path = project_root / "utils.py"
    utils_path.write_text("VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr(utils_loader, "_project_root", lambda: project_root)
    monkeypatch.setattr(utils_loader, "_UTILS_CACHE", None)

    original_stub = sys.modules.pop("_bot_real_utils", None)
    try:
        resolved = utils_loader._resolve_utils_path()
        assert resolved == utils_path.resolve()

        module = utils_loader._load_from_source()
        assert getattr(module, "VALUE") == 1
    finally:
        if original_stub is not None:
            sys.modules["_bot_real_utils"] = original_stub
        else:
            sys.modules.pop("_bot_real_utils", None)
