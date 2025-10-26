import os
from pathlib import Path

import pytest

os.environ.setdefault("CSRF_SECRET", "test-secret")

import server


def test_prepare_model_cache_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "cache-dir"

    resolved = server._prepare_model_cache_dir(str(target))

    assert resolved == target.resolve()
    assert resolved.is_dir()


def test_prepare_model_cache_dir_rejects_symlink(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)

    assert server._prepare_model_cache_dir(str(link)) is None


@pytest.mark.parametrize("value", ["", None])
def test_prepare_model_cache_dir_handles_empty(value: str | None) -> None:
    assert server._prepare_model_cache_dir(value) is None


def test_prepare_model_cache_dir_rejects_regular_file(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("data", encoding="utf-8")

    assert server._prepare_model_cache_dir(str(file_path)) is None


def test_prepare_model_cache_dir_ignores_non_creatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sub" / "cache"

    def _raise(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(Path, "mkdir", _raise)

    assert server._prepare_model_cache_dir(str(target)) is None
