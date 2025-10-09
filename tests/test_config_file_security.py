import errno
from pathlib import Path

import pytest

from bot import config


def test_open_config_file_rejects_symlink(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")

    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - symlinks may be unavailable
        if exc.errno in {errno.EPERM, errno.EACCES}:
            pytest.skip(f"symlinks unavailable: {exc}")
        raise

    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="symlink"):
        config.open_config_file(link)
