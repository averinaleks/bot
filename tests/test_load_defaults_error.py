from pathlib import Path

import contextlib
import pytest
from bot import config


def test_load_defaults_raises_config_load_error(monkeypatch):
    config_dir = Path(config.CONFIG_PATH).resolve().parent
    bad = config_dir / "bad.json"
    bad.write_text("{invalid")
    monkeypatch.setattr(config, "CONFIG_PATH", str(bad))
    monkeypatch.setattr(config, "DEFAULTS", None)
    try:
        with pytest.raises(config.ConfigLoadError):
            config.load_defaults()
    finally:
        with contextlib.suppress(FileNotFoundError):
            bad.unlink()
