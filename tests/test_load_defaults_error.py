import pytest
from bot import config


def test_load_defaults_raises_config_load_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{invalid")
    monkeypatch.setattr(config, "CONFIG_PATH", str(bad))
    monkeypatch.setattr(config, "DEFAULTS", None)
    with pytest.raises(config.ConfigLoadError):
        config.load_defaults()
