import json

from bot import config


def test_load_config_accepts_external_absolute_path(tmp_path):
    external_dir = tmp_path / "outside"
    external_dir.mkdir()
    config_path = external_dir / "custom_config.json"
    config_path.write_text(json.dumps({"max_positions": 42}), encoding="utf-8")

    cfg = config.load_config(str(config_path))

    assert cfg.max_positions == 42
