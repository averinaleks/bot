import json
from pathlib import Path

import pytest

from bot import config


def test_max_position_pct_bounds():
    cfg_dir = Path(config.CONFIG_PATH).parent
    path = cfg_dir / "tmp_config_bounds.json"
    try:
        path.write_text(json.dumps({"max_position_pct": 1.5}))
        with pytest.raises(ValueError):
            config.load_config(str(path))
        path.write_text(json.dumps({"max_position_pct": -0.1}))
        with pytest.raises(ValueError):
            config.load_config(str(path))
        path.write_text(json.dumps({"max_position_pct": 0.2}))
        cfg = config.load_config(str(path))
        assert cfg.max_position_pct == 0.2
    finally:
        path.unlink(missing_ok=True)
