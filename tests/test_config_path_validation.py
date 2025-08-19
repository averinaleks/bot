import pytest
from bot import config


def test_load_config_rejects_parent_directory():
    with pytest.raises(ValueError):
        config.load_config("../config.json")
