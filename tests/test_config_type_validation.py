import pytest

from config import BotConfig


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_positions", "5"),
        ("enable_grid_search", "false"),
        ("backup_ws_urls", "ws"),
    ],
)
def test_invalid_types_raise_value_error(key, value):
    with pytest.raises(ValueError, match=key):
        BotConfig(**{key: value})
