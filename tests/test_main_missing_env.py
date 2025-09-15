import sys
import types

import pytest
from pydantic import BaseModel

import trading_bot


def test_main_missing_required_env(monkeypatch):
    class Dummy(BaseModel):
        x: int

    def fake_get_settings():
        Dummy()  # raises ValidationError due to missing field

    stub = types.SimpleNamespace(get_settings=fake_get_settings)
    monkeypatch.setitem(sys.modules, "data_handler", stub)

    with pytest.raises(SystemExit):
        trading_bot.main()

