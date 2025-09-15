import atexit
import importlib
import sys
from unittest.mock import AsyncMock
import types


def test_shutdown_all_registered_and_called(monkeypatch):
    funcs = []
    monkeypatch.setattr(atexit, "register", lambda func: funcs.append(func))
    sys.modules.pop("telegram_logger", None)
    module = importlib.import_module("telegram_logger")

    mock = AsyncMock()
    monkeypatch.setattr(module, "TelegramLogger", types.SimpleNamespace(shutdown=mock))

    assert module._shutdown_all in funcs
    for func in funcs:
        func()

    mock.assert_awaited_once()
