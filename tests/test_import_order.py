import importlib
import sys
import types
import logging
import os
import pytest


@pytest.fixture(autouse=True)
def _test_mode_env(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    yield
    monkeypatch.delenv("TEST_MODE", raising=False)

def test_telegramlogger_injection_order():
    sys.modules.pop('trade_manager', None)
    utils_stub = types.ModuleType('utils')
    class StubTL:
        pass
    utils_stub.TelegramLogger = StubTL
    utils_stub.logger = logging.getLogger('test')
    async def _cde(*a, **k):
        return False
    utils_stub.check_dataframe_empty = _cde
    utils_stub.check_dataframe_empty_async = _cde
    utils_stub.is_cuda_available = lambda: False
    async def _safe_api_call(exchange, method: str, *args, **kwargs):
        return await getattr(exchange, method)(*args, **kwargs)
    utils_stub.safe_api_call = _safe_api_call
    sys.modules['utils'] = utils_stub
    sys.modules['bot.utils'] = utils_stub

    tm = importlib.import_module('trade_manager')
    assert tm.TelegramLogger is StubTL
    sys.modules.pop('utils', None)
    sys.modules.pop('bot.utils', None)
    sys.modules.pop("trade_manager", None)
