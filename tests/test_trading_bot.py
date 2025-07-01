import os
import sys
import types

# Ensure the project root is on the Python path so that 'trading_bot' can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Stub heavy dependencies before importing trading_bot
numba_mod = types.ModuleType('numba')
numba_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
numba_mod.jit = lambda *a, **k: (lambda f: f)
numba_mod.prange = range
sys.modules.setdefault('numba', numba_mod)
sys.modules.setdefault('numba.cuda', numba_mod.cuda)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
pybit_mod = types.ModuleType('pybit')
ut_mod = types.ModuleType('unified_trading')
ut_mod.HTTP = object
pybit_mod.unified_trading = ut_mod
sys.modules.setdefault('pybit', pybit_mod)
sys.modules.setdefault('pybit.unified_trading', ut_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)

import trading_bot


def test_send_trade_timeout_env(monkeypatch):
    called = {}

    def fake_post(url, json=None, timeout=None):
        called['timeout'] = timeout
        class Resp:
            status_code = 200
        return Resp()

    monkeypatch.setattr(trading_bot.requests, 'post', fake_post)
    monkeypatch.setenv('TRADE_MANAGER_TIMEOUT', '9')
    trading_bot.send_trade('BTCUSDT', 'buy', 100.0, {'trade_manager_url': 'http://tm'})
    assert called['timeout'] == 9.0

