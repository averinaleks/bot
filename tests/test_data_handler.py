import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import importlib
import types
import logging
import pytest
from config import BotConfig

# Stub heavy dependencies before importing data_handler
sys.modules.setdefault('websockets', types.ModuleType('websockets'))
pybit_mod = types.ModuleType('pybit')
ut_mod = types.ModuleType('unified_trading')
ut_mod.HTTP = object
pybit_mod.unified_trading = ut_mod
sys.modules.setdefault('pybit', pybit_mod)
sys.modules.setdefault('pybit.unified_trading', ut_mod)
ray_mod = types.ModuleType('ray')
ray_mod.remote = lambda *a, **k: (lambda f: f)
sys.modules.setdefault('ray', ray_mod)
tenacity_mod = types.ModuleType('tenacity')
tenacity_mod.retry = lambda *a, **k: (lambda f: f)
tenacity_mod.wait_exponential = lambda *a, **k: None
tenacity_mod.stop_after_attempt = lambda *a, **k: (lambda f: f)
sys.modules.setdefault('tenacity', tenacity_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
numba_mod = types.ModuleType('numba')
numba_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
numba_mod.jit = lambda *a, **k: (lambda f: f)
numba_mod.prange = range
import importlib.machinery
numba_mod.__spec__ = importlib.machinery.ModuleSpec("numba", None)
sys.modules.setdefault('numba', numba_mod)
sys.modules.setdefault('numba.cuda', numba_mod.cuda)

# Replace utils with a stub that overrides TelegramLogger
real_utils = importlib.import_module('utils')
utils_stub = types.ModuleType('utils')
utils_stub.__dict__.update(real_utils.__dict__)
class DummyTL:
    def __init__(self, *a, **k):
        pass
    async def send_telegram_message(self, *a, **k):
        pass
    @classmethod
    async def shutdown(cls):
        pass
utils_stub.TelegramLogger = DummyTL
utils_stub.logger = logging.getLogger('test')
sys.modules['utils'] = utils_stub
os.environ['TEST_MODE'] = '1'

from data_handler import DataHandler

class DummyExchange:
    def __init__(self, volumes):
        self.volumes = volumes
    async def fetch_ticker(self, symbol):
        return {'quoteVolume': self.volumes.get(symbol, 0)}

@pytest.mark.asyncio
async def test_select_liquid_pairs_plain_symbol_included():
    cfg = BotConfig(cache_dir='/tmp', max_symbols=5)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    markets = {
        'BTCUSDT': {'active': True},
        'BTC/USDT': {'active': True},
    }
    pairs = await dh.select_liquid_pairs(markets)
    assert 'BTCUSDT' in pairs
