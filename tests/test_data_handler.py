import os, sys
import importlib
import types
import logging
import pytest
from config import BotConfig

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


class DummyWS:
    def __init__(self):
        self.sent = []
    async def send(self, message):
        self.sent.append(message)
    async def recv(self):
        return '{"success": true}'


@pytest.mark.asyncio
async def test_ws_rate_limit_zero_no_exception():
    cfg = BotConfig(cache_dir='/tmp', ws_rate_limit=0)
    dh = DataHandler(cfg, None, None, exchange=DummyExchange({'BTCUSDT': 1.0}))
    ws = DummyWS()
    await dh._send_subscriptions(ws, ['BTCUSDT'], 'primary')
    assert ws.sent
