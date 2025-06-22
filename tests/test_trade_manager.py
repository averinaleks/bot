import pandas as pd
import pytest
import sys, types

if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch

class DummyTelegramLogger:
    def __init__(self, *a, **kw):
        pass
    async def send_telegram_message(self, *a, **kw):
        pass

import utils
utils.TelegramLogger = DummyTelegramLogger

from trade_manager import TradeManager

class DummyExchange:
    def __init__(self):
        self.orders = []
    async def fetch_balance(self):
        return {'total': {'USDT': 1000}}
    async def create_order(self, symbol, type, side, amount, price, params):
        self.orders.append({'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'params': params})
        return {'id': '1'}

class DummyIndicators:
    def __init__(self):
        self.atr = pd.Series([1.0])

class DummyDataHandler:
    def __init__(self):
        self.exchange = DummyExchange()
        self.usdt_pairs = ['BTCUSDT']
        idx = pd.MultiIndex.from_tuples([
            ('BTCUSDT', pd.Timestamp('2020-01-01'))
        ], names=['symbol', 'timestamp'])
        self.ohlcv = pd.DataFrame({'close': [100]}, index=idx)
        self.indicators = {'BTCUSDT': DummyIndicators()}

def make_config():
    return {
        'cache_dir': '/tmp',
        'max_positions': 5,
        'leverage': 10,
        'min_risk_per_trade': 0.01,
        'max_risk_per_trade': 0.05,
        'check_interval': 1,
        'performance_window': 60,
        'sl_multiplier': 1.0,
        'tp_multiplier': 2.0,
    }

def test_position_calculations():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute
    import asyncio
    size = asyncio.run(tm.calculate_position_size('BTCUSDT', 100, 1.0, 1.5))
    assert size == pytest.approx(10 / (1.5 * 10))

    stop_loss_price = 100 - 1.5 * 1.0
    take_profit_price = 100 + 2.5 * 1.0
    assert stop_loss_price == pytest.approx(98.5)
    assert take_profit_price == pytest.approx(102.5)

