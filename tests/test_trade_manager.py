import pandas as pd
import pytest
import sys
import types
import logging
import os
from trade_manager import TradeManager

if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch

class DummyTelegramLogger:
    def __init__(self, *a, **kw):
        pass
    async def send_telegram_message(self, *a, **kw):
        pass

utils = types.ModuleType('utils')
utils.TelegramLogger = DummyTelegramLogger
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
sys.modules['utils'] = utils

tenacity_mod = types.ModuleType('tenacity')
tenacity_mod.retry = lambda *a, **k: (lambda f: f)
tenacity_mod.wait_exponential = lambda *a, **k: None
tenacity_mod.stop_after_attempt = lambda *a, **k: None
sys.modules['tenacity'] = tenacity_mod
scipy_mod = types.ModuleType('scipy')
stats_mod = types.ModuleType('scipy.stats')
stats_mod.zscore = lambda a, axis=0: (a - a.mean()) / a.std()
scipy_mod.__version__ = "1.0"
scipy_mod.stats = stats_mod
sys.modules.setdefault('scipy', scipy_mod)
sys.modules.setdefault('scipy.stats', stats_mod)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)
joblib_mod = types.ModuleType('joblib')
joblib_mod.dump = lambda *a, **k: None
joblib_mod.load = lambda *a, **k: {}
sys.modules.setdefault('joblib', joblib_mod)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DummyExchange:
    def __init__(self):
        self.orders = []
    async def fetch_balance(self):
        return {'total': {'USDT': 1000}}
    async def create_order(self, symbol, type, side, amount, price, params):
        self.orders.append({'method': 'create_order', 'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'params': params})
        return {'id': '1'}
    async def create_order_with_take_profit_and_stop_loss(self, symbol, type, side, amount, price, takeProfit, stopLoss, params):
        self.orders.append({'method': 'create_order_with_tp_sl', 'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'tp': takeProfit, 'sl': stopLoss,
                             'params': params})
        return {'id': '2'}

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

    async def get_atr(self, symbol: str) -> float:
        ind = self.indicators.get(symbol)
        return float(ind.atr.iloc[-1]) if ind else 0.0

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


def test_open_position_places_tp_sl_orders():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert dh.exchange.orders, 'no orders created'
    order = dh.exchange.orders[0]
    assert order['method'] == 'create_order_with_tp_sl'
    assert order['tp'] == pytest.approx(102.0)
    assert order['sl'] == pytest.approx(99.0)


def test_trailing_stop_to_breakeven():
    dh = DummyDataHandler()
    cfg = make_config()
    cfg.update({
        'trailing_stop_percentage': 1.0,
        'trailing_stop_coeff': 0.0,
        'trailing_stop_multiplier': 1.0,
    })
    tm = TradeManager(cfg, dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})
        await tm.check_trailing_stop('BTCUSDT', 101)

    import asyncio
    asyncio.run(run())

    assert len(dh.exchange.orders) >= 2
    assert tm.positions.iloc[0]['breakeven_triggered'] is True
    assert tm.positions.iloc[0]['size'] < dh.exchange.orders[0]['amount']

