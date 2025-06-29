import pandas as pd
import pytest
import sys
import types
import logging
import os
from config import BotConfig

# Stub heavy dependencies before importing the trade manager
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch

ray_mod = types.ModuleType('ray')
ray_mod.remote = lambda *a, **k: (lambda f: f)
ray_mod.get = lambda x: x
sys.modules.setdefault('ray', ray_mod)

sk_mod = types.ModuleType('sklearn')
model_sel = types.ModuleType('sklearn.model_selection')
model_sel.GridSearchCV = object
sk_mod.model_selection = model_sel
base_estimator = types.ModuleType('sklearn.base')
base_estimator.BaseEstimator = object
sk_mod.base = base_estimator
sys.modules.setdefault('sklearn', sk_mod)
sys.modules.setdefault('sklearn.model_selection', model_sel)
sys.modules.setdefault('sklearn.base', base_estimator)
utils_stub = types.ModuleType('utils')
class _TL:
    def __init__(self, *a, **k):
        pass
    async def send_telegram_message(self, *a, **k):
        pass
utils_stub.TelegramLogger = _TL
utils_stub.logger = logging.getLogger('test')
async def _cde_stub(*a, **kw):
    return False
utils_stub.check_dataframe_empty = _cde_stub
sys.modules['utils'] = utils_stub
os.environ["TEST_MODE"] = "1"
tenacity_mod = types.ModuleType('tenacity')
tenacity_mod.retry = lambda *a, **k: (lambda f: f)
tenacity_mod.wait_exponential = lambda *a, **k: None
tenacity_mod.stop_after_attempt = lambda *a, **k: None
sys.modules.setdefault('tenacity', tenacity_mod)
joblib_mod = types.ModuleType('joblib')
joblib_mod.dump = lambda *a, **k: None
joblib_mod.load = lambda *a, **k: {}
sys.modules.setdefault('joblib', joblib_mod)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)

import trade_manager
from trade_manager import TradeManager  # noqa: E402

def test_utils_injected_before_trade_manager_import():
    assert trade_manager.TelegramLogger is _TL

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DummyExchange:
    def __init__(self):
        self.orders = []
        self.fail = False
    async def fetch_balance(self):
        return {'total': {'USDT': 1000}}
    async def create_order(self, symbol, type, side, amount, price, params):
        self.orders.append({'method': 'create_order', 'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'params': params})
        if self.fail:
            return {'retCode': 1}
        return {'id': '1'}
    async def create_order_with_take_profit_and_stop_loss(self, symbol, type, side, amount, price, takeProfit, stopLoss, params):
        self.orders.append({'method': 'create_order_with_tp_sl', 'symbol': symbol, 'type': type, 'side': side,
                             'amount': amount, 'price': price, 'tp': takeProfit, 'sl': stopLoss,
                             'params': params})
        if self.fail:
            return {'retCode': 1}
        return {'id': '2'}

class DummyIndicators:
    def __init__(self):
        self.atr = pd.Series([1.0])

class DummyDataHandler:
    def __init__(self, fresh: bool = True, fail_order: bool = False):
        self.exchange = DummyExchange()
        self.exchange.fail = fail_order
        self.usdt_pairs = ['BTCUSDT']
        idx = pd.MultiIndex.from_tuples([
            ('BTCUSDT', pd.Timestamp('2020-01-01'))
        ], names=['symbol', 'timestamp'])
        self.ohlcv = pd.DataFrame({'close': [100]}, index=idx)
        self.indicators = {'BTCUSDT': DummyIndicators()}
        self.fresh = fresh

    async def get_atr(self, symbol: str) -> float:
        ind = self.indicators.get(symbol)
        return float(ind.atr.iloc[-1]) if ind else 0.0

    async def is_data_fresh(self, symbol: str, timeframe: str = 'primary', max_delay: float = 60) -> bool:
        return self.fresh

def make_config():
    return BotConfig(
        cache_dir='/tmp',
        max_positions=5,
        leverage=10,
        min_risk_per_trade=0.01,
        max_risk_per_trade=0.05,
        check_interval=1,
        performance_window=60,
        sl_multiplier=1.0,
        tp_multiplier=2.0,
    )

def test_position_calculations():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute
    import asyncio
    size = asyncio.run(tm.calculate_position_size('BTCUSDT', 100, 1.0, 1.5))
    assert size == pytest.approx(10 / (1.5 * 10))

    sl, tp = tm.calculate_stop_loss_take_profit('buy', 100, 1.0, 1.5, 2.5)
    assert sl == pytest.approx(98.5)
    assert tp == pytest.approx(102.5)


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


def test_open_position_skips_existing():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert len(dh.exchange.orders) == 1
    assert len(tm.positions) == 1


def test_open_position_concurrent_single_entry():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await asyncio.gather(
            tm.open_position('BTCUSDT', 'buy', 100, {}),
            tm.open_position('BTCUSDT', 'buy', 100, {}),
        )

    import asyncio
    asyncio.run(run())

    assert len(tm.positions) == 1


def test_open_position_many_concurrent_single_entry():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await asyncio.gather(
            *[
                tm.open_position('BTCUSDT', 'buy', 100, {})
                for _ in range(5)
            ]
        )

    import asyncio
    asyncio.run(run())

    assert len(tm.positions) == 1


def test_open_position_failed_order_not_recorded():
    dh = DummyDataHandler(fail_order=True)
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert len(tm.positions) == 0


def test_is_data_fresh():
    fresh_dh = DummyDataHandler(fresh=True)
    stale_dh = DummyDataHandler(fresh=False)

    import asyncio
    assert asyncio.run(fresh_dh.is_data_fresh('BTCUSDT')) is True
    assert asyncio.run(stale_dh.is_data_fresh('BTCUSDT')) is False

