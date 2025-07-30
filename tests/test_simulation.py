import asyncio
import logging
import sys
import types
import pandas as pd
import pytest
from bot.config import BotConfig

# Stub heavy dependencies
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch
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
utils = types.ModuleType('utils')
class DummyTL:
    def __init__(self, *a, **k):
        pass
    async def send_telegram_message(self, *a, **k):
        pass
    @classmethod
    async def shutdown(cls):
        pass
utils.TelegramLogger = DummyTL
utils.logger = logging.getLogger('test')
async def _cde(*a, **kw):
    return False
utils.check_dataframe_empty = _cde
utils.check_dataframe_empty_async = _cde
utils.is_cuda_available = lambda: False
async def _safe_api_call(exchange, method: str, *args, **kwargs):
    return await getattr(exchange, method)(*args, **kwargs)
utils.safe_api_call = _safe_api_call
sys.modules['utils'] = utils
joblib_mod = types.ModuleType('joblib')
joblib_mod.dump = lambda *a, **k: None
joblib_mod.load = lambda *a, **k: {}
sys.modules.setdefault('joblib', joblib_mod)
import os
os.environ['TEST_MODE'] = '1'

sys.modules.pop('trade_manager', None)
sys.modules.pop('simulation', None)
from bot.trade_manager import TradeManager
from bot.simulation import HistoricalSimulator

class DummyExchange:
    def __init__(self):
        self.orders = []
    async def fetch_balance(self):
        return {'total': {'USDT': 1000}}
    async def create_order(self, symbol, typ, side, amount, price, params):
        self.orders.append({'method': 'create_order', 'price': price})
        return {'id': '1'}
    async def create_order_with_take_profit_and_stop_loss(self, symbol, typ, side, amount, price, tp, sl, params):
        self.orders.append({'method': 'create_order_with_tp_sl', 'tp': tp, 'sl': sl})
        return {'id': '2'}

class DummyDataHandler:
    def __init__(self):
        self.exchange = DummyExchange()
        self.usdt_pairs = ['BTCUSDT']
        idx = pd.date_range('2020-01-01', periods=3, freq='1min', tz='UTC')
        df = pd.DataFrame({
            'open': [100, 101, 101],
            'high': [100.5, 101.5, 101],
            'low': [99.5, 100.5, 98],
            'close': [100, 101, 98],
            'atr': [1.0, 1.0, 1.0],
        }, index=idx)
        df['symbol'] = 'BTCUSDT'
        self.history = df.set_index('symbol', append=True).swaplevel(0,1)
        self.history.index.names = ['symbol', 'timestamp']
        self.ohlcv = self.history.iloc[:1]
        self.indicators = {'BTCUSDT': types.SimpleNamespace(atr=pd.Series([1.0]*3, index=idx), df=df)}
        async def _opt(symbol):
            return {}
        self.parameter_optimizer = types.SimpleNamespace(optimize=_opt)
        self.funding_rates = {'BTCUSDT': 0.0}
        self.open_interest = {'BTCUSDT': 0.0}
        self.config = BotConfig(cache_dir='/tmp')
    async def get_atr(self, symbol: str) -> float:
        return 1.0
    async def is_data_fresh(self, symbol: str, timeframe: str = 'primary', max_delay: float = 60) -> bool:
        return True
    async def synchronize_and_update(self, symbol, df, fr, oi, ob, timeframe='primary'):
        self.ohlcv = pd.concat([self.ohlcv, df], ignore_index=False).sort_index()

@pytest.mark.asyncio
async def test_simulator_trailing_stop():
    dh = DummyDataHandler()
    cfg = BotConfig(cache_dir='/tmp', trailing_stop_percentage=1.0, trailing_stop_coeff=0.0, trailing_stop_multiplier=1.0)
    tm = TradeManager(cfg, dh, None, None, None)

    first = {'n': 0}
    async def fake_eval(symbol):
        first['n'] += 1
        return 'buy' if first['n'] == 1 else None
    tm.evaluate_signal = fake_eval
    async def fake_close(symbol, price, reason='Manual'):
        if (
            'symbol' in tm.positions.index.names
            and symbol in tm.positions.index.get_level_values('symbol')
        ):
            idx = tm.positions.loc[
                tm.positions.index.get_level_values('symbol') == symbol
            ].index[0]
            tm.positions = tm.positions.drop(idx)
        dh.exchange.orders.append({'method': 'close', 'price': price})
    tm.close_position = fake_close
    async def _noop(*_a, **_k):
        pass
    tm.check_exit_signal = _noop

    sim = HistoricalSimulator(dh, tm)
    start = dh.history.index.get_level_values('timestamp').min()
    end = dh.history.index.get_level_values('timestamp').max()
    await sim.run(start, end, speed=1000)
    assert tm.positions.empty
    assert len(dh.exchange.orders) >= 2
