import asyncio
import contextlib
import logging
import sys
import types
import os
import pandas as pd
import pytest
import tempfile
from bot.config import BotConfig

# Stub heavy dependencies before importing the trade manager
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

utils_stub = types.ModuleType('utils')
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
async def _cde_stub(*a, **kw):
    return False
utils_stub.check_dataframe_empty = _cde_stub
utils_stub.check_dataframe_empty_async = _cde_stub
utils_stub.is_cuda_available = lambda: False
async def _safe_api_call(exchange, method: str, *args, **kwargs):
    return await getattr(exchange, method)(*args, **kwargs)
utils_stub.safe_api_call = _safe_api_call
sys.modules['utils'] = utils_stub
sys.modules.pop('trade_manager', None)
joblib_mod = types.ModuleType('joblib')
joblib_mod.dump = lambda *a, **k: None
joblib_mod.load = lambda *a, **k: {}
sys.modules.setdefault('joblib', joblib_mod)


@pytest.fixture(scope="module", autouse=True)
def _set_test_mode():
    mp = pytest.MonkeyPatch()
    mp.setenv("TEST_MODE", "1")
    yield
    mp.undo()


@pytest.fixture(scope="module", autouse=True)
def _import_trade_manager(_set_test_mode):
    global trade_manager, TradeManager
    from bot import trade_manager as tm
    from bot.trade_manager import TradeManager as TM
    trade_manager = tm
    TradeManager = TM
    yield

@pytest.fixture(scope="module", autouse=True)
def _cleanup_telegram_logger(_import_trade_manager):
    yield
    asyncio.run(trade_manager.TelegramLogger.shutdown())

class DummyExchange:
    def __init__(self):
        self.orders = []

class DummyDataHandler:
    def __init__(self):
        self.exchange = DummyExchange()
        self.usdt_pairs = ['BTCUSDT']
        idx = pd.MultiIndex.from_tuples([
            ('BTCUSDT', pd.Timestamp('2020-01-01'))
        ], names=['symbol', 'timestamp'])
        self.ohlcv = pd.DataFrame({'close': [100], 'atr': [1.0]}, index=idx)
        self.indicators = {}
        self.parameter_optimizer = types.SimpleNamespace(optimize=lambda s: {})

    async def get_atr(self, symbol: str) -> float:
        return 1.0
    async def is_data_fresh(self, symbol: str, timeframe: str = 'primary', max_delay: float = 60) -> bool:
        return True

class DummyModelBuilder:
    def __init__(self):
        self.predictive_models = {'BTCUSDT': object()}
    async def retrain_symbol(self, symbol):
        pass


def make_config():
    return BotConfig(
        cache_dir=tempfile.mkdtemp(),
        check_interval=1,
        performance_window=1,
        order_retry_delay=0,
        reversal_margin=0.05,
    )

@pytest.mark.asyncio
async def test_monitor_performance_recovery(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)
    tm.returns_by_symbol['BTCUSDT'].append((pd.Timestamp.now(tz='UTC').timestamp(), 0.1))

    call = {'n': 0}
    orig_now = pd.Timestamp.now
    def fake_now(*a, **k):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')
        return orig_now(*a, **k)
    monkeypatch.setattr(pd.Timestamp, 'now', fake_now)

    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(trade_manager.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(tm.monitor_performance())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2

@pytest.mark.asyncio
async def test_manage_positions_recovery(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)
    idx = pd.MultiIndex.from_tuples([
        ('BTCUSDT', pd.Timestamp('2020-01-01'))
    ], names=['symbol', 'timestamp'])
    tm.positions = pd.DataFrame({
        'side': ['buy'],
        'size': [1],
        'entry_price': [100],
        'tp_multiplier': [2],
        'sl_multiplier': [1],
        'stop_loss_price': [99],
        'highest_price': [100],
        'lowest_price': [0],
        'breakeven_triggered': [False],
    }, index=idx)

    call = {'n': 0}
    async def fake_check(symbol, price):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')
    monkeypatch.setattr(tm, 'check_trailing_stop', fake_check)
    monkeypatch.setattr(tm, 'check_stop_loss_take_profit', lambda *a, **k: None)
    monkeypatch.setattr(tm, 'check_exit_signal', lambda *a, **k: None)

    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(trade_manager.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(tm.manage_positions())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2

@pytest.mark.asyncio
async def test_process_symbol_recovery(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)
    monkeypatch.setattr(tm, 'open_position', lambda *a, **k: None)

    call = {'n': 0}
    async def fake_eval(symbol):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')
        return 'buy'
    monkeypatch.setattr(tm, 'evaluate_signal', fake_eval)

    orig_sleep = asyncio.sleep
    async def fast_sleep(_):
        await orig_sleep(0)
    monkeypatch.setattr(trade_manager.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(tm.process_symbol('BTCUSDT'))
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2


@pytest.mark.asyncio
async def test_process_symbol_data_fresh_error(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)

    call = {'n': 0}

    async def fake_eval(symbol):
        call['n'] += 1
        return 'buy'

    async def raise_conn_error(*a, **k):
        raise ConnectionError('offline')

    monkeypatch.setattr(tm, 'evaluate_signal', fake_eval)
    monkeypatch.setattr(dh, 'is_data_fresh', raise_conn_error)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(trade_manager.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(tm.process_symbol('BTCUSDT'))
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2
    assert dh.exchange.orders == []

sys.modules.pop('utils', None)
os.environ.pop("TEST_MODE", None)
