import asyncio
import contextlib
import logging
import sys
import types
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
def _retry(max_attempts, delay_fn):
    def decorator(func):
        return func
    return decorator
utils_stub.retry = _retry
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
    import bot.trade_manager.core as tm
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
        check_interval=1.0,
        performance_window=1,
        order_retry_delay=0.0,
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
        'position': [1],
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
async def test_run_cancels_pending_tasks_on_failure(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)

    failure_started = asyncio.Event()
    cancel_events = [asyncio.Event(), asyncio.Event()]

    async def failing_task():
        failure_started.set()
        await asyncio.sleep(0)
        raise RuntimeError('boom')

    def make_long_task(idx: int):
        async def _runner():
            try:
                while True:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                cancel_events[idx].set()
                raise

        return _runner

    monkeypatch.setattr(tm, 'monitor_performance', failing_task)
    monkeypatch.setattr(tm, 'manage_positions', make_long_task(0))
    monkeypatch.setattr(tm, 'ranked_signal_loop', make_long_task(1))

    run_task = asyncio.create_task(tm.run())

    await asyncio.wait_for(failure_started.wait(), timeout=1)

    with pytest.raises(trade_manager.TradeManagerTaskError):
        await run_task

    for event in cancel_events:
        await asyncio.wait_for(event.wait(), timeout=1)


@pytest.mark.asyncio
async def test_open_and_close_short_loop():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)

    async def fake_size(*a, **k):
        return 1.0

    async def fake_place(*a, **k):
        return {"id": "1"}

    tm.calculate_position_size = fake_size
    tm.place_order = fake_place

    await tm.open_position('BTCUSDT', 'sell', 100, {})
    await tm.close_position('BTCUSDT', 90, 'Manual')

    assert len(tm.returns_by_symbol['BTCUSDT']) == 1
    assert tm.returns_by_symbol['BTCUSDT'][0][1] > 0


@pytest.mark.asyncio
async def test_ranked_signal_loop_recovery(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, DummyModelBuilder(), None, None)

    call = {'n': 0}

    async def fake_execute():
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('boom')
        raise asyncio.CancelledError()

    monkeypatch.setattr(tm, 'execute_top_signals_once', fake_execute)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(trade_manager.asyncio, 'sleep', fast_sleep)

    task = asyncio.create_task(tm.ranked_signal_loop())

    with pytest.raises(asyncio.CancelledError):
        await task

    assert call['n'] >= 2

sys.modules.pop('utils', None)
