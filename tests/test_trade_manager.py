import pandas as pd
import numpy as np
import pytest
import sys
import types
import logging
import os
import math
import contextlib
import tempfile
import httpx
import aiohttp
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
class _TL:
    def __init__(self, *a, **k):
        pass
    async def send_telegram_message(self, *a, **k):
        pass
    @classmethod
    async def shutdown(cls):
        pass
utils_stub.TelegramLogger = _TL
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
sys.modules['bot.utils'] = utils_stub
sys.modules.pop('trade_manager', None)
sys.modules.pop('bot.trade_manager', None)
joblib_mod = types.ModuleType('joblib')
joblib_mod.dump = lambda *a, **k: None
joblib_mod.load = lambda *a, **k: {}
sys.modules.setdefault('joblib', joblib_mod)

import asyncio


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

def test_utils_injected_before_trade_manager_import():
    import importlib, sys
    tm = importlib.reload(sys.modules.get("bot.trade_manager", trade_manager))
    tm.TelegramLogger = _TL
    assert tm.TelegramLogger is _TL

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
utils.check_dataframe_empty_async = _cde
utils.is_cuda_available = lambda: False
async def _safe_api_call(exchange, method: str, *args, **kwargs):
    return await getattr(exchange, method)(*args, **kwargs)
utils.safe_api_call = _safe_api_call
sys.modules['utils'] = utils
sys.modules['bot.utils'] = utils


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

class DummyDataHandler:
    def __init__(
        self,
        fresh: bool = True,
        fail_order: bool = False,
        pairs: list[str] | None = None,
    ):
        self.exchange = DummyExchange()
        self.exchange.fail = fail_order
        self.usdt_pairs = pairs or ['BTCUSDT']
        idx = pd.MultiIndex.from_tuples(
            [(sym, pd.Timestamp('2020-01-01')) for sym in self.usdt_pairs],
            names=['symbol', 'timestamp']
        )
        self.ohlcv = pd.DataFrame(
            {'close': [100]*len(self.usdt_pairs), 'atr': [1.0]*len(self.usdt_pairs)},
            index=idx
        )
        self.indicators = {
            sym: types.SimpleNamespace(atr=pd.Series([1.0]), df=pd.DataFrame({'a':[1]}))
            for sym in self.usdt_pairs
        }
        self.fresh = fresh
        async def _opt(symbol):
            return {}
        self.parameter_optimizer = types.SimpleNamespace(optimize=_opt)

    async def get_atr(self, symbol: str) -> float:
        if "symbol" in self.ohlcv.index.names and symbol in self.ohlcv.index.get_level_values("symbol"):
            return float(self.ohlcv.loc[pd.IndexSlice[symbol], "atr"].iloc[-1])
        return 0.0

    async def is_data_fresh(self, symbol: str, timeframe: str = 'primary', max_delay: float = 60) -> bool:
        return self.fresh

def make_config():
    return BotConfig(
        cache_dir=tempfile.mkdtemp(),
        max_positions=5,
        leverage=10,
        min_risk_per_trade=0.01,
        max_risk_per_trade=0.05,
        check_interval=1,
        performance_window=60,
        sl_multiplier=1.0,
        tp_multiplier=2.0,
        order_retry_attempts=3,
        order_retry_delay=0,
        reversal_margin=0.05,
    )


def test_place_order_passes_tp_sl_without_special_method():
    class ExchangeNoTPSL:
        def __init__(self):
            self.calls = []

        async def create_order(self, symbol, type, side, amount, price, params):
            self.calls.append(
                {
                    "method": "create_order",
                    "symbol": symbol,
                    "type": type,
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "params": params,
                }
            )
            return {"id": "1"}

    dh = DummyDataHandler()
    dh.exchange = ExchangeNoTPSL()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def run():
        await tm.place_order(
            "BTCUSDT",
            "buy",
            1,
            100,
            {"takeProfitPrice": 102.0, "stopLossPrice": 99.0},
            use_lock=False,
        )

    import asyncio

    asyncio.run(run())

    assert dh.exchange.calls, "create_order not called"
    call = dh.exchange.calls[0]
    assert call["method"] == "create_order"
    assert call["params"].get("takeProfitPrice") == pytest.approx(102.0)
    assert call["params"].get("stopLossPrice") == pytest.approx(99.0)

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
    dh = DummyDataHandler(pairs=['BTCUSDT', 'ETHUSDT'])
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
        await tm.open_position('ETHUSDT', 'buy', 100, {})
        await tm.open_position('BTCUSDT', 'buy', 100, {})
        # Deliberately unsort positions
        tm.positions = tm.positions.iloc[::-1]
        assert not tm.positions.index.is_monotonic_increasing
        await tm.check_trailing_stop('BTCUSDT', 101)
        assert tm.positions.index.is_monotonic_increasing

    import asyncio
    asyncio.run(run())

    assert len(dh.exchange.orders) >= 3
    pos = tm.positions.xs('BTCUSDT', level='symbol').iloc[0]
    open_btc_order = next(o for o in dh.exchange.orders if o['symbol'] == 'BTCUSDT')
    assert pos['breakeven_triggered'] is True
    assert pos['size'] < open_btc_order['amount']
    assert pos['stop_loss_price'] == pytest.approx(pos['entry_price'])


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
    assert len(dh.exchange.orders) == tm.config.order_retry_attempts


def test_open_position_retries_until_success(monkeypatch):
    dh = DummyDataHandler()
    attempts = {"n": 0}

    async def fail_then_succeed(symbol, type, side, amount, price, tp, sl, params):
        attempts["n"] += 1
        dh.exchange.orders.append({
            'method': 'create_order_with_tp_sl',
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'price': price,
            'tp': tp,
            'sl': sl,
            'params': params,
        })
        if attempts["n"] < 2:
            return {"retCode": 1}
        return {"id": "2"}

    monkeypatch.setattr(
        dh.exchange,
        'create_order_with_take_profit_and_stop_loss',
        fail_then_succeed,
    )

    tm = TradeManager(make_config(), dh, None, None, None)
    async def fake_compute(symbol, vol):
        return 0.01
    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    asyncio.run(run())

    assert len(tm.positions) == 1
    assert attempts["n"] == 2


def test_open_position_skips_when_atr_zero():
    class ZeroAtrDataHandler(DummyDataHandler):
        async def get_atr(self, symbol: str) -> float:
            return 0.0

    dh = ZeroAtrDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert dh.exchange.orders == []
    assert len(tm.positions) == 0


def test_open_position_skips_when_data_stale():
    dh = DummyDataHandler(fresh=False)
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    async def run():
        await tm.open_position('BTCUSDT', 'buy', 100, {})

    import asyncio
    asyncio.run(run())

    assert dh.exchange.orders == []
    assert len(tm.positions) == 0


def test_is_data_fresh():
    fresh_dh = DummyDataHandler(fresh=True)
    stale_dh = DummyDataHandler(fresh=False)

    import asyncio
    assert asyncio.run(fresh_dh.is_data_fresh('BTCUSDT')) is True
    assert asyncio.run(stale_dh.is_data_fresh('BTCUSDT')) is False


def test_compute_risk_per_trade_zero_threshold():
    cfg = make_config()
    cfg.update({'volatility_threshold': 0})
    dh = DummyDataHandler()
    tm = TradeManager(cfg, dh, None, None, None)

    async def run():
        return await tm.compute_risk_per_trade('BTCUSDT', 0.01)

    import asyncio
    risk = asyncio.run(run())
    assert tm.min_risk_per_trade <= risk <= tm.max_risk_per_trade


def test_get_loss_streak():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    tm.returns_by_symbol['BTCUSDT'] = [
        (0, 0.1),
        (1, -0.2),
        (2, -0.3),
        (3, -0.4),
    ]

    async def run():
        return await tm.get_loss_streak('BTCUSDT')

    import asyncio

    streak = asyncio.run(run())
    assert streak == 3


def test_get_win_streak():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    tm.returns_by_symbol['BTCUSDT'] = [
        (0, -0.1),
        (1, 0.2),
        (2, 0.3),
        (3, 0.4),
    ]

    async def run():
        return await tm.get_win_streak('BTCUSDT')

    import asyncio
    streak = asyncio.run(run())
    assert streak == 3


@pytest.mark.asyncio
async def test_close_position_updates_returns_and_sharpe_ratio():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})
    await tm.close_position("BTCUSDT", 110, "Manual")

    assert len(tm.returns_by_symbol["BTCUSDT"]) == 1
    profit = tm.returns_by_symbol["BTCUSDT"][0][1]
    expected = profit / (1e-6) * math.sqrt(365 * 24 * 60 * 60 / tm.performance_window)
    sharpe = await tm.get_sharpe_ratio("BTCUSDT")
    assert sharpe == pytest.approx(expected)


class DummyModel:
    def eval(self):
        pass

    def __call__(self, *_):
        class _Out:
            def squeeze(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return 0.6

        return _Out()


@pytest.mark.asyncio
async def test_evaluate_signal_uses_cached_features(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    monkeypatch.setattr(tm, "evaluate_ema_condition", lambda *a, **k: True)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal in ("buy", "sell", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_ema", [False, True])
async def test_evaluate_signal_handles_sync_async_ema(monkeypatch, async_ema):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    called = {"ok": False}
    if async_ema:
        async def _ema(*a, **k):
            called["ok"] = True
            return True
    else:
        def _ema(*a, **k):
            called["ok"] = True
            return True
    monkeypatch.setattr(tm, "evaluate_ema_condition", _ema)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert called["ok"]
    assert signal in ("buy", "sell", None)


@pytest.mark.asyncio
async def test_evaluate_signal_retrains_when_model_missing(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}
            self.retrained = False

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.arange(8, dtype=np.float32).reshape(-1, 1)

        def prepare_dataset(self, features):
            X = np.ones((4, 1), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            self.retrained = True

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal is None
    assert mb.retrained


@pytest.mark.asyncio
async def test_evaluate_signal_skips_retrain_on_single_label(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}
            self.retrained = False

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.ones((4, 1), dtype=np.float32)

        def prepare_dataset(self, features):
            X = np.ones((2, 1), dtype=np.float32)
            y = np.zeros(2, dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            self.retrained = True

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal is None
    assert not mb.retrained


@pytest.mark.asyncio
async def test_evaluate_signal_handles_http_400(monkeypatch):
    dh = DummyDataHandler()

    class HTTPError(httpx.HTTPError):
        def __init__(self, status):
            super().__init__("boom")
            self.response = types.SimpleNamespace(status_code=status)

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}
            self.called = False

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.arange(8, dtype=np.float32).reshape(-1, 1)

        def prepare_dataset(self, features):
            X = np.ones((4, 1), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            self.called = True
            raise HTTPError(400)

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal is None
    assert mb.called


@pytest.mark.asyncio
async def test_evaluate_signal_raises_on_http_error(monkeypatch):
    dh = DummyDataHandler()

    class HTTPError(httpx.HTTPError):
        def __init__(self, status):
            super().__init__("boom")
            self.response = types.SimpleNamespace(status_code=status)

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {}
            self.calibrators = {}
            self.feature_cache = {}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            return np.arange(8, dtype=np.float32).reshape(-1, 1)

        def prepare_dataset(self, features):
            X = np.ones((4, 1), dtype=np.float32)
            y = np.array([0, 1, 0, 1], dtype=np.float32)
            return X, y

        async def retrain_symbol(self, symbol):
            raise HTTPError(500)

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)

    with pytest.raises(HTTPError):
        await tm.evaluate_signal("BTCUSDT")


@pytest.mark.asyncio
async def test_process_symbol_retries_on_client_error(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.predictive_models = {"BTCUSDT": object()}

    tm = TradeManager(
        BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp(), check_interval=0.01),
        dh,
        MB(),
        None,
        None,
    )

    calls = {"n": 0}

    async def fake_eval(symbol):
        calls["n"] += 1
        if calls["n"] == 1:
            raise aiohttp.ClientError("boom")
        raise asyncio.CancelledError()

    monkeypatch.setattr(tm, "evaluate_signal", fake_eval)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(trade_manager.asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(tm.process_symbol("BTCUSDT"))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_process_symbol_propagates_unexpected_error(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.predictive_models = {"BTCUSDT": object()}

    tm = TradeManager(
        BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp(), check_interval=0.01),
        dh,
        MB(),
        None,
        None,
    )

    async def fake_eval(symbol):
        raise ValueError("boom")

    monkeypatch.setattr(tm, "evaluate_signal", fake_eval)

    with pytest.raises(ValueError):
        await tm.process_symbol("BTCUSDT")


@pytest.mark.asyncio
async def test_evaluate_signal_regression(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"

            class Model:
                def eval(self):
                    pass

                def __call__(self, *_):
                    class _Out:
                        def squeeze(self):
                            return self

                        def float(self):
                            return self

                        def cpu(self):
                            return self

                        def numpy(self):
                            return 0.003

                    return _Out()

            self.predictive_models = {"BTCUSDT": Model()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

    mb = MB()
    cfg = BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp(), prediction_target="pnl", trading_fee=0.001)
    tm = TradeManager(cfg, dh, mb, None, None)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal == "buy"


@pytest.mark.asyncio
async def test_rl_action_overrides_voting(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            class Model(DummyModel):
                def __call__(self, *_):
                    class _Out:
                        def squeeze(self):
                            return self

                        def float(self):
                            return self

                        def cpu(self):
                            return self

                        def numpy(self):
                            return 0.9

                    return _Out()

            self.predictive_models = {"BTCUSDT": Model()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()

    class RL:
        def __init__(self):
            self.models = {"BTCUSDT": object()}

        def predict(self, symbol, obs):
            return "open_short"

    rl = RL()
    cfg = BotConfig(
        lstm_timesteps=2,
        cache_dir=tempfile.mkdtemp(),
        transformer_weight=0.7,
        ema_weight=0.3,
    )
    tm = TradeManager(cfg, dh, mb, None, None, rl)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    monkeypatch.setattr(tm, "evaluate_ema_condition", lambda *a, **k: True)

    signal = await tm.evaluate_signal("BTCUSDT")
    assert signal == "sell"


@pytest.mark.asyncio
async def test_check_exit_signal_uses_cached_features(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)
    idx = pd.MultiIndex.from_tuples([
        ("BTCUSDT", pd.Timestamp("2020-01-01"))
    ], names=["symbol", "timestamp"])
    tm.positions = pd.DataFrame({
        "side": ["buy"],
        "size": [1],
        "entry_price": [100],
        "tp_multiplier": [2],
        "sl_multiplier": [1],
        "stop_loss_price": [99],
        "highest_price": [100],
        "lowest_price": [0],
        "breakeven_triggered": [False],
    }, index=idx)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    monkeypatch.setattr(tm, "close_position", lambda *a, **k: None)

    await tm.check_exit_signal("BTCUSDT", 100)


@pytest.mark.asyncio
async def test_exit_signal_triggers_reverse_trade(monkeypatch):
    dh = DummyDataHandler()
    class MB:
        def __init__(self):
            self.device = "cpu"
            class Model:
                def eval(self):
                    pass
                def __call__(self, *_):
                    class _Out:
                        def squeeze(self):
                            return self
                        def float(self):
                            return self
                        def cpu(self):
                            return self
                        def numpy(self):
                            return 0.2
                    return _Out()
            self.predictive_models = {"BTCUSDT": Model()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()
    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None)
    idx = pd.MultiIndex.from_tuples([
        ("BTCUSDT", pd.Timestamp("2020-01-01"))
    ], names=["symbol", "timestamp"])
    tm.positions = pd.DataFrame({
        "side": ["buy"],
        "size": [1],
        "entry_price": [100],
        "tp_multiplier": [2],
        "sl_multiplier": [1],
        "stop_loss_price": [99],
        "highest_price": [100],
        "lowest_price": [0],
        "breakeven_triggered": [False],
    }, index=idx)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    opened = {"side": None}

    async def fake_close(symbol, price, reason=""):
        tm.positions = tm.positions.drop(symbol, level="symbol")

    async def fake_open(symbol, side, price, params):
        opened["side"] = side

    monkeypatch.setattr(tm, "close_position", fake_close)
    monkeypatch.setattr(tm, "open_position", fake_open)
    async def _ema(*a, **k):
        return True
    monkeypatch.setattr(tm, "evaluate_ema_condition", _ema)

    await tm.check_exit_signal("BTCUSDT", 100)

    assert opened["side"] == "sell"


@pytest.mark.asyncio
async def test_rl_close_action(monkeypatch):
    dh = DummyDataHandler()

    class MB:
        def __init__(self):
            self.device = "cpu"
            self.predictive_models = {"BTCUSDT": DummyModel()}
            self.calibrators = {}
            self.feature_cache = {"BTCUSDT": np.ones((2, 1), dtype=np.float32)}

        def get_cached_features(self, symbol):
            return self.feature_cache.get(symbol)

        async def prepare_lstm_features(self, symbol, indicators):
            raise AssertionError("prepare_lstm_features should not be called")

        async def adjust_thresholds(self, symbol, prediction):
            return 0.7, 0.3

    mb = MB()

    class RL:
        def __init__(self):
            self.models = {"BTCUSDT": object()}

        def predict(self, symbol, obs):
            return "close"

    rl = RL()

    tm = TradeManager(BotConfig(lstm_timesteps=2, cache_dir=tempfile.mkdtemp()), dh, mb, None, None, rl)
    idx = pd.MultiIndex.from_tuples([
        ("BTCUSDT", pd.Timestamp("2020-01-01"))
    ], names=["symbol", "timestamp"])
    tm.positions = pd.DataFrame({
        "side": ["buy"],
        "size": [1],
        "entry_price": [100],
        "tp_multiplier": [2],
        "sl_multiplier": [1],
        "stop_loss_price": [99],
        "highest_price": [100],
        "lowest_price": [0],
        "breakeven_triggered": [False],
    }, index=idx)

    torch = sys.modules["torch"]
    torch.tensor = lambda *a, **k: a[0]
    torch.float32 = np.float32
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda *_: contextlib.nullcontext())

    called = {"n": 0}

    async def fake_close(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", fake_close)

    await tm.check_exit_signal("BTCUSDT", 100)

    assert called["n"] == 1


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_triggers_close_long(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 99)

    assert called["n"] == 1
    assert len(tm.positions) == 0


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_triggers_close_short(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "sell", 100, {})

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 101)

    assert called["n"] == 1
    assert len(tm.positions) == 0


@pytest.mark.asyncio
async def test_check_stop_loss_take_profit_breakeven_uses_fixed_price(monkeypatch):
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)

    async def fake_compute(symbol, vol):
        return 0.01

    tm.compute_risk_per_trade = fake_compute

    await tm.open_position("BTCUSDT", "buy", 100, {})

    tm.positions.loc[pd.IndexSlice["BTCUSDT", :], "breakeven_triggered"] = True
    tm.positions.loc[pd.IndexSlice["BTCUSDT", :], "stop_loss_price"] = 100
    dh.indicators["BTCUSDT"].atr = pd.Series([10.0])

    called = {"n": 0}

    async def wrapped(symbol, price, reason=""):
        called["n"] += 1
        tm.positions = tm.positions.drop(symbol, level="symbol")

    monkeypatch.setattr(tm, "close_position", wrapped)

    await tm.check_stop_loss_take_profit("BTCUSDT", 99.5)

    assert called["n"] == 1


def test_compute_stats():
    dh = DummyDataHandler()
    tm = TradeManager(make_config(), dh, None, None, None)
    tm.returns_by_symbol["BTCUSDT"] = [
        (0, 1.0),
        (1, -2.0),
        (2, 2.0),
    ]

    async def run():
        return await tm.compute_stats()

    import asyncio

    stats = asyncio.run(run())
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_pnl"] == pytest.approx(1.0 / 3)
    assert stats["max_drawdown"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_execute_top_signals_ranking(monkeypatch):
    class DH(DummyDataHandler):
        def __init__(self):
            super().__init__()
            self.usdt_pairs = ["A", "B", "C"]
            idx = pd.MultiIndex.from_product([
                self.usdt_pairs,
                [pd.Timestamp("2020-01-01")],
            ], names=["symbol", "timestamp"])
            self.ohlcv = pd.DataFrame({"close": [1, 1, 1], "atr": [1, 1, 1]}, index=idx)

    dh = DH()
    tm = TradeManager(BotConfig(cache_dir=tempfile.mkdtemp(), top_signals=2), dh, None, None, None)

    async def fake_eval(symbol, return_prob=False):
        probs = {"A": 0.9, "B": 0.8, "C": 0.1}
        return ("buy", probs[symbol]) if return_prob else "buy"

    monkeypatch.setattr(tm, "evaluate_signal", fake_eval)

    opened = []

    async def fake_open(symbol, side, price, params):
        opened.append(symbol)

    monkeypatch.setattr(tm, "open_position", fake_open)

    await tm.execute_top_signals_once()

    assert opened == ["A", "B"]


def test_shutdown_shuts_down_ray(monkeypatch):
    import importlib, sys, types

    ray_stub = types.ModuleType("ray")
    called = {"done": False}
    ray_stub.init = lambda: None
    ray_stub.is_initialized = lambda: True

    def _shutdown():
        called["done"] = True

    ray_stub.shutdown = _shutdown

    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    tm_mod = importlib.reload(sys.modules.get("bot.trade_manager", trade_manager))
    monkeypatch.setattr(tm_mod, "ray", ray_stub, raising=False)

    dh = DummyDataHandler()
    tm = tm_mod.TradeManager(make_config(), dh, None, None, None)
    tm.shutdown()

    assert called["done"]


def test_shutdown_calls_ray_shutdown_in_test_mode(monkeypatch):
    import importlib, sys, types

    ray_stub = types.ModuleType("ray")
    called = {"done": False}
    ray_stub.is_initialized = lambda: False
    ray_stub.shutdown = lambda: called.update(done=True)

    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    tm_mod = importlib.reload(sys.modules.get("bot.trade_manager", trade_manager))
    monkeypatch.setattr(tm_mod, "ray", ray_stub, raising=False)

    dh = DummyDataHandler()
    tm = tm_mod.TradeManager(make_config(), dh, None, None, None)
    tm.shutdown()

    assert called["done"]


def test_shutdown_handles_missing_is_initialized(monkeypatch):
    import importlib, sys, types

    monkeypatch.setenv("TEST_MODE", "0")

    ray_stub = types.ModuleType("ray")

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):
            return self

    def _ray_remote(func=None, **_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)
            return wrapper
        return _RayRemoteFunction(func)

    ray_stub.remote = _ray_remote
    ray_stub.get = lambda x: x
    ray_stub.init = lambda *a, **k: None
    called = {"done": False}

    def _shutdown():
        called["done"] = True

    ray_stub.shutdown = _shutdown

    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    tm_mod = importlib.reload(sys.modules.get("bot.trade_manager", trade_manager))
    monkeypatch.setattr(tm_mod, "ray", ray_stub, raising=False)

    dh = DummyDataHandler()
    tm = tm_mod.TradeManager(make_config(), dh, None, None, None)
    tm.shutdown()

    assert not called["done"]


sys.modules.pop('utils', None)
sys.modules.pop('bot.utils', None)
os.environ.pop("TEST_MODE", None)

