import os
import sys

import numpy as np
import pandas as pd
import types
import pytest
import importlib.util
import contextlib
from bot.config import BotConfig
import asyncio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import tempfile

pytestmark = pytest.mark.requires_sklearn
from collections import deque

try:  # require functional torch installation for these tests
    import torch
    import torch.nn  # noqa: F401
except Exception:
    pytest.skip('torch not available', allow_module_level=True)

# Provide dummy stable_baselines3 if missing
if "stable_baselines3" not in sys.modules:
    sb3 = types.ModuleType("stable_baselines3")

    class DummyModel:
        def __init__(self, *a, **kw):
            pass

        def learn(self, *a, **kw):
            return self

        def predict(self, obs, deterministic=True):
            return np.array([1]), None

    sb3.PPO = DummyModel
    sb3.DQN = DummyModel
    common = types.ModuleType("stable_baselines3.common")
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")

    class DummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]

    vec_env.DummyVecEnv = DummyVecEnv
    common.vec_env = vec_env
    sb3.common = common
    sys.modules["stable_baselines3"] = sb3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.vec_env"] = vec_env

from bot import model_builder
from bot.model_builder import ModelBuilder, _train_model_remote

class DummyIndicators:
    def __init__(self, length):
        base = np.arange(length, dtype=float)
        self.ema30 = pd.Series(base)
        self.ema100 = pd.Series(base + 1)
        self.ema200 = pd.Series(base + 2)
        self.rsi = pd.Series(base + 3)
        self.adx = pd.Series(base + 4)
        self.macd = pd.Series(base + 5)
        self.atr = pd.Series(base + 6)

class DummyDataHandler:
    def __init__(self, df):
        self.ohlcv = df
        n = len(df)
        self.funding_rates = {"BTCUSDT": np.linspace(0.1, 0.2, n)}
        self.open_interest = {"BTCUSDT": np.linspace(0.2, 0.3, n)}
        self.open_interest_change = {"BTCUSDT": np.linspace(0.0, 0.1, n)}
        self.usdt_pairs = ["BTCUSDT"]

class DummyTradeManager:
    pass

def create_model_builder(df):
    tmpdir = tempfile.TemporaryDirectory()
    config = BotConfig(
        cache_dir=tmpdir.name,
        min_data_length=len(df),
        lstm_timesteps=2,
        lstm_batch_size=2,
        model_type="tft",
    )
    data_handler = DummyDataHandler(df)
    trade_manager = DummyTradeManager()
    mb = ModelBuilder(config, data_handler, trade_manager)
    mb._tmpdir = tmpdir
    return mb

def make_df(length=5):
    idx = pd.date_range("2020-01-01", periods=length, freq="min")
    df = pd.DataFrame({
        "close": np.linspace(1, 2, length),
        "open": np.linspace(1, 2, length),
        "high": np.linspace(1, 2, length),
        "low": np.linspace(1, 2, length),
        "volume": np.linspace(1, 2, length),
    }, index=idx)
    df["symbol"] = "BTCUSDT"
    df = df.set_index(["symbol", df.index])
    return df

def test_prepare_lstm_features_shape():
    df = make_df()
    mb = create_model_builder(df)
    indicators = DummyIndicators(len(df))
    features = asyncio.run(mb.prepare_lstm_features("BTCUSDT", indicators))
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(df), 8)


def test_prepare_lstm_features_with_short_indicators():
    df = make_df()
    mb = create_model_builder(df)
    indicators = DummyIndicators(len(df) - 2)
    features = asyncio.run(mb.prepare_lstm_features("BTCUSDT", indicators))
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(df), 8)


def test_prepare_lstm_features_with_long_indicators():
    df = make_df(200)
    mb = create_model_builder(df)
    indicators = DummyIndicators(1000)
    features = asyncio.run(mb.prepare_lstm_features("BTCUSDT", indicators))
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(df), 15)


def test_prepare_lstm_features_short_history_returns_empty():
    df = make_df(3)
    mb = create_model_builder(df)
    mb.config.min_data_length = 10
    indicators = DummyIndicators(len(df))
    features = asyncio.run(mb.prepare_lstm_features("BTCUSDT", indicators))
    assert isinstance(features, np.ndarray)
    assert features.size == 0

@pytest.mark.parametrize("model_type", ["mlp", "tft"])
def test_train_model_remote_returns_state_and_predictions(model_type):
    X = np.random.rand(20, 3, 2).astype(np.float32)
    y = (np.random.rand(20) > 0.5).astype(np.float32)
    func = getattr(_train_model_remote, "_function", _train_model_remote)
    state, preds, labels = func(X, y, batch_size=2, model_type=model_type)
    assert isinstance(state, dict)
    assert len(preds) == len(labels)
    assert isinstance(preds, list)
    assert isinstance(labels, list)


def test_train_model_remote_tft_predictions():
    X = np.random.rand(12, 3, 2).astype(np.float32)
    y = (np.random.rand(12) > 0.5).astype(np.float32)
    func = getattr(_train_model_remote, "_function", _train_model_remote)
    state, preds, labels = func(X, y, batch_size=2, model_type="tft")
    assert isinstance(state, dict)
    assert len(preds) == len(labels) > 0


@pytest.mark.asyncio
async def test_training_loop_recovery(monkeypatch, tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path), retrain_interval=0)
    dh = types.SimpleNamespace(usdt_pairs=["BTCUSDT"])
    mb = ModelBuilder(cfg, dh, DummyTradeManager())

    call = {"n": 0}

    async def fake_retrain(symbol):
        call["n"] += 1
        if call["n"] == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(mb, "retrain_symbol", fake_retrain)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(model_builder.asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(mb.train())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call["n"] >= 2


@pytest.mark.asyncio
async def test_threshold_offset_decay(tmp_path):
    cfg = BotConfig(
        cache_dir=str(tmp_path),
        threshold_adjustment=0.05,
        threshold_decay_rate=0.1,
        loss_streak_threshold=3,
        win_streak_threshold=3,
        base_probability_threshold=0.6,
        prediction_history_size=5,
    )
    dh = types.SimpleNamespace(ohlcv=pd.DataFrame({"close": []}), usdt_pairs=["BTCUSDT"])

    class TM:
        def __init__(self):
            self.loss = 3
            self.win = 0
            self.last_volatility = {}

        async def get_loss_streak(self, _):
            return self.loss

        async def get_win_streak(self, _):
            return self.win

        async def get_sharpe_ratio(self, _):
            return 0.0

    tm = TM()
    mb = ModelBuilder(cfg, dh, tm)
    long1, short1 = await mb.adjust_thresholds("BTCUSDT", 0.6)
    first_offset = mb.threshold_offset["BTCUSDT"]
    assert first_offset == pytest.approx(0.05 * 0.9)
    assert long1 == pytest.approx(0.6 + first_offset)
    assert short1 == pytest.approx(1 - long1)

    tm.loss = 0
    long2, _ = await mb.adjust_thresholds("BTCUSDT", 0.6)
    second_offset = mb.threshold_offset["BTCUSDT"]
    assert second_offset == pytest.approx(first_offset * 0.9)
    assert long2 == pytest.approx(0.6 + second_offset)


@pytest.mark.asyncio
async def test_loss_and_win_streak_adjustment(tmp_path):
    cfg = BotConfig(
        cache_dir=str(tmp_path),
        threshold_adjustment=0.05,
        threshold_decay_rate=0.1,
        loss_streak_threshold=2,
        win_streak_threshold=2,
        base_probability_threshold=0.6,
        prediction_history_size=5,
    )
    dh = types.SimpleNamespace(ohlcv=pd.DataFrame({"close": []}), usdt_pairs=["BTCUSDT"])

    class TM:
        def __init__(self):
            self.loss = 2
            self.win = 0
            self.last_volatility = {}

        async def get_loss_streak(self, _):
            return self.loss

        async def get_win_streak(self, _):
            return self.win

        async def get_sharpe_ratio(self, _):
            return 0.0

    tm = TM()
    mb = ModelBuilder(cfg, dh, tm)

    start_offset = mb.threshold_offset.get("BTCUSDT", 0.0)
    assert start_offset == 0.0

    await mb.adjust_thresholds("BTCUSDT", 0.6)
    after_loss = mb.threshold_offset["BTCUSDT"]
    inc = after_loss / (1 - cfg.threshold_decay_rate) - start_offset
    assert inc == pytest.approx(cfg.threshold_adjustment)

    tm.loss = 0
    tm.win = 2

    await mb.adjust_thresholds("BTCUSDT", 0.6)
    after_win = mb.threshold_offset["BTCUSDT"]
    dec = after_win / (1 - cfg.threshold_decay_rate) - after_loss
    assert dec == pytest.approx(-cfg.threshold_adjustment)


def test_save_and_load_state_transformer(tmp_path):
    df = make_df()
    cfg = BotConfig(
        cache_dir=str(tmp_path),
        min_data_length=len(df),
        lstm_timesteps=2,
        lstm_batch_size=2,
        model_type="tft",
        nn_framework="pytorch",
    )
    dh = DummyDataHandler(df)
    tm = DummyTradeManager()
    mb = ModelBuilder(cfg, dh, tm)
    torch_mods = model_builder._get_torch_modules()
    TFT = torch_mods["TemporalFusionTransformer"]
    torch = torch_mods["torch"]
    model = TFT(15)
    mb.predictive_models["BTCUSDT"] = model
    scaler = StandardScaler().fit(np.random.rand(3, 15))
    mb.scalers["BTCUSDT"] = scaler
    mb.last_save_time = 0
    mb.save_state()

    mb2 = ModelBuilder(cfg, dh, tm)
    mb2.load_state()
    state1 = mb.predictive_models["BTCUSDT"].state_dict()
    state2 = mb2.predictive_models["BTCUSDT"].state_dict()
    for k in state1:
        assert torch.allclose(state1[k], state2[k])


def test_compute_prediction_metrics(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path), performance_window=3)
    dh = types.SimpleNamespace(usdt_pairs=["BTCUSDT"])
    mb = ModelBuilder(cfg, dh, DummyTradeManager())
    mb.prediction_history["BTCUSDT"] = deque([(0.9, 1), (0.1, 0), (0.2, 1)], maxlen=3)
    metrics = mb.compute_prediction_metrics("BTCUSDT")
    assert metrics is not None
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    expected_brier = brier_score_loss([1, 0, 1], [0.9, 0.1, 0.2])
    assert metrics["brier_score"] == pytest.approx(expected_brier)


@pytest.mark.asyncio
async def test_performance_based_retraining(monkeypatch, tmp_path):
    cfg = BotConfig(
        cache_dir=str(tmp_path),
        retrain_interval=1000,
        retrain_threshold=0.8,
        performance_window=3,
    )
    dh = types.SimpleNamespace(usdt_pairs=["BTCUSDT"])
    mb = ModelBuilder(cfg, dh, DummyTradeManager())
    mb.prediction_history["BTCUSDT"] = deque([(0.9, 1), (0.1, 0), (0.2, 1)], maxlen=3)

    call = {"n": 0}

    async def fake_retrain(symbol):
        call["n"] += 1

    monkeypatch.setattr(mb, "retrain_symbol", fake_retrain)

    orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await orig_sleep(0)

    monkeypatch.setattr(model_builder.asyncio, "sleep", fast_sleep)

    task = asyncio.create_task(mb.train())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(task, 0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call["n"] >= 1


@pytest.mark.asyncio
async def test_base_threshold_convergence(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path), prediction_history_size=10)
    dh = types.SimpleNamespace(ohlcv=pd.DataFrame({"close": []}), usdt_pairs=["BTCUSDT"])

    class TM:
        last_volatility = {}

        async def get_loss_streak(self, _):
            return 0

        async def get_win_streak(self, _):
            return 0

        async def get_sharpe_ratio(self, _):
            return 0.0

    tm = TM()
    mb = ModelBuilder(cfg, dh, tm)
    mb.prediction_history["BTCUSDT"] = deque([], maxlen=10)
    data = [(0.7, 1), (0.3, 0), (0.8, 1), (0.2, 0), (0.6, 0)]
    for p, l in data:
        mb.prediction_history["BTCUSDT"].append((p, l))
        mb.compute_prediction_metrics("BTCUSDT")
    assert mb.base_thresholds["BTCUSDT"] == pytest.approx(0.8)
    long_thr, short_thr = await mb.adjust_thresholds("BTCUSDT", 0.5)
    assert long_thr == pytest.approx(0.8)
    assert short_thr == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_base_threshold_clamped_min(tmp_path):
    cfg = BotConfig(cache_dir=str(tmp_path), prediction_history_size=10)
    dh = types.SimpleNamespace(ohlcv=pd.DataFrame({"close": []}), usdt_pairs=["BTCUSDT"])

    class TM:
        last_volatility = {}

        async def get_loss_streak(self, _):
            return 0

        async def get_win_streak(self, _):
            return 0

        async def get_sharpe_ratio(self, _):
            return 0.0

    tm = TM()
    mb = ModelBuilder(cfg, dh, tm)
    mb.prediction_history["BTCUSDT"] = deque([], maxlen=10)
    data = [(0.1, 1), (0.2, 1), (0.3, 1), (0.8, 0), (0.6, 0)]
    for p, l in data:
        mb.prediction_history["BTCUSDT"].append((p, l))
        mb.compute_prediction_metrics("BTCUSDT")
    assert mb.base_thresholds["BTCUSDT"] == pytest.approx(0.5)
    long_thr, short_thr = await mb.adjust_thresholds("BTCUSDT", 0.5)
    assert long_thr == pytest.approx(0.5)
    assert short_thr == pytest.approx(0.5)


def test_freeze_torch_base_layers():
    torch_mods = model_builder._get_torch_modules()
    CNNGRU = torch_mods["CNNGRU"]
    model = CNNGRU(2, 64, 2, 0.2)
    model_builder._freeze_torch_base_layers(model, "gru")
    assert not any(p.requires_grad for p in model.conv.parameters())
    assert any(p.requires_grad for p in model.gru.parameters())


def test_train_model_remote_freezes_layers():
    X = np.random.rand(10, 3, 2).astype(np.float32)
    y = (np.random.rand(10) > 0.5).astype(np.float32)
    torch_mods = model_builder._get_torch_modules()
    CNNGRU = torch_mods["CNNGRU"]
    torch = torch_mods["torch"]
    model = CNNGRU(2, 64, 2, 0.2)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    func = getattr(_train_model_remote, "_function", _train_model_remote)
    state, _, _ = func(
        X,
        y,
        batch_size=2,
        model_type="gru",
        framework="pytorch",
        initial_state=init_state,
        epochs=1,
        n_splits=2,
        early_stopping_patience=1,
        freeze_base_layers=True,
    )
    new_model = CNNGRU(2, 64, 2, 0.2)
    new_model.load_state_dict(state)
    assert torch.allclose(new_model.conv.weight, init_state["conv.weight"])

