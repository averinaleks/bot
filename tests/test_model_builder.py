import os
import sys

import numpy as np
import pandas as pd
import types
import pytest
import importlib.util
import contextlib
from config import BotConfig
import asyncio

try:  # require functional torch installation for these tests
    import torch
    import torch.nn  # noqa: F401
except Exception:
    pytest.skip('torch not available', allow_module_level=True)

# Provide dummy stable_baselines3 if missing
if "stable_baselines3" not in sys.modules:
    sb3 = types.ModuleType("stable_baselines3")
    sb3.PPO = object
    sb3.DQN = object
    common = types.ModuleType("stable_baselines3.common")
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")
    vec_env.DummyVecEnv = object
    common.vec_env = vec_env
    sb3.common = common
    sys.modules["stable_baselines3"] = sb3
    sys.modules["stable_baselines3.common"] = common
    sys.modules["stable_baselines3.common.vec_env"] = vec_env

import model_builder
from model_builder import ModelBuilder, _train_model_remote

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
        self.usdt_pairs = ["BTCUSDT"]

class DummyTradeManager:
    pass

def create_model_builder(df):
    config = BotConfig(
        cache_dir="/tmp",
        min_data_length=len(df),
        lstm_timesteps=2,
        lstm_batch_size=2,
        model_type="cnn_lstm",
    )
    data_handler = DummyDataHandler(df)
    trade_manager = DummyTradeManager()
    return ModelBuilder(config, data_handler, trade_manager)

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
    assert features.shape == (len(df), 14)


def test_prepare_lstm_features_with_short_indicators():
    df = make_df()
    mb = create_model_builder(df)
    indicators = DummyIndicators(len(df) - 2)
    features = asyncio.run(mb.prepare_lstm_features("BTCUSDT", indicators))
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(df), 14)

@pytest.mark.parametrize("model_type", ["cnn_lstm", "mlp"])
def test_train_model_remote_returns_state_and_predictions(model_type):
    X = np.random.rand(20, 3, 2).astype(np.float32)
    y = (np.random.rand(20) > 0.5).astype(np.float32)
    func = getattr(_train_model_remote, "_function", _train_model_remote)
    state, preds, labels = func(X, y, batch_size=2, model_type=model_type)
    assert isinstance(state, dict)
    assert len(preds) == len(labels)
    assert isinstance(preds, list)
    assert isinstance(labels, list)


@pytest.mark.asyncio
async def test_training_loop_recovery(monkeypatch):
    cfg = BotConfig(cache_dir="/tmp", retrain_interval=0)
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
