import os, sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import asyncio
import numpy as np
import pandas as pd
import types
import pytest

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
from model_builder import ModelBuilder, _train_lstm_remote

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
        self.funding_rates = {"BTCUSDT": 0.1}
        self.open_interest = {"BTCUSDT": 0.2}
        self.usdt_pairs = ["BTCUSDT"]

class DummyTradeManager:
    pass

def create_model_builder(df):
    config = {
        "cache_dir": "/tmp",
        "min_data_length": len(df),
        "lstm_timesteps": 2,
        "lstm_batch_size": 2,
    }
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

@pytest.mark.asyncio
def test_prepare_lstm_features_shape():
    df = make_df()
    mb = create_model_builder(df)
    indicators = DummyIndicators(len(df))
    features = asyncio.run(mb.prepare_lstm_features("BTCUSDT", indicators))
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(df), 14)

@pytest.mark.asyncio
def test_train_lstm_remote_returns_state_and_predictions():
    X = np.random.rand(20, 3, 2).astype(np.float32)
    y = (np.random.rand(20) > 0.5).astype(np.float32)
    state, preds, labels = _train_lstm_remote._function(X, y, batch_size=2)
    assert isinstance(state, dict)
    assert len(preds) == len(labels)
    assert isinstance(preds, list)
    assert isinstance(labels, list)
