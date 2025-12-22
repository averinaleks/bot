import sys
import types
import importlib
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("utils_real", os.path.join(ROOT, "utils.py"))
utils_real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_real)
sys.modules['utils'] = utils_real

# Stub stable_baselines3 if missing
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

# Stub gymnasium if missing
if "gymnasium" not in sys.modules:
    gym_stub = types.ModuleType("gymnasium")

    class DummyDiscrete:
        def __init__(self, n):
            self.n = n

    class DummyBox:
        def __init__(self, low, high, shape, dtype):
            self.shape = shape

    class DummyEnv:
        action_space = DummyDiscrete(2)
        observation_space = DummyBox(-1, 1, (4,), float)

        def reset(self, seed=None, options=None):
            return np.zeros(self.observation_space.shape), {}

        def step(self, action):
            obs = np.zeros(self.observation_space.shape)
            reward = 1.0
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, {}

    gym_stub.Env = object
    gym_stub.spaces = types.SimpleNamespace(Discrete=DummyDiscrete, Box=DummyBox)
    gym_stub.make = lambda env_id: DummyEnv()
    sys.modules["gymnasium"] = gym_stub

from bot import model_builder  # noqa: E402
from bot.config import BotConfig  # noqa: E402
from bot.model_builder import RLAgent  # noqa: E402

importlib.reload(model_builder)


class DummyModelBuilder:
    def __init__(self):
        self.device = "cpu"
        self.predictive_models = {}

    async def preprocess(self, df, symbol):
        return df


class DummyIndicators:
    def __init__(self, length, index):
        base = pd.Series(np.zeros(length), index=index)
        self.ema30 = base
        self.ema100 = base
        self.ema200 = base
        self.rsi = base
        self.adx = base
        self.macd = base
        self.atr = base


class DummyDataHandler:
    def __init__(self, df, indicators):
        self.ohlcv = df
        self.funding_rates = {"BTCUSDT": 0.0}
        self.open_interest = {"BTCUSDT": 0.0}
        self.indicators = {"BTCUSDT": indicators}
        self.usdt_pairs = ["BTCUSDT"]


@pytest.mark.asyncio
async def test_train_symbol_and_predict(sample_ohlcv):
    idx = pd.MultiIndex.from_product([
        ["BTCUSDT"], sample_ohlcv.index
    ], names=["symbol", "timestamp"])
    df = sample_ohlcv.copy()
    df.index = idx
    indicators = DummyIndicators(len(df), df.index.droplevel("symbol"))
    dh = DummyDataHandler(df, indicators)
    agent = RLAgent(BotConfig(rl_timesteps=1), dh, DummyModelBuilder())
    await agent.train_symbol("BTCUSDT")
    assert "BTCUSDT" in agent.models
    features = await agent._prepare_features("BTCUSDT", indicators)
    action = agent.predict(
        "BTCUSDT", features.iloc[0].to_numpy(dtype=np.float32)
    )
    assert action in {"hold", "open_long", "open_short", "close"}
