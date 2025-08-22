import asyncio
import types
import numpy as np
import pytest
from bot.config import BotConfig
from bot import model_builder

class DummyTL:
    async def send_telegram_message(self, *args, **kwargs):
        pass

class DummyDH:
    def __init__(self):
        self.usdt_pairs = []
        self.telegram_logger = DummyTL()

class DummyTM:
    pass

class DummyExplainer:
    def __init__(self, model, data):
        pass
    def shap_values(self, data):
        return [np.zeros(data.shape)]


@pytest.mark.asyncio
async def test_shap_cache_file_safe_symbol(tmp_path, monkeypatch):
    cfg = BotConfig(
        cache_dir=str(tmp_path),
        model_type="tft",
        nn_framework="pytorch",
        min_data_length=1,
        lstm_timesteps=1,
    )
    class DummyLinear:
        def __init__(self, *a, **k):
            self.training = False
        def to(self, device):
            return self
        def eval(self):
            self.training = False
        def train(self):
            self.training = True
        def parameters(self):
            return iter([types.SimpleNamespace(device="cpu")])

    class DummyTorch:
        class nn:
            Linear = DummyLinear
        @staticmethod
        def tensor(x, dtype=None, device=None):
            class DummyTensor:
                def __init__(self, arr):
                    self.arr = np.array(arr)
                def to(self, device):
                    return self
                def view(self, *args):
                    self.arr = self.arr.reshape(*args)
                    return self
                def size(self, dim):
                    return self.arr.shape[dim]
                @property
                def shape(self):
                    return self.arr.shape
            return DummyTensor(x)
        float32 = np.float32
        @staticmethod
        def device(name):
            return name
        @staticmethod
        def no_grad():
            class Ctx:
                def __enter__(self):
                    pass
                def __exit__(self, exc, val, tb):
                    pass
            return Ctx()

    monkeypatch.setattr(
        model_builder,
        "_get_torch_modules",
        lambda: {"torch": DummyTorch()},
    )
    mb = model_builder.ModelBuilder(cfg, DummyDH(), DummyTM())
    torch = model_builder._get_torch_modules()["torch"]
    model = torch.nn.Linear(1, 1)
    X = np.random.rand(5, 1, 1).astype(np.float32)
    shap_stub = types.SimpleNamespace(GradientExplainer=DummyExplainer)
    monkeypatch.setattr(model_builder, "shap", shap_stub)
    await mb.compute_shap_values("BTC/USDT:PERP", model, X)
    assert (tmp_path / "shap" / "shap_BTC_USDT_PERP.pkl").exists()
