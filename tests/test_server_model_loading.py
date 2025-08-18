import sys, types, asyncio, pytest

# Stub modules to avoid heavy dependencies
class _FailingLoader:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise RuntimeError("fail")

transformers = types.ModuleType('transformers')
transformers.AutoTokenizer = _FailingLoader
transformers.AutoModelForCausalLM = _FailingLoader
sys.modules['transformers'] = transformers

torch = types.ModuleType('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules['torch'] = torch

import server


def test_load_model_async_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Failed to load both primary and fallback models"):
        asyncio.run(server.load_model_async())
