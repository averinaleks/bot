import sys, types, asyncio, pytest, importlib


class _FailingLoader:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise RuntimeError("fail")


def test_load_model_async_raises_runtime_error(monkeypatch):
    with monkeypatch.context() as m:
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = _FailingLoader
        transformers.AutoModelForCausalLM = _FailingLoader
        m.setitem(sys.modules, "transformers", transformers)

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        m.setitem(sys.modules, "torch", torch)

        import server

        with pytest.raises(RuntimeError, match="Failed to load both primary and fallback models"):
            asyncio.run(server.load_model_async())

    try:
        importlib.reload(server)
    except ModuleNotFoundError:
        pass
