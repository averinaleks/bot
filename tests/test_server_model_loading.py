import sys, types, asyncio, pytest, importlib, os


class _FailingLoader:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise OSError("fail")


def test_load_model_async_raises_runtime_error(monkeypatch):
    with monkeypatch.context() as m:
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = _FailingLoader
        transformers.AutoModelForCausalLM = _FailingLoader
        m.setitem(sys.modules, "transformers", transformers)

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        m.setitem(sys.modules, "torch", torch)

        os.environ["CSRF_SECRET"] = "testsecret"
        import server

        with pytest.raises(RuntimeError, match="Failed to load both primary and fallback models"):
            asyncio.run(server.model_manager.load_model_async())

    try:
        importlib.reload(server)
    except ModuleNotFoundError:
        pass
    finally:
        os.environ.pop("CSRF_SECRET", None)
