import sys, types, asyncio, pytest, importlib, os

os.environ["API_KEYS"] = "testkey"


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


class _ConditionalTokenizer:
    calls: list[bool] = []

    @staticmethod
    def from_pretrained(*args, **kwargs):
        flag = kwargs.get("local_files_only", False)
        _ConditionalTokenizer.calls.append(flag)
        if flag:
            return object()
        raise OSError("network down")


class _DummyModel:
    def to(self, *_args, **_kwargs):
        return self


class _ConditionalModel:
    calls: list[bool] = []

    @staticmethod
    def from_pretrained(*args, **kwargs):
        flag = kwargs.get("local_files_only", False)
        _ConditionalModel.calls.append(flag)
        if flag:
            return _DummyModel()
        raise OSError("network down")


def test_load_model_uses_local_cache(monkeypatch):
    with monkeypatch.context() as m:
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = _ConditionalTokenizer
        transformers.AutoModelForCausalLM = _ConditionalModel
        m.setitem(sys.modules, "transformers", transformers)

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        m.setitem(sys.modules, "torch", torch)

        os.environ["CSRF_SECRET"] = "testsecret"
        import server

        result = server.model_manager.load_model()
        assert result == "primary"
        assert _ConditionalTokenizer.calls == [False, True]
        assert _ConditionalModel.calls == [True]

    try:
        importlib.reload(server)
    except ModuleNotFoundError:
        pass
    finally:
        os.environ.pop("CSRF_SECRET", None)
        _ConditionalTokenizer.calls.clear()
        _ConditionalModel.calls.clear()
