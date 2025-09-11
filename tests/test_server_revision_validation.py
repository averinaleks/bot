import os
import sys
import types
import pytest


class _DummyTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return object()


class _DummyModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        class _Model:
            def to(self, *args, **kwargs):
                return self
        return _Model()


def test_invalid_model_revision(monkeypatch):
    with monkeypatch.context() as m:
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = _DummyTokenizer
        transformers.AutoModelForCausalLM = _DummyModel
        m.setitem(sys.modules, "transformers", transformers)

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        m.setitem(sys.modules, "torch", torch)

        os.environ["CSRF_SECRET"] = "testsecret"
        os.environ["GPT_MODEL_REVISION"] = "invalid"

        import server

        with pytest.raises(ValueError, match="GPT_MODEL_REVISION must be a 40-character SHA commit"):
            server.model_manager.load_model()

    os.environ.pop("CSRF_SECRET", None)
    os.environ.pop("GPT_MODEL_REVISION", None)
    try:
        del sys.modules["server"]
    except KeyError:
        pass
