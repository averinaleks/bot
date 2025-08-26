class AutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return object()


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        class _Dummy:
            def to(self, *_args, **_kwargs):
                return self
        return _Dummy()
