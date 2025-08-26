import pytest
from bot import utils


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_gpu_functions_disabled(monkeypatch, device):
    if device == "gpu":
        monkeypatch.delenv("FORCE_CPU", raising=False)
        if not utils.is_cuda_available():
            pytest.skip("GPU not available")
        assert utils.is_cuda_available() is True
    else:
        monkeypatch.setenv("FORCE_CPU", "1")
        assert utils.is_cuda_available() is False
