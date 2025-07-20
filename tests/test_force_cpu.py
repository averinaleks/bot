import os
import importlib
import utils


def test_force_cpu():
    os.environ["FORCE_CPU"] = "1"
    importlib.reload(utils)
    assert utils.is_cuda_available() is False
