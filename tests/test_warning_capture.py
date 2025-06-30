import warnings
import pytest


def api_v1():
    warnings.warn(UserWarning("api v1, should use functions from v2"))
    return 1


def test_api_v1_warns():
    with pytest.warns(UserWarning, match="api v1, should use functions from v2"):
        assert api_v1() == 1
