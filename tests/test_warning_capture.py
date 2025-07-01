import warnings
import pytest


def api_v1():
    warnings.warn(UserWarning("api v1, should use functions from v2"))
    return 1


def api_v2():
    """Replacement for :func:`api_v1` that does not emit warnings."""
    return 1


def test_api_v2_no_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        assert api_v2() == 1
        assert not captured
