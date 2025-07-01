import warnings
import pytest


def api_v1():
    """Legacy API now delegates to :func:`api_v2` without emitting warnings."""
    return api_v2()


def api_v2():
    """Replacement for :func:`api_v1` that does not emit warnings."""
    return 1


def test_api_v2_no_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        assert api_v2() == 1
        assert not captured


def test_api_v1_no_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        assert api_v1() == 1
        assert not captured
