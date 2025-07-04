import warnings


def api_v2():
    """Replacement for :func:`api_v1` that does not emit warnings."""
    return 1


def test_api_v2_no_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        assert api_v2() == 1
        assert not captured
