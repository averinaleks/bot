"""Tests for :mod:`safe_html_parser`.

These tests validate that the SafeHTMLParser enforces the maximum feed size
and behaves like HTMLParser for small inputs.
"""

from safe_html_parser import SafeHTMLParser, DEFAULT_MAX_FEED


def test_small_input_parses():
    parser = SafeHTMLParser()
    parser.feed("<div>ok</div>")
    # No exception should be raised and parser should have consumed input.
    assert parser._fed == len("<div>ok</div>")


def test_large_input_raises():
    parser = SafeHTMLParser(max_feed_size=10)
    parser.feed("<div>")
    try:
        parser.feed("x" * 20)
    except ValueError as exc:  # pragma: no cover - explicit check
        assert "maximum" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for oversized input")
