"""Tests for :mod:`safe_html_parser`.

These tests validate that the SafeHTMLParser enforces the maximum feed size
and behaves like HTMLParser for small inputs.
"""

import pytest

from safe_html_parser import SafeHTMLParser


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


def test_counts_bytes_not_chars():
    parser = SafeHTMLParser(max_feed_size=4)
    # emoji is four bytes in UTF-8
    parser.feed("😀")
    assert parser._fed == len("😀".encode("utf-8"))
    with pytest.raises(ValueError):
        parser.feed("😀")


def test_close_resets_counter():
    parser = SafeHTMLParser(max_feed_size=5)
    parser.feed("12345")
    parser.close()
    # After closing, parser should be reusable without raising.
    parser.feed("abcde")
    assert parser._fed == 5
