"""Tests for :mod:`safe_html_parser`.

These tests validate that the SafeHTMLParser enforces the maximum feed size
and behaves like HTMLParser for small inputs.
"""

import pytest

from safe_html_parser import DEFAULT_MAX_FEED, SafeHTMLParser


def test_small_input_parses():
    parser = SafeHTMLParser()
    parser.feed("<div>ok</div>")
    # No exception should be raised and parser should have consumed input.
    assert parser.fed_bytes == len("<div>ok</div>")


def test_large_input_raises():
    parser = SafeHTMLParser(max_feed_size=10)
    parser.feed("<div>")
    try:
        parser.feed("x" * 20)
    except ValueError as exc:  # pragma: no cover - explicit check
        assert "maximum" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for oversized input")


def test_massive_default_input_raises():
    parser = SafeHTMLParser()
    massive = "x" * (DEFAULT_MAX_FEED * 2)
    with pytest.raises(ValueError):
        parser.feed(massive)


@pytest.mark.parametrize("invalid_size", [0, -1])
def test_non_positive_max_feed_size_raises(invalid_size):
    with pytest.raises(ValueError):
        SafeHTMLParser(max_feed_size=invalid_size)


def test_counts_bytes_not_chars():
    parser = SafeHTMLParser(max_feed_size=4)
    # emoji is four bytes in UTF-8
    parser.feed("😀")
    assert parser.fed_bytes == len("😀".encode("utf-8"))
    with pytest.raises(ValueError):
        parser.feed("😀")


def test_multilingual_input_counts_bytes():
    parser = SafeHTMLParser(max_feed_size=10)
    multilingual = "ä漢字"  # mix of 2-byte and 3-byte characters
    parser.feed(multilingual)
    assert parser.fed_bytes == len(multilingual.encode("utf-8"))
    with pytest.raises(ValueError):
        parser.feed("😀")


def test_close_resets_counter():
    parser = SafeHTMLParser(max_feed_size=5)
    parser.feed("12345")
    parser.close()
    # After closing, parser should be reusable without raising.
    parser.feed("abcde")
    assert parser.fed_bytes == 5
