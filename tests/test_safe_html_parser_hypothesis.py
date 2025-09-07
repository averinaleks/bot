import sys
import types
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, settings

from safe_html_parser import SafeHTMLParser

# Replace any SimpleNamespace entries in sys.modules with ModuleType to satisfy Hypothesis
for name, module in list(sys.modules.items()):
    if isinstance(module, types.SimpleNamespace):
        replacement = types.ModuleType(name)
        replacement.__dict__.update(module.__dict__)
        sys.modules[name] = replacement


@given(st.text(max_size=100))
@settings(deadline=None)
def test_small_inputs_parse_without_error(html):
    parser = SafeHTMLParser()
    parser.feed(html)
    assert parser.fed_bytes == len(html.encode("utf-8"))


@given(st.text(min_size=1).filter(lambda s: len(s.encode("utf-8")) > 10))
@settings(deadline=None)
def test_inputs_exceeding_limit_raise(html):
    parser = SafeHTMLParser(max_feed_size=10)
    with pytest.raises(ValueError):
        parser.feed(html)
