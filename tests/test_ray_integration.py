"""Integration checks that require the real Ray package."""

from __future__ import annotations

import importlib
import sys

import pytest

import security
from security import SAFE_RAY_VERSION

pytestmark = pytest.mark.integration


@pytest.fixture
def _reload_ray_compat():
    """Ensure ``bot.ray_compat`` is re-imported with the current environment."""

    def _reload():
        for name in ["bot.ray_compat"]:
            sys.modules.pop(name, None)
        return importlib.import_module("bot.ray_compat")

    return _reload


def test_real_ray_module_is_used(monkeypatch, _reload_ray_compat):
    ray_pkg = pytest.importorskip("ray")
    monkeypatch.setenv("TEST_MODE", "0")

    original_checker = security.ensure_minimum_ray_version
    calls: dict[str, str] = {}

    def _tracking_checker(module):
        calls["version"] = getattr(module, "__version__", "")
        return original_checker(module)

    monkeypatch.setattr("security.ensure_minimum_ray_version", _tracking_checker)

    compat = _reload_ray_compat()

    assert not compat.IS_RAY_STUB, "The Ray stub should not be active when the real package is installed."
    assert compat.ray is ray_pkg
    assert calls["version"] == ray_pkg.__version__

    from packaging.version import Version

    assert Version(ray_pkg.__version__) >= SAFE_RAY_VERSION

    @compat.ray.remote
    def _double(value: int) -> int:
        return value * 2

    result = compat.ray.get(_double.remote(21))
    assert result == 42
