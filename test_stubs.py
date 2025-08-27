"""Lightweight stubs for optional heavy dependencies in tests.

When the environment variable ``TEST_MODE`` is set to ``"1"`` this module
injects simplified stand-ins for external libraries such as :mod:`ray` and
``httpx``.  Importing this module has no effect outside of test mode.
"""

from __future__ import annotations

import os
import sys
import types


IS_TEST_MODE = False


def apply() -> None:
    """Patch heavy dependencies with lightweight stubs in test mode."""
    global IS_TEST_MODE
    IS_TEST_MODE = os.getenv("TEST_MODE") == "1"
    if not IS_TEST_MODE:
        return

    # ------------------------------------------------------------------ Ray
    ray_mod = types.ModuleType("ray")

    class _RayRemoteFunction:
        def __init__(self, func):
            self._function = func

        def remote(self, *args, **kwargs):
            return self._function(*args, **kwargs)

        def options(self, *args, **kwargs):  # pragma: no cover - simple chain
            return self

    def _ray_remote(func=None, **_kwargs):
        if func is None:
            def wrapper(f):
                return _RayRemoteFunction(f)

            return wrapper
        return _RayRemoteFunction(func)

    _ray_state = {"initialized": False}

    def _init(*_a, **_k):
        _ray_state["initialized"] = True

    def _is_initialized() -> bool:
        return _ray_state["initialized"]

    def _shutdown(*_a, **_k):
        _ray_state["initialized"] = False

    ray_mod.remote = _ray_remote
    ray_mod.get = lambda x: x
    ray_mod.init = _init
    ray_mod.is_initialized = _is_initialized
    ray_mod.shutdown = _shutdown
    sys.modules["ray"] = ray_mod

    # ----------------------------------------------------------------- HTTPX
    httpx_mod = types.ModuleType("httpx")
    httpx_mod.HTTPError = Exception
    sys.modules["httpx"] = httpx_mod

    # ------------------------------------------------------------------- PyBit
    pybit_mod = types.ModuleType("pybit")
    ut_mod = types.ModuleType("unified_trading")
    ut_mod.HTTP = object
    pybit_mod.unified_trading = ut_mod
    sys.modules["pybit"] = pybit_mod
    sys.modules["pybit.unified_trading"] = ut_mod

    # ------------------------------------------------------------------ a2wsgi
    a2wsgi_mod = types.ModuleType("a2wsgi")
    a2wsgi_mod.WSGIMiddleware = lambda app: app
    sys.modules["a2wsgi"] = a2wsgi_mod

    # ------------------------------------------------------------------ uvicorn
    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.middleware = types.SimpleNamespace(
        wsgi=types.SimpleNamespace(WSGIMiddleware=lambda app: app)
    )
    sys.modules["uvicorn"] = uvicorn_mod

    # ------------------------------------------------------------------- Flask
    try:  # pragma: no cover - best effort
        from flask import Flask

        if not hasattr(Flask, "asgi_app"):
            Flask.asgi_app = property(lambda self: self.wsgi_app)
    except Exception:
        pass


apply()


__all__ = ["IS_TEST_MODE", "apply"]

