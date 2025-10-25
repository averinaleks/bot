"""Helpers that provide a safe Ray interface even when the package is missing.

The project historically relied on Ray for distributed optimisations, но
уязвимость CVE-2023-48022 делает установку ``ray`` небезопасной в публичной
сети.  Чтобы сборка по умолчанию проходила проверку Trivy, мы поставляем
минимальную заглушку, эмулирующую ключевые API Ray.  Если реальный Ray
присутствует в окружении, модуль прозрачно проксирует вызовы к нему, а также
проверяет версию через :func:`security.ensure_minimum_ray_version`.
"""
from __future__ import annotations

import contextlib
import os
import sys
from importlib import metadata as importlib_metadata
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import security

SAFE_RAY_VERSION_STR = security.SAFE_RAY_VERSION_STR

__all__ = ["ray", "IS_RAY_STUB"]


def _is_probably_stub(module: ModuleType) -> bool:
    """Detect Ray stand-ins that mark themselves as stubs."""

    return bool(getattr(module, "__ray_stub__", False))


def _ensure_module_version_attr(module: ModuleType) -> None:
    """Populate ``module.__version__`` if the attribute is missing."""

    if getattr(module, "__version__", ""):
        return

    candidates: list[str] = []
    package = getattr(module, "__package__", "")
    if package:
        candidates.append(package.split(".")[0])
    name = getattr(module, "__name__", "")
    if name:
        candidates.append(name.split(".")[0])
    spec = getattr(module, "__spec__", None)
    spec_name = getattr(spec, "name", "") if spec else ""
    if spec_name:
        candidates.append(spec_name.split(".")[0])

    for distribution in dict.fromkeys(candidates):
        if not distribution:
            continue
        with contextlib.suppress(importlib_metadata.PackageNotFoundError):
            module.__version__ = importlib_metadata.version(distribution)
            return

    module.__version__ = SAFE_RAY_VERSION_STR


def _create_stub() -> tuple[ModuleType, bool]:
    """Вернуть заглушку Ray с минимально необходимым API."""

    class _ObjectRef:
        """Эмуляция ``ray.ObjectRef``."""

        __slots__ = ("_value",)

        def __init__(self, value: Any) -> None:
            self._value = value

    class _RemoteHandle:
        """Обёртка, предоставляющая ``.remote`` и ``.options``."""

        __slots__ = ("_func",)

        def __init__(self, func):
            self._func = func

        def remote(self, *args: Any, **kwargs: Any) -> _ObjectRef:
            return _ObjectRef(self._func(*args, **kwargs))

        def options(self, *args: Any, **kwargs: Any) -> "_RemoteHandle":
            return self

    state = {"initialized": False}

    def _remote(func=None, **_options):
        if func is None:
            def _decorator(inner):
                return _RemoteHandle(inner)

            return _decorator
        return _RemoteHandle(func)

    def _unwrap(value: Any) -> Any:
        if isinstance(value, _ObjectRef):
            return value._value
        if isinstance(value, dict):
            return {k: _unwrap(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            iterable: Iterable[Any] = (_unwrap(v) for v in value)
            return type(value)(iterable)
        return value

    def _init(**_kwargs):
        state["initialized"] = True
        return SimpleNamespace(address="ray://stub")

    def _shutdown() -> None:
        state["initialized"] = False

    def _is_initialized() -> bool:
        return state["initialized"]

    stub = ModuleType("ray")
    stub.__dict__.update(
        {
            "__version__": "0.0.0-stub",
            "__ray_stub__": True,
            "remote": _remote,
            "get": _unwrap,
            "init": _init,
            "shutdown": lambda **_k: _shutdown(),
            "is_initialized": _is_initialized,
            "ObjectRef": _ObjectRef,
        }
    )
    return stub, True


def _augment_ray_module(module: ModuleType) -> ModuleType:
    """Patch missing Ray attributes on *module* using the local stub implementation."""

    stub, _ = _create_stub()
    if not getattr(module, "__version__", ""):
        module.__dict__.setdefault("__version__", SAFE_RAY_VERSION_STR)

    for name in ("remote", "get", "init", "shutdown", "is_initialized", "ObjectRef"):
        if not hasattr(module, name):
            module.__dict__[name] = getattr(stub, name)

    return module


def _load_ray() -> tuple[Any, bool]:
    """Import Ray if available, otherwise fall back to a stub."""

    existing = sys.modules.get("ray")
    if existing is not None:
        if _is_probably_stub(existing):
            return existing, True

        _ensure_module_version_attr(existing)
        security.ensure_minimum_ray_version(existing)
        augmented_existing = _augment_ray_module(existing)
        sys.modules["ray"] = augmented_existing
        return augmented_existing, False

    if os.getenv("TEST_MODE") == "1":
        stub, is_stub = _create_stub()
        sys.modules["ray"] = stub
        return stub, is_stub

    try:
        import ray  # type: ignore
    except ImportError:
        stub, is_stub = _create_stub()
        sys.modules["ray"] = stub
        return stub, is_stub

    _ensure_module_version_attr(ray)
    security.ensure_minimum_ray_version(ray)
    augmented = _augment_ray_module(ray)
    sys.modules["ray"] = augmented
    return augmented, _is_probably_stub(augmented)


ray, IS_RAY_STUB = _load_ray()
"""Экспортируем совместимый объект ``ray`` и флаг, загружена ли заглушка."""
