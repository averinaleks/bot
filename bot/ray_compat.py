"""Helpers that provide a safe Ray interface even when the package is missing.

The project historically relied on Ray for distributed optimisations, но
уязвимость CVE-2023-48022 делает установку ``ray`` небезопасной в публичной
сети.  Чтобы сборка по умолчанию проходила проверку Trivy, мы поставляем
минимальную заглушку, эмулирующую ключевые API Ray.  Если реальный Ray
присутствует в окружении, модуль прозрачно проксирует вызовы к нему, а также
проверяет версию через :func:`security.ensure_minimum_ray_version`.
"""
from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

from security import ensure_minimum_ray_version

__all__ = ["ray", "IS_RAY_STUB"]


def _create_stub() -> tuple[ModuleType, bool]:
    """Вернуть заглушку Ray с минимально необходимым API."""

    class _ObjectRef:
        """Эмуляция ``ray.ObjectRef``."""

        def __init__(self, value: Any) -> None:
            self._value = value

    class _RemoteHandle:
        """Обёртка, предоставляющая ``.remote`` и ``.options``."""

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


def _load_ray() -> tuple[Any, bool]:
    """Import Ray if available, otherwise fall back to a stub."""

    existing = sys.modules.get("ray")
    if existing is not None:
        is_stub = getattr(existing, "__ray_stub__", False)
        if not is_stub:
            ensure_minimum_ray_version(existing)
        return existing, is_stub

    if os.getenv("TEST_MODE") == "1":
        stub, is_stub = _create_stub()
        return stub, is_stub

    try:
        import ray  # type: ignore
    except ImportError:
        stub, is_stub = _create_stub()
        return stub, is_stub
    else:  # pragma: no cover - настоящая установка Ray недоступна в CI
        ensure_minimum_ray_version(ray)
        return ray, False


ray, IS_RAY_STUB = _load_ray()
"""Экспортируем совместимый объект ``ray`` и флаг, загружена ли заглушка."""
