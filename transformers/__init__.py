"""Lightweight compatibility shim for optional ``transformers`` dependency."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import site
import sys
from types import ModuleType
from typing import Iterable


def _candidate_paths() -> Iterable[str]:
    """Yield unique site-package paths to search for the real library."""
    seen: set[str] = set()
    for attr in ("getsitepackages", "getusersitepackages"):
        getter = getattr(site, attr, None)
        if getter is None:
            continue
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive: site config errors
            continue
        if isinstance(value, str):
            iterable = [value]
        else:
            iterable = list(value)
        for path in iterable:
            if not path:
                continue
            normalized = path.rstrip("/")
            if normalized not in seen:
                seen.add(normalized)
                yield normalized


_SHIM_MODULE = sys.modules[__name__]


def _load_real_module() -> ModuleType | None:
    """Attempt to import the real HuggingFace transformers package."""
    for path in _candidate_paths():
        try:
            spec = importlib.machinery.PathFinder().find_spec(__name__, [path])
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            continue
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[__name__] = module
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - fallback to shim on import failure
            sys.modules[__name__] = _SHIM_MODULE
            continue
        return module
    return None


_real_module = _load_real_module()
if _real_module is not None:
    globals().update(_real_module.__dict__)
else:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # pragma: no cover - simple stub
            return object()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # pragma: no cover - simple stub
            class _Dummy:
                def to(self, *_args, **_kwargs):
                    return self

            return _Dummy()
