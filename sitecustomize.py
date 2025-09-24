from __future__ import annotations

import importlib
import importlib.util
import os
import runpy
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

try:
    from pip._internal.cli.main import main as _pip_main
except Exception:  # pragma: no cover - pip should always be importable, but guard just in case
    _pip_main = None


def _ensure_packages(packages: Iterable[tuple[str, str]]) -> None:
    if os.environ.get("BOT_AUTO_INSTALL_DISABLED") == "1":
        return

    for module_name, requirement in packages:
        try:
            spec_found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive guard
            spec_found = False
        if spec_found:
            continue

        _run_pip_install(requirement)

        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError) as exc:  # pragma: no cover - unexpected
            raise RuntimeError(
                f"Dependency '{requirement}' was installed but importing '{module_name}' still failed"
            ) from exc

        if spec is None:  # pragma: no cover - unexpected
            raise RuntimeError(
                f"Dependency '{requirement}' was installed but importing '{module_name}' still failed"
            )


@contextmanager
def _temporarily_overridden_env(name: str, value: str) -> Iterator[None]:
    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _run_pip(args: Sequence[str]) -> int:
    if _pip_main is not None:
        return _pip_main(list(args))

    original_argv = sys.argv[:]
    try:
        sys.argv = ["pip", *args]
        runpy.run_module("pip", run_name="__main__", alter_sys=True)
    except SystemExit as exc:  # pragma: no cover - depends on pip internals
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    finally:
        sys.argv = original_argv

    return 0


def _run_pip_install(requirement: str) -> None:
    with _temporarily_overridden_env("BOT_AUTO_INSTALL_DISABLED", "1"):
        exit_code = _run_pip(["install", requirement])

    if exit_code != 0:
        raise RuntimeError(
            f"pip failed to install required dependency '{requirement}' (exit code {exit_code})"
        )


_required_packages: list[tuple[str, str]] = [
    ("numpy", "numpy==2.2.6"),
    ("pandas", "pandas==2.3.2"),
    ("pydantic", "pydantic==2.11.9"),
    ("flask", "flask>=3.0.3,<4"),
    ("psutil", "psutil>=5.9.0"),
    ("polars", "polars>=1.6.0"),
    ("pyarrow", "pyarrow>=15.0.0"),
    ("joblib", "joblib>=1.3"),
]

if sys.version_info < (3, 12):
    _required_packages.append(("scikit-learn", "scikit-learn==1.7.2"))

_ensure_packages(_required_packages)
