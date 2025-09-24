from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Iterable


def _ensure_packages(packages: Iterable[tuple[str, str]]) -> None:
    if os.environ.get("BOT_AUTO_INSTALL_DISABLED") == "1":
        return

    for module_name, requirement in packages:
        try:
            importlib.import_module(module_name)
            continue
        except ModuleNotFoundError:
            pass

        _run_pip_install(requirement)

        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - unexpected
            raise RuntimeError(
                f"Dependency '{requirement}' was installed but importing '{module_name}' still failed"
            ) from exc


def _run_pip_install(requirement: str) -> None:
    env = os.environ.copy()
    env["BOT_AUTO_INSTALL_DISABLED"] = "1"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", requirement],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
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
