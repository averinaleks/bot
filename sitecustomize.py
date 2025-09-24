from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from types import ModuleType
from typing import Callable, Mapping, NamedTuple


class PackageConfig(NamedTuple):
    """Configuration describing how to install and import a dependency."""

    requirement: str
    importer: Callable[[], ModuleType]


def _ensure_packages(packages: Mapping[str, PackageConfig]) -> None:
    if os.environ.get("BOT_AUTO_INSTALL_DISABLED") == "1":
        return

    for module_name, config in packages.items():
        try:
            config.importer()
            continue
        except ModuleNotFoundError:
            pass

        _run_pip_install(config.requirement)

        try:
            config.importer()
        except ModuleNotFoundError as exc:  # pragma: no cover - unexpected
            raise RuntimeError(
                "Dependency '%s' was installed but importing '%s' still failed"
                % (config.requirement, module_name)
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


def _import_numpy() -> ModuleType:
    return import_module("numpy")


def _import_pandas() -> ModuleType:
    return import_module("pandas")


def _import_pydantic() -> ModuleType:
    return import_module("pydantic")


def _import_flask() -> ModuleType:
    return import_module("flask")


def _import_psutil() -> ModuleType:
    return import_module("psutil")


def _import_polars() -> ModuleType:
    return import_module("polars")


def _import_pyarrow() -> ModuleType:
    return import_module("pyarrow")


def _import_joblib() -> ModuleType:
    return import_module("joblib")


def _import_sklearn() -> ModuleType:
    return import_module("sklearn")


_REQUIRED_PACKAGES: dict[str, PackageConfig] = {
    "numpy": PackageConfig("numpy==2.2.6", _import_numpy),
    "pandas": PackageConfig("pandas==2.3.2", _import_pandas),
    "pydantic": PackageConfig("pydantic==2.11.9", _import_pydantic),
    "flask": PackageConfig("flask>=3.0.3,<4", _import_flask),
    "psutil": PackageConfig("psutil>=5.9.0", _import_psutil),
    "polars": PackageConfig("polars>=1.6.0", _import_polars),
    "pyarrow": PackageConfig("pyarrow>=15.0.0", _import_pyarrow),
    "joblib": PackageConfig("joblib>=1.3", _import_joblib),
}

if sys.version_info < (3, 12):
    _REQUIRED_PACKAGES["scikit-learn"] = PackageConfig(
        "scikit-learn==1.7.2", _import_sklearn
    )

_ensure_packages(_REQUIRED_PACKAGES)
