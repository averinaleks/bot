from __future__ import annotations

import importlib
import importlib.util
import os
import runpy
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, cast

_PipMain = Callable[[list[str] | None], int]


def _load_pip_main() -> _PipMain | None:
    try:
        module = importlib.import_module("pip._internal.cli.main")
    except Exception:  # pragma: no cover - pip should always be importable, but guard just in case
        return None

    main_callable = getattr(module, "main", None)
    if not callable(main_callable):  # pragma: no cover - defensive
        return None
    return cast(_PipMain, main_callable)


_pip_main = _load_pip_main()


def _running_under_codeql() -> bool:
    """Return ``True`` when the current interpreter is launched by CodeQL."""

    if os.environ.get("CODEQL_DIST"):
        return True

    return any(name.startswith("CODEQL_EXTRACTOR_PYTHON") for name in os.environ)


def _ensure_packages(packages: Iterable[tuple[str, str]]) -> None:
    if os.environ.get("BOT_AUTO_INSTALL_DISABLED") == "1" or _running_under_codeql():
        return

    for module_name, requirement in packages:
        try:
            spec_found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive guard
            spec_found = False
        if spec_found:
            continue

        _run_pip_install(requirement)

        # Installing a package mutates ``sys.path`` by writing .pth files and
        # populates the import caches used by ``importlib``.  Without
        # invalidating those caches the subsequent ``find_spec`` lookup would
        # still return ``None`` even though the distribution is now
        # available.  The CI environment exercises this scenario and, prior to
        # the invalidation, caused the helper to raise a ``RuntimeError``
        # despite the installation succeeding.  Refreshing the caches ensures
        # the interpreter sees the newly installed modules immediately.
        importlib.invalidate_caches()

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
    ("numpy", "numpy==2.1.3"),
    ("pandas", "pandas==2.2.3"),
    ("pydantic", "pydantic==2.12.0"),
    ("flask", "flask>=3.0.3,<4"),
    ("requests", "requests>=2.32.3"),
    ("httpx", "httpx>=0.27.0"),
    ("aiohttp", "aiohttp>=3.13.0"),
    ("openai", "openai==2.2.0"),
    ("bcrypt", "bcrypt>=4.1.3"),
    ("psutil", "psutil>=5.9.0"),
    ("polars", "polars>=1.6.0"),
    ("pyarrow", "pyarrow>=15.0.0"),
    ("joblib", "joblib>=1.3"),
    ("cloudpickle", "cloudpickle>=3.1"),
]

_bcrypt_stub_path = Path(__file__).with_name("bcrypt.py")
_test_mode_enabled = os.environ.get("TEST_MODE") == "1"
_ci_environment = os.environ.get("CI", "").lower() == "true"

if (_test_mode_enabled or _ci_environment) and _bcrypt_stub_path.is_file():
    _required_packages = [
        requirement
        for requirement in _required_packages
        if requirement[0] != "bcrypt"
    ]

if sys.version_info < (3, 12):
    _required_packages.append(("sklearn", "scikit-learn==1.7.2"))

_ensure_packages(_required_packages)


def _wrap_tempfile_mkdtemp() -> None:
    """Ensure ``tempfile.mkdtemp`` always creates the parent directory."""

    original = getattr(tempfile, "_bot_original_mkdtemp", None)
    if original is None:
        original = tempfile.mkdtemp
        tempfile._bot_original_mkdtemp = original  # type: ignore[attr-defined]

        def _safe_mkdtemp(*args, **kwargs):
            target_dir = kwargs.get("dir")
            if target_dir is None:
                target_dir = tempfile.gettempdir()
            if target_dir:
                try:
                    Path(target_dir).mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise RuntimeError(
                        f"Не удалось подготовить каталог временных файлов: {target_dir!s}"
                    ) from exc
            try:
                return original(*args, **kwargs)
            except FileNotFoundError:
                if target_dir:
                    Path(target_dir).mkdir(parents=True, exist_ok=True)
                return original(*args, **kwargs)

        tempfile.mkdtemp = _safe_mkdtemp


_wrap_tempfile_mkdtemp()
