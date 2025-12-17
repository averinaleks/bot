"""Lightweight entrypoint for the bundled ``pip_audit`` stub.

The real ``pip-audit`` CLI lives in the upstream project and is intentionally
not vendored here to keep optional security tooling light-weight. Without this
module ``python -m pip_audit`` fails with ``ModuleNotFoundError``, which breaks
local security workflows and CI jobs that expect a module entrypoint.

When the upstream package is available in the environment we call into its
``pip_audit.__main__.main`` implementation directly. Otherwise we exit with a
clear, actionable error explaining how to install the full tool.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

from importlib import metadata


def _load_upstream_main() -> callable:
    """Resolve the upstream ``pip_audit`` entrypoint.

    The helper ensures we only import the external dependency when available
    and raises a clear error when the executable is missing from the
    environment. Importing and delegating directly avoids spawning
    subprocesses, which keeps invocation simple and removes command injection
    concerns flagged by security scanners.
    """

    try:
        metadata.version("pip-audit")
    except metadata.PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "pip-audit CLI components are not installed; "
            "run `pip install pip-audit` to enable security scans"
        ) from exc

    # Remove this stub's package directory from the search path so that imports
    # resolve to the real installation rather than recursing into the shim.
    package_root = Path(__file__).resolve().parents[1]
    search_path = [
        entry for entry in sys.path if Path(entry).resolve() != package_root
    ]

    original_sys_path = sys.path
    removed_modules: dict[str, object] = {}
    try:
        for name in list(sys.modules):
            if name == "pip_audit" or name.startswith("pip_audit."):
                removed_modules[name] = sys.modules.pop(name)

        sys.path = search_path
        upstream = importlib.import_module("pip_audit.__main__")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pip-audit is installed but its CLI entrypoint cannot be located"
        ) from exc
    finally:
        sys.modules.update(removed_modules)
        sys.path = original_sys_path

    main = getattr(upstream, "main", None)
    if main is None:
        raise RuntimeError("pip-audit is installed but exposes no CLI entrypoint")
    return main


def main(argv: list[str] | None = None) -> int:
    """Invoke the external ``pip-audit`` CLI when available."""

    upstream_main = _load_upstream_main()
    # Fall back to an empty argument list so the upstream implementation parses
    # its defaults consistently.
    return int(upstream_main(argv or []))


if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``
    sys.exit(main(sys.argv[1:]))
