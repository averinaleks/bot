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
import importlib.util
import sys
from pathlib import Path
from typing import Iterable


def _load_upstream_main() -> callable:
    """Resolve the upstream ``pip_audit`` entrypoint.

    The helper ensures we only import the external dependency when available
    and raises a clear error when the executable is missing from the
    environment. Importing and delegating directly avoids spawning
    subprocesses, which keeps invocation simple and removes command injection
    concerns flagged by security scanners.
    """

    spec = importlib.util.find_spec("pip_audit.__main__")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "pip-audit CLI components are not installed; "
            "run `pip install pip-audit` to enable security scans"
        )

    # When this stub is the only ``pip_audit`` module on the path the spec
    # resolves to this file, which would cause ``main`` to call back into
    # itself. Treat that as the same missing-dependency case to preserve the
    # clear ModuleNotFoundError raised previously.
    if Path(spec.origin).resolve() == Path(__file__).resolve():
        raise ModuleNotFoundError(
            "pip-audit CLI components are not installed; "
            "run `pip install pip-audit` to enable security scans"
        )

    upstream = importlib.import_module("pip_audit.__main__")

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
