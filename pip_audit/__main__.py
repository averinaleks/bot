"""Lightweight entrypoint for the bundled ``pip_audit`` stub.

The real ``pip-audit`` CLI lives in the upstream project and is intentionally
not vendored here to keep optional security tooling light-weight. Without this
module ``python -m pip_audit`` fails with ``ModuleNotFoundError``, which breaks
local security workflows and CI jobs that expect a module entrypoint.

When the upstream package is available in the environment we delegate to the
``pip-audit`` executable. Otherwise we exit with a clear, actionable error
explaining how to install the full tool.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Iterable


def _build_command(argv: Iterable[str] | None) -> list[str]:
    executable = shutil.which("pip-audit")
    if executable is None:
        raise ModuleNotFoundError(
            "pip-audit CLI components are not installed; "
            "run `pip install pip-audit` to enable security scans"
        )

    command = [executable]
    if argv:
        command.extend(argv)
    return command


def main(argv: list[str] | None = None) -> int:
    """Invoke the external ``pip-audit`` CLI when available."""

    command = _build_command(argv)
    result = subprocess.run(command, check=False)
    return int(result.returncode)


if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``
    sys.exit(main(sys.argv[1:]))
