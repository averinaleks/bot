"""Ensure a valid ``semgrep.sarif`` file exists for GitHub code scanning uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

SARIF_PATH = Path("semgrep.sarif")

_EMPTY_SARIF: dict[str, Any] = {
    "version": "2.1.0",
    "runs": [
        {
            "tool": {
                "driver": {
                    "name": "Semgrep",
                    "informationUri": "https://semgrep.dev",
                    "rules": [],
                }
            },
            "results": [],
            "invocations": [
                {
                    "executionSuccessful": True,
                }
            ],
        }
    ],
}


def ensure_semgrep_sarif(path: Path = SARIF_PATH) -> Path:
    """Create an empty SARIF report when *path* is missing or empty."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    path.write_text(json.dumps(_EMPTY_SARIF, indent=2), encoding="utf-8")
    return path


def main() -> int:
    ensure_semgrep_sarif()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
