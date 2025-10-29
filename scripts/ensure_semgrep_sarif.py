"""Semgrep SARIF helpers used by CI workflows.

This module serves two purposes for the Semgrep GitHub Actions workflow:

* ensure that a ``semgrep.sarif`` file always exists so that the upload step
  does not fail when Semgrep produces no findings;
* compute the number of findings inside the report and expose it via
  ``GITHUB_OUTPUT`` so subsequent steps can decide whether the SARIF file
  should be uploaded or if the workflow should fail due to real findings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

try:
    from ._filesystem import write_secure_text
except ImportError:  # pragma: no cover - executed when run as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts._filesystem import write_secure_text  # type: ignore[import-not-found]

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

    write_secure_text(path, json.dumps(_EMPTY_SARIF, indent=2))
    return path


def sarif_result_count(path: Path = SARIF_PATH) -> int:
    """Return the total number of findings contained in *path*."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(f"Invalid SARIF report at {path}: {exc}") from exc

    runs = data.get("runs", [])
    return sum(len(run.get("results", [])) for run in runs)


def write_github_output(path: Path, *, upload: bool, findings: int, sarif_path: Path) -> None:
    """Append outputs consumed by subsequent workflow steps."""

    write_secure_text(
        path,
        (
            f"upload={'true' if upload else 'false'}\n"
            f"result_count={findings}\n"
            f"sarif_path={sarif_path}\n"
        ),
        append=True,
    )


def _normalize_github_output(path: Path | None) -> Path | None:
    """Return ``None`` when *path* is unset, empty, or refers to a directory."""

    if path is None:
        return None

    # ``Path("")`` normalizes to ``Path(".")`` which is not a real file target and
    # causes ``os.open`` to raise ``IsADirectoryError``.  Treat it as missing so the
    # script can run outside of GitHub Actions where ``GITHUB_OUTPUT`` is unset.
    if path == Path("."):
        return None

    if path.exists() and path.is_dir():
        return None

    text = str(path).strip()
    return path if text else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=SARIF_PATH,
        help="Path to the Semgrep SARIF file (default: semgrep.sarif)",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="File path supplied by GitHub Actions to persist step outputs.",
    )
    args = parser.parse_args(argv)

    sarif_path = ensure_semgrep_sarif(args.path)
    findings = sarif_result_count(sarif_path)
    has_findings = findings > 0

    github_output = _normalize_github_output(args.github_output)

    if github_output is None and args.github_output is not None:
        print(
            "GitHub output path was empty or a directory; skipping output export.",
            file=sys.stderr,
        )

    if github_output is not None:
        write_github_output(
            github_output,
            upload=has_findings,
            findings=findings,
            sarif_path=sarif_path,
        )

    if has_findings:
        print(f"Semgrep detected {findings} potential issue(s).")
    else:
        print(f"No Semgrep findings detected; ensured empty SARIF at {sarif_path}.")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
