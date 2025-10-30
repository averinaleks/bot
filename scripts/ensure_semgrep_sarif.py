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
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from ._filesystem import write_secure_text
except ImportError:  # pragma: no cover - executed when run as a script
    import importlib.util
    import sys

    _FS_RELATIVE_PATH = Path(__file__).resolve().parent / "_filesystem.py"

    module_name = "scripts._filesystem"
    spec = importlib.util.spec_from_file_location(module_name, _FS_RELATIVE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive branch
        raise ImportError("Unable to load scripts._filesystem helper")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    write_secure_text = module.write_secure_text  # type: ignore[attr-defined]

SARIF_PATH = Path("semgrep.sarif")
_FD_PREFIXES = {"fd", "pipe"}
_FD_PATH_PATTERN = re.compile(
    r"^/(?:proc/(?:self|thread-self|\d+)/fd|dev/fd)/(\d+)$"
)

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


def write_github_output(
    target: Path | int,
    *,
    upload: bool,
    findings: int,
    sarif_path: Path,
) -> None:
    """Append outputs consumed by subsequent workflow steps.

    Parameters
    ----------
    target:
        Either the filesystem path backing ``GITHUB_OUTPUT`` or a numeric file
        descriptor exposed by GitHub's workflow runner.  The latter is
        normalised from ``fd:X`` specifications when callers pass a file
        descriptor explicitly.
    """

    payload = (
        f"upload={'true' if upload else 'false'}\n"
        f"result_count={findings}\n"
        f"sarif_path={sarif_path}\n"
    )

    if isinstance(target, int):
        try:
            dup_fd = os.dup(target)
        except OSError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"Unable to duplicate GitHub output file descriptor {target}: {exc}"
            ) from exc

        try:
            with os.fdopen(dup_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                f"Unable to write Semgrep outputs via descriptor {target}: {exc}"
            ) from exc

        return

    write_secure_text(
        target,
        payload,
        append=True,
        allow_special_files=True,
    )


def _descriptor_from_path(candidate: Path) -> int | None:
    """Return a file descriptor when *candidate* points to ``/proc/*/fd``."""

    normalized = os.path.normpath(str(candidate))
    match = _FD_PATH_PATTERN.match(normalized)
    if match is None:
        return None

    try:
        fd = int(match.group(1), 10)
    except ValueError:  # pragma: no cover - defensive guard
        return None

    if fd < 0:
        return None

    try:
        os.fstat(fd)
    except OSError:
        return None

    return fd


def _normalize_github_output(value: Path | str | None) -> Path | int | None:
    """Return a usable GitHub output target or ``None`` when invalid."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    prefix, _, descriptor = text.partition(":")
    if prefix in _FD_PREFIXES and descriptor:
        descriptor = descriptor.strip()
        try:
            fd = int(descriptor, 10)
        except ValueError:
            return None

        if fd < 0:
            return None

        return fd

    path = Path(text)

    # ``Path("")`` normalizes to ``Path(".")`` which is not a real file target and
    # causes ``os.open`` to raise ``IsADirectoryError``.  Treat it as missing so the
    # script can run outside of GitHub Actions where ``GITHUB_OUTPUT`` is unset.
    if path == Path("."):
        return None

    if path.exists() and path.is_dir():
        return None

    descriptor = _descriptor_from_path(path)
    if descriptor is not None:
        return descriptor

    # GitHub Actions may expose ``GITHUB_OUTPUT`` as a symlink that ultimately
    # resolves to the writable command file hosted in the runner workspace.
    # ``Path.resolve(strict=False)`` safely follows the link even if the target
    # does not yet exist (for example when the file will be created by this
    # script).  Falling back to the original ``path`` would later trigger the
    # security guard inside :func:`write_secure_text` which refuses to operate
    # on symlinks, causing the workflow step to fail.  Resolving here keeps the
    # guardrails for regular usage while still supporting the CI environment.
    if path.is_symlink():
        try:
            target_text = os.readlink(path)
        except OSError:
            target_text = None
        else:
            target_path = Path(target_text)
            if not target_path.is_absolute():
                target_path = (path.parent / target_path).resolve(strict=False)

            descriptor = _descriptor_from_path(target_path)
            if descriptor is not None:
                return descriptor

        try:
            resolved = path.resolve(strict=False)
        except OSError:
            return None

        descriptor = _descriptor_from_path(resolved)
        if descriptor is not None:
            return descriptor

        if resolved.exists() and resolved.is_dir():
            return None

        path = resolved

    return path


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
