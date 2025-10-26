"""Run pip-audit with repository specific tweaks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast
from typing import TYPE_CHECKING


class PipAuditUnavailableError(RuntimeError):
    """Raised when pip-audit is not available in the current environment."""

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from pip_audit._audit import AuditOptions, Auditor
    from pip_audit._dependency_source import DependencySourceError, PipSource
    from pip_audit._format import JsonFormat
    from pip_audit._service import PyPIService
    from pip_audit._service.interface import (
        ResolvedDependency,
        ServiceError,
        SkippedDependency,
        VulnerabilityResult,
    )
    from pip_audit._state import AuditState


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pip-audit with project defaults.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pip-audit.json"),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when dependency metadata cannot be collected.",
    )
    return parser.parse_args(argv)


def _should_ignore_skip(spec: SkippedDependency) -> bool:
    name = spec.name.lower()
    if name != "torch":
        return False
    reason = spec.skip_reason.lower()
    return "could not be audited" in reason and "torch" in reason


def _should_ignore_vulnerability(
    spec: ResolvedDependency, vulnerability: VulnerabilityResult
) -> bool:
    if spec.name.lower() != "pip":
        return False

    identifiers = {vulnerability.id.lower(), *(alias.lower() for alias in vulnerability.aliases)}
    ignored_ids = {
        "ghsa-4xh5-x5gv-qwph",
        "cve-2025-8869",
        "pysec-2025-8869",
        "bit-pip-2025-8869",
    }
    return bool(identifiers & ignored_ids)


_MAX_AUDIT_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


def _load_pip_audit_dependencies() -> tuple[
    "AuditState",
    "Auditor",
    "AuditOptions",
    "DependencySourceError",
    "JsonFormat",
    "PyPIService",
    "PipSource",
    "ResolvedDependency",
    "SkippedDependency",
    "VulnerabilityResult",
]:
    try:
        from pip_audit._audit import AuditOptions, Auditor
        from pip_audit._dependency_source import DependencySourceError, PipSource
        from pip_audit._format import JsonFormat
        from pip_audit._service import PyPIService
        from pip_audit._service.interface import (
            ResolvedDependency,
            SkippedDependency,
            VulnerabilityResult,
        )
        from pip_audit._state import AuditState
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in environments without pip-audit
        raise PipAuditUnavailableError(
            "pip-audit is required to run this script. Install the 'pip-audit' package to use it."
        ) from exc

    return (
        AuditState,
        Auditor,
        AuditOptions,
        DependencySourceError,
        JsonFormat,
        PyPIService,
        PipSource,
        ResolvedDependency,
        SkippedDependency,
        VulnerabilityResult,
    )


def _run_audit_once(
    strict: bool,
) -> Tuple[Dict["ResolvedDependency", List["VulnerabilityResult"]], List["SkippedDependency"]]:
    (
        AuditState,
        Auditor,
        AuditOptions,
        DependencySourceError,
        JsonFormat,  # noqa: F841 - imported for consistency with type checking branch
        PyPIService,
        PipSource,
        ResolvedDependency,
        SkippedDependency,
        VulnerabilityResult,
    ) = _load_pip_audit_dependencies()

    state = AuditState()
    auditor = Auditor(PyPIService(), options=AuditOptions(dry_run=False))
    source = PipSource(state=state)

    result: Dict[ResolvedDependency, List[VulnerabilityResult]] = {}
    skipped: List[SkippedDependency] = []

    try:
        for spec, vulns in auditor.audit(source):
            if spec.is_skipped():
                spec = cast(SkippedDependency, spec)
                if _should_ignore_skip(spec):
                    continue
                skipped.append(spec)
                continue

            spec = cast(ResolvedDependency, spec)
            filtered_vulns = [v for v in vulns if not _should_ignore_vulnerability(spec, v)]
            result[spec] = filtered_vulns
    except DependencySourceError as exc:
        print(str(exc), file=sys.stderr)
        if strict:
            raise

    return result, skipped


def _run_audit(
    strict: bool,
) -> Tuple[Dict["ResolvedDependency", List["VulnerabilityResult"]], List["SkippedDependency"]]:
    last_error: Exception | None = None
    delay = _RETRY_BACKOFF_SECONDS

    for attempt in range(1, _MAX_AUDIT_RETRIES + 1):
        try:
            return _run_audit_once(strict=strict)
        except PipAuditUnavailableError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == _MAX_AUDIT_RETRIES:
                break

            wait_for = min(30.0, delay)
            print(
                f"pip-audit query failed (attempt {attempt}/{_MAX_AUDIT_RETRIES}): {exc}. "
                f"Retrying in {wait_for:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait_for)
            delay *= 2

    assert last_error is not None
    raise last_error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_path: Path = args.output

    (_, _, _, _, JsonFormat, _, _, _, _, _) = _load_pip_audit_dependencies()

    result, skipped = _run_audit(strict=args.strict)

    if skipped and args.strict:
        for entry in skipped:
            print(entry.skip_reason, file=sys.stderr)
        return 1

    vuln_count = sum(len(v) for v in result.values())
    pkg_count = sum(1 for vulns in result.values() if vulns)

    formatter = JsonFormat(output_desc=True, output_aliases=True)
    output_path.write_text(json.dumps(json.loads(formatter.format(result, [])), indent=2))

    if vuln_count:
        print(
            f"Found {vuln_count} known vulnerabilities in {pkg_count} package(s)",
            file=sys.stderr,
        )
        return 1

    if skipped:
        print(
            f"No known vulnerabilities found, {len(skipped)} dependencies skipped",
            file=sys.stderr,
        )
    else:
        print("No known vulnerabilities found", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
