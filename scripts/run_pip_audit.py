"""Run pip-audit with repository specific tweaks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

try:
    from pip_audit._audit import AuditOptions, Auditor
    from pip_audit._dependency_source import (
        DependencySourceError,
        PipSource,
    )
    from pip_audit._format import JsonFormat
    from pip_audit._service import PyPIService
    from pip_audit._service.interface import (
        ResolvedDependency,
        ServiceError,
        SkippedDependency,
        VulnerabilityResult,
    )
    from pip_audit._state import AuditState
except ModuleNotFoundError:  # pragma: no cover - fallback for local invocation
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from pip_audit._audit import AuditOptions, Auditor
    from pip_audit._dependency_source import (
        DependencySourceError,
        PipSource,
    )
    from pip_audit._format import JsonFormat
    from pip_audit._service import PyPIService
    from pip_audit._service.interface import (
        ResolvedDependency,
        ServiceError,
        SkippedDependency,
        VulnerabilityResult,
    )
    from pip_audit._state import AuditState

try:  # pragma: no cover - exercised when helpers ship without the module
    from scripts._filesystem import write_secure_text
except Exception:  # pragma: no cover - fallback for legacy commits
    import errno as _errno
    import os as _os
    import stat as _stat

    def write_secure_text(
        path: Path,
        content: str,
        *,
        append: bool = False,
        permissions: int = 0o600,
        encoding: str = "utf-8",
        dir_permissions: int | None = 0o700,
        allow_special_files: bool = False,
    ) -> None:
        if dir_permissions is not None:
            parent = path.parent
            if parent and parent != Path("."):
                parent.mkdir(parents=True, exist_ok=True, mode=dir_permissions)

        flags = _os.O_WRONLY | _os.O_CREAT
        flags |= _os.O_APPEND if append else _os.O_TRUNC

        fd = _os.open(path, flags, permissions)
        try:
            if hasattr(_os, "fchmod"):
                _os.fchmod(fd, permissions)
            else:  # pragma: no cover - Windows compatibility
                _os.chmod(path, permissions)
            info = _os.fstat(fd)
            if not _stat.S_ISREG(info.st_mode):
                if allow_special_files and _stat.S_ISFIFO(info.st_mode):
                    pass
                else:
                    raise OSError(_errno.EPERM, "target file must be a regular file")
            try:
                link_info = _os.lstat(path)
            except OSError:
                link_info = None
            if link_info is not None and _stat.S_ISLNK(link_info.st_mode):
                raise OSError(_errno.EPERM, "refusing to write through symlink")
            mode = "a" if append else "w"
            with _os.fdopen(fd, mode, encoding=encoding, closefd=False) as handle:
                handle.write(content)
        finally:
            _os.close(fd)


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


def _run_audit_once(
    strict: bool,
) -> Tuple[Dict[ResolvedDependency, List[VulnerabilityResult]], List[SkippedDependency]]:
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
) -> Tuple[Dict[ResolvedDependency, List[VulnerabilityResult]], List[SkippedDependency]]:
    last_error: ServiceError | None = None
    delay = _RETRY_BACKOFF_SECONDS

    for attempt in range(1, _MAX_AUDIT_RETRIES + 1):
        try:
            return _run_audit_once(strict=strict)
        except ServiceError as exc:
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

    if last_error is None:
        raise RuntimeError(
            "pip-audit retry loop exited without capturing the underlying error"
        )
    raise last_error


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_path: Path = args.output

    result, skipped = _run_audit(strict=args.strict)

    if skipped and args.strict:
        for entry in skipped:
            print(entry.skip_reason, file=sys.stderr)
        return 1

    vuln_count = sum(len(v) for v in result.values())
    pkg_count = sum(1 for vulns in result.values() if vulns)

    formatter = JsonFormat(output_desc=True, output_aliases=True)
    write_secure_text(
        output_path,
        json.dumps(json.loads(formatter.format(result, [])), indent=2),
    )

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
