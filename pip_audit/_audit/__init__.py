"""Minimal audit module for pip_audit stub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

from pip_audit._service.interface import (
    ResolvedDependency,
    SkippedDependency,
    VulnerabilityResult,
)


@dataclass(frozen=True)
class AuditOptions:
    """Simplified representation of pip-audit options."""

    dry_run: bool = False


class Auditor:
    """Auditor implementation that returns no vulnerabilities."""

    def __init__(self, service, options: AuditOptions | None = None):
        self._service = service
        self._options = options or AuditOptions()

    def audit(self, source) -> Iterator[Tuple[ResolvedDependency | SkippedDependency, Iterable[VulnerabilityResult]]]:
        """Return an empty iterator of audit results."""

        return iter(())
