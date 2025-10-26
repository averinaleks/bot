"""Minimal service interface definitions for pip_audit stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


class ServiceError(Exception):
    """Base service error raised by the stub."""


@dataclass(frozen=True)
class ResolvedDependency:
    """Represents a dependency resolved by the audit service."""

    name: str
    version: str

    def is_skipped(self) -> bool:
        return False


@dataclass(frozen=True)
class SkippedDependency:
    """Represents a dependency that could not be audited."""

    name: str
    skip_reason: str

    def is_skipped(self) -> bool:
        return True


@dataclass(frozen=True)
class VulnerabilityResult:
    """Represents a single vulnerability affecting a dependency."""

    id: str
    description: str = ""
    severity: str | None = None
    fix_versions: Sequence[str] = field(default_factory=tuple)
    aliases: Iterable[str] = field(default_factory=tuple)
