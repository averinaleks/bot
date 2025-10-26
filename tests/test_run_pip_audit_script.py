from __future__ import annotations

from typing import List, Tuple

import pytest

try:
    from pip_audit._service.interface import (
        ResolvedDependency,
        ServiceError,
        SkippedDependency,
        VulnerabilityResult,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when pip-audit is not installed
    class ServiceError(Exception):
        """Fallback pip-audit service error used in tests when pip-audit is unavailable."""

    class ResolvedDependency:
        def __init__(self, name: str, version: str | None = None) -> None:
            self.name = name
            self.version = version

        def is_skipped(self) -> bool:  # pragma: no cover - simple fallback behaviour
            return False

    class SkippedDependency(ResolvedDependency):
        def __init__(self, name: str, skip_reason: str = "") -> None:
            super().__init__(name)
            self.skip_reason = skip_reason

        def is_skipped(self) -> bool:  # pragma: no cover - simple fallback behaviour
            return True

    class VulnerabilityResult:
        def __init__(self, identifier: str, aliases: List[str] | None = None) -> None:
            self.id = identifier
            self.aliases = aliases or []

from scripts import run_pip_audit


class _FakeServiceError(ServiceError):
    """Concrete subclass for testing since ServiceError is abstract."""

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return "boom"


def _fake_result() -> Tuple[dict[ResolvedDependency, List[VulnerabilityResult]], List[SkippedDependency]]:
    return {}, []


def test_run_audit_retries_after_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[bool] = []

    def fake_once(strict: bool):
        calls.append(strict)
        if len(calls) == 1:
            raise _FakeServiceError()
        return _fake_result()

    monkeypatch.setattr(run_pip_audit, "_run_audit_once", fake_once)
    monkeypatch.setattr(run_pip_audit.time, "sleep", lambda _: None)

    result = run_pip_audit._run_audit(strict=True)

    assert result == _fake_result()
    assert calls == [True, True]


def test_run_audit_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(strict: bool):
        raise _FakeServiceError()

    monkeypatch.setattr(run_pip_audit, "_run_audit_once", always_fail)
    monkeypatch.setattr(run_pip_audit.time, "sleep", lambda _: None)

    with pytest.raises(ServiceError):
        run_pip_audit._run_audit(strict=False)
