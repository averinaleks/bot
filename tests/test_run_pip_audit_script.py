from __future__ import annotations

from typing import List, Tuple

import pytest

from pip_audit._service.interface import ResolvedDependency, ServiceError, SkippedDependency, VulnerabilityResult

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
