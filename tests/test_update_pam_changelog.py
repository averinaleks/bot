from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from docker.scripts import update_pam_changelog


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
        return datetime(2025, 1, 2, 3, 4, 5, tzinfo=tz)


@pytest.fixture(autouse=True)
def _freeze_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_pam_changelog, "datetime", _FixedDatetime)


def _prepare_changelog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> Path:
    changelog = tmp_path / "debian" / "changelog"
    changelog.parent.mkdir(parents=True)
    changelog.write_text(content)
    monkeypatch.setattr(update_pam_changelog, "CHANGELOG_PATH", changelog)
    return changelog


def test_appends_entry_with_current_base_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    existing = (
        "pam (1.5.3-5ubuntu5.6) noble; urgency=medium\n\n"
        "  * Existing upstream note.\n\n"
        " -- Upstream Maintainer <maint@example.com>  Mon, 13 Jan 2025 11:10:09 +0000\n"
    )
    changelog = _prepare_changelog(monkeypatch, tmp_path, existing)

    update_pam_changelog.main()

    result = changelog.read_text()
    assert result.startswith(
        "pam (1.5.3-5ubuntu5.6+bot1) noble; urgency=medium\n\n"
        "  * Pull in upstream commits b3020da7 and b7b96362 to harden pam_unix\n"
        "    against CVE-2024-10041 and retain the pam_access nodns fix for\n"
        "    CVE-2024-10963.\n\n"
        " -- Security Bot <security@example.com>  Thu, 02 Jan 2025 03:04:05 +0000\n\n"
    )
    assert existing in result


def test_is_idempotent_when_entry_already_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patched = (
        "pam (1.5.3-5ubuntu5.6+bot1) noble; urgency=medium\n\n"
        "  * Pull in upstream commits b3020da7 and b7b96362 to harden pam_unix\n"
        "    against CVE-2024-10041 and retain the pam_access nodns fix for\n"
        "    CVE-2024-10963.\n\n"
        " -- Security Bot <security@example.com>  Thu, 02 Jan 2025 03:04:05 +0000\n\n"
        "pam (1.5.3-5ubuntu5.6) noble; urgency=medium\n\n"
        "  * Existing upstream note.\n\n"
        " -- Upstream Maintainer <maint@example.com>  Mon, 13 Jan 2025 11:10:09 +0000\n"
    )
    changelog = _prepare_changelog(monkeypatch, tmp_path, patched)

    update_pam_changelog.main()

    assert changelog.read_text() == patched
