from __future__ import annotations

from pathlib import Path
import tempfile

from scripts.github_paths import allowed_github_directories, resolve_github_path


def test_resolve_github_path_accepts_runner_temp(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))

    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    event_path = runner_temp / "_github_workflow" / "event.json"
    event_path.parent.mkdir(parents=True)
    event_path.write_text("{}", encoding="utf-8")

    resolved = resolve_github_path(str(event_path))
    assert resolved == event_path.resolve()

    directories = allowed_github_directories()
    assert runner_temp.resolve() in directories


def test_resolve_github_path_rejects_untrusted_locations(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "nested" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))

    monkeypatch.delenv("RUNNER_TEMP", raising=False)
    tempfile.tempdir = str((tmp_path / "runner_temp").resolve())

    outside_root = tmp_path.parent / "outside"
    outside_root.mkdir(exist_ok=True)
    malicious = outside_root / "event.json"
    malicious.write_text("{}", encoding="utf-8")

    assert resolve_github_path(str(malicious)) is None
