import json

from scripts import disable_desktop_credential_helper as helper


def test_malformed_config_is_rebuilt(tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "docker-config"
    config_dir.mkdir()
    config_path = config_dir / helper.CONFIG_FILENAME
    config_path.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setenv(helper.DOCKER_CONFIG_ENV, str(config_dir))

    helper.main()

    backups = list(config_dir.glob("config.json.backup-" + "*"))
    assert backups, "backup of invalid config should be created"

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    assert loaded == {}

    output = capsys.readouterr().out
    assert "Invalid Docker config" in output
    assert "Rewrote Docker config" in output


def test_removes_desktop_helpers_and_backups(tmp_path, monkeypatch):
    config_dir = tmp_path / "docker-config"
    config_dir.mkdir()
    config_path = config_dir / helper.CONFIG_FILENAME
    config = {
        "credsStore": "desktop",
        "credStore": "Desktop",
        "credHelpers": {
            "registry-1.docker.io": "desktop",
            "ghcr.io": "desktop.exe",
            "keep": "pass",
        },
        "auths": {"registry-1.docker.io": {}},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setenv(helper.DOCKER_CONFIG_ENV, str(config_dir))

    helper.main()

    backups = list(config_dir.glob("config.json.backup-*"))
    assert backups, "backup should be created before modifying config"

    rewritten = json.loads(config_path.read_text(encoding="utf-8"))
    assert "credsStore" not in rewritten
    assert "credStore" not in rewritten
    assert rewritten.get("credHelpers") == {"keep": "pass"}
    assert rewritten["auths"] == {"registry-1.docker.io": {}}
