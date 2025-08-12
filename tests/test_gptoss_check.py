from pathlib import Path

import gptoss_check.check_code as check_code
from gptoss_check import main as gptoss_main


def test_skip_message(capsys, tmp_path):
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=true")
    gptoss_main.main(config_path=cfg)
    out = capsys.readouterr().out.lower()
    assert "skipped" in out


def test_run_message(capsys, tmp_path, monkeypatch):
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=false")

    called = []

    def fake_run():
        called.append(True)

    monkeypatch.setattr(check_code, "run", fake_run)
    monkeypatch.setattr(check_code, "wait_for_api", lambda *args, **kwargs: None)
    monkeypatch.setenv("GPT_OSS_API", "http://gptoss:8000")
    gptoss_main.main(config_path=cfg)
    out = capsys.readouterr().out
    assert "Running GPT-OSS check" in out
    assert "GPT-OSS check completed" in out
    assert called


def test_run_without_api(monkeypatch, capsys):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    check_code.run()
    out = capsys.readouterr().out
    assert "GPT_OSS_API" in out
