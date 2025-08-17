from pathlib import Path
import logging

import gptoss_check.check_code as check_code
from gptoss_check import main as gptoss_main


def test_skip_message(caplog, tmp_path):
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=true")
    with caplog.at_level(logging.INFO):
        gptoss_main.main(config_path=cfg)
    assert "skipped" in caplog.text.lower()


def test_run_message(caplog, tmp_path, monkeypatch):
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=false")

    called = []

    def fake_run():
        called.append(True)

    monkeypatch.setattr(check_code, "run", fake_run)
    monkeypatch.setattr(check_code, "wait_for_api", lambda *args, **kwargs: None)
    monkeypatch.setenv("GPT_OSS_API", "http://gptoss:8000")
    with caplog.at_level(logging.INFO):
        gptoss_main.main(config_path=cfg)
    assert "Running GPT-OSS check" in caplog.text
    assert "GPT-OSS check completed" in caplog.text
    assert called


def test_run_without_api(monkeypatch, caplog):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with caplog.at_level(logging.WARNING):
        check_code.run()
    assert "GPT_OSS_API" in caplog.text
