import os
import subprocess
from unittest.mock import MagicMock, call

import pytest


@pytest.fixture
def docker_env(monkeypatch):
    monkeypatch.setenv("AVERINALEKS", "fake_user")
    monkeypatch.setenv("BOT", "fake_password")


@pytest.fixture
def run_mock(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock)
    return mock


def test_build_and_push(tmp_path, docker_env, run_mock):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM alpine:3.18\nRUN echo test > /test.txt\n")
    image = f"{os.environ['AVERINALEKS']}/bot-test-image:latest"

    subprocess.run(["docker", "build", "-t", image, str(tmp_path)], check=True)
    subprocess.run(
        ["docker", "login", "-u", os.environ["AVERINALEKS"], "--password-stdin"],
        input=os.environ["BOT"].encode(),
        check=True,
    )
    subprocess.run(["docker", "push", image], check=True)

    assert run_mock.call_args_list == [
        call(["docker", "build", "-t", image, str(tmp_path)], check=True),
        call(
            ["docker", "login", "-u", os.environ["AVERINALEKS"], "--password-stdin"],
            input=os.environ["BOT"].encode(),
            check=True,
        ),
        call(["docker", "push", image], check=True),
    ]
