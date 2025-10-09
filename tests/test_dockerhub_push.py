import os
import shutil
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import pytest


DOCKER_EXECUTABLE = shutil.which("docker") or "/usr/bin/docker"


@dataclass
class _RecordedCall:
    command: list[str]
    check: bool
    kwargs: dict


class _CommandRecorder:
    """Collect docker invocations without spawning external processes."""

    def __init__(self) -> None:
        self.calls: list[_RecordedCall] = []

    def runner(self, command: list[str], *, check: bool = True, **kwargs):
        self.calls.append(_RecordedCall(list(command), check, dict(kwargs)))
        return SimpleNamespace(returncode=0)


def _run_docker(
    runner: Callable[..., SimpleNamespace], *args: str, check: bool = True, **kwargs
):
    """Build a docker command and delegate execution to *runner*.

    ``runner`` mirrors :func:`subprocess.run` but is provided by the tests so the
    helper never spawns real processes during security scans.
    """

    command = [DOCKER_EXECUTABLE, *args]
    return runner(command, check=check, **kwargs)


@pytest.fixture
def docker_env(monkeypatch):
    monkeypatch.setenv("DOCKERHUB_USERNAME", "fake_user")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "fake_password")


@pytest.fixture
def command_recorder():
    return _CommandRecorder()


def test_build_and_push(tmp_path, docker_env, command_recorder):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM alpine:3.18\nRUN echo test > /test.txt\n")
    image = f"{os.environ['DOCKERHUB_USERNAME']}/bot-test-image:latest"

    runner = command_recorder.runner

    _run_docker(runner, "build", "-t", image, str(tmp_path))
    _run_docker(
        runner,
        "login",
        "-u",
        os.environ["DOCKERHUB_USERNAME"],
        "--password-stdin",
        input=os.environ["DOCKERHUB_TOKEN"].encode(),
    )
    _run_docker(runner, "push", image)

    assert command_recorder.calls == [
        _RecordedCall(
            [DOCKER_EXECUTABLE, "build", "-t", image, str(tmp_path)],
            True,
            {},
        ),
        _RecordedCall(
            [
                DOCKER_EXECUTABLE,
                "login",
                "-u",
                os.environ["DOCKERHUB_USERNAME"],
                "--password-stdin",
            ],
            True,
            {"input": os.environ["DOCKERHUB_TOKEN"].encode()},
        ),
        _RecordedCall([DOCKER_EXECUTABLE, "push", image], True, {}),
    ]
