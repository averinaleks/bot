import os
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is not installed")
@pytest.mark.skipif(
    not os.getenv("AVERINALEKS") or not os.getenv("BOT"),
    reason="Environment variables AVERINALEKS and BOT must be set for Docker Hub",
)
def test_build_and_push(tmp_path):
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
