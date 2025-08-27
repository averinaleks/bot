import os
import subprocess
from pathlib import Path

def test_run_dependabot_requires_token():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    env.pop("TOKEN", None)
    proc = subprocess.run([
        "bash",
        str(script),
    ], capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert "TOKEN is not set" in proc.stderr
