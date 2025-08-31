import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch


def test_run_dependabot_requires_token():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    env.pop("TOKEN", None)

    mock_proc = Mock(
        returncode=1,
        stderr="TOKEN is not set; export a PAT with repo and security_events scopes\n",
    )
    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        proc = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env=env
        )

    mock_run.assert_called_once_with(
        ["bash", str(script)], capture_output=True, text=True, env=env
    )
    assert "TOKEN is not set" in proc.stderr
