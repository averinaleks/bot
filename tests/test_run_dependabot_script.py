import os
import subprocess  # nosec B404
from pathlib import Path
from textwrap import dedent


def _fake_curl(tmp_path: Path, status: str = "422", body: str = "{}") -> Path:
    """Create a fake curl executable that records the output path."""

    script = tmp_path / "curl"
    script.write_text(
        dedent(
            f"""
            #!/usr/bin/env bash
            set -euo pipefail
            out_file=""
            while (($#)); do
              case "$1" in
                -o)
                  shift
                  out_file="$1"
                  ;;
              esac
              shift || true
            done
            cat <<'EOF' > "${{out_file}}"
            {body}
            EOF
            printf '%s' '{status}'
            """
        ).lstrip()
    )
    script.chmod(0o755)
    return script.parent


def test_run_dependabot_requires_token():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    env.pop("TOKEN", None)
    env.pop("GITHUB_TOKEN", None)

    # Bandit: the script executes within a temporary environment controlled by the test.
    proc = subprocess.run(  # nosec
        ["bash", str(script)], capture_output=True, text=True, env=env
    )

    assert proc.returncode == 1
    assert "TOKEN or GITHUB_TOKEN is not set" in proc.stderr


def test_run_dependabot_ignores_already_queued_requests(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    fake_path = _fake_curl(
        tmp_path, status="422", body='{"message":"Update already queued"}'
    )
    summary = tmp_path / "summary.md"
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    # Bandit: a non-sensitive placeholder token is sufficient for test isolation.
    env["TOKEN"] = "dummy"  # nosec
    env["PATH"] = f"{fake_path}:{env['PATH']}"
    env["GITHUB_STEP_SUMMARY"] = str(summary)

    # Bandit: repeated invocation uses the same controlled environment as above.
    proc = subprocess.run(  # nosec
        ["bash", str(script)], capture_output=True, text=True, env=env
    )

    assert proc.returncode == 0
    assert "status 422" in proc.stderr
    assert "Update already queued" in summary.read_text()
