import os
import secrets
import shutil
# Bandit note: subprocess используется для контролируемого вызова git.
import subprocess  # nosec B404
from pathlib import Path
from textwrap import dedent


BASH_EXECUTABLE = shutil.which("bash") or "/bin/bash"


def _run_dependabot(script: Path, env: dict[str, str]):
    """Выполнить скрипт dependabot через абсолютный путь до bash."""

    # Bandit note: команда git фиксирована и использует доверенные аргументы.
    return subprocess.run(  # nosec B603
        [BASH_EXECUTABLE, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )


def _fake_curl(
    tmp_path: Path, status: str = "422", body: str = "{}", exit_code: int = 0
) -> Path:
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
            exit {exit_code}
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

    # Bandit note - the script executes within a temporary environment controlled by the test.
    proc = _run_dependabot(script, env)

    assert proc.returncode == 1
    assert "TOKEN or GITHUB_TOKEN is not set" in proc.stderr


def test_run_dependabot_ignores_already_queued_requests(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    fake_path = _fake_curl(
        tmp_path,
        status="422",
        body='{"message":"Update already queued"}',
        exit_code=22,
    )
    summary = tmp_path / "summary.md"
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    # Bandit note - a non-sensitive placeholder token is sufficient for test isolation.
    env["TOKEN"] = secrets.token_hex(8)
    env["PATH"] = f"{fake_path}:{env['PATH']}"
    env["GITHUB_STEP_SUMMARY"] = str(summary)

    # Bandit note - repeated invocation uses the same controlled environment as above.
    proc = _run_dependabot(script, env)

    assert proc.returncode == 0
    assert "status 422" in proc.stderr
    assert "Update already queued" in summary.read_text()


def test_run_dependabot_handles_missing_endpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    fake_path = _fake_curl(tmp_path, status="404", body='{"message":"Not Found"}', exit_code=22)
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    env["TOKEN"] = secrets.token_hex(8)
    env["PATH"] = f"{fake_path}:{env['PATH']}"

    proc = _run_dependabot(script, env)

    assert proc.returncode == 0
    assert "Dependabot endpoint returned 404" in proc.stderr


def test_run_dependabot_propagates_unexpected_curl_failure(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dependabot.sh"
    fake_path = _fake_curl(tmp_path, status="000", body="{}", exit_code=7)
    env = os.environ.copy()
    env["GITHUB_REPOSITORY"] = "owner/repo"
    env["TOKEN"] = secrets.token_hex(8)
    env["PATH"] = f"{fake_path}:{env['PATH']}"

    proc = _run_dependabot(script, env)

    assert proc.returncode == 7
    assert "curl failed with exit code 7" in proc.stderr
