from pathlib import Path

WORKFLOW_PATH = Path('.github/workflows/gptoss_review.yml')


def test_gptoss_workflow_uses_python3() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert (
        'python3 "${helpers_dir}/scripts/gptoss_mock_server.py"' in workflow_text
        or 'script_path="${helpers_dir}/scripts/gptoss_mock_server.py"' in workflow_text
    )
    assert (
        'python3 "${helpers_dir}/scripts/prepare_gptoss_diff.py"' in workflow_text
        or 'script_path="${helpers_dir}/scripts/prepare_gptoss_diff.py"' in workflow_text
    )
    assert (
        'python3 "${helpers_dir}/scripts/run_gptoss_review.py"' in workflow_text
        or 'script_path="${helpers_dir}/scripts/run_gptoss_review.py"' in workflow_text
    )
    assert 'python3 "$script_path"' in workflow_text
    assert 'python "${helpers_dir}/scripts/gptoss_mock_server.py"' not in workflow_text
    assert 'python "${helpers_dir}/scripts/prepare_gptoss_diff.py"' not in workflow_text
    assert 'python "${helpers_dir}/scripts/run_gptoss_review.py"' not in workflow_text
    assert 'python "$script_path"' not in workflow_text
    assert 'HELPERS_DIR: gptoss_helpers' in workflow_text
    assert 'path: ${{ env.HELPERS_DIR }}' in workflow_text


def test_pr_status_step_has_missing_script_guard() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert '${helpers_dir}/scripts/check_pr_status.py' in workflow_text
    assert '::notice::Скрипт ${helpers_dir}/scripts/check_pr_status.py не найден' in workflow_text


def test_helpers_are_preserved_outside_workspace() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert 'runner_temp%/}/gptoss_helpers' in workflow_text
    assert "printf 'HELPERS_DIR=%s\\n' \"$safe_dir\" >> \"$GITHUB_ENV\"" in workflow_text


def test_pr_status_step_sets_outputs_on_failure() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    marker_options = [
        'if ! python3 "${helpers_dir}/scripts/check_pr_status.py"',
        'if ! python3 "$script_path"',
    ]
    marker = next((candidate for candidate in marker_options if candidate in workflow_text), None)
    assert marker is not None, 'missing PR status failure handler'
    next_step = '      - name: Checkout PR head'

    start = workflow_text.index(marker)
    end = workflow_text.index(next_step, start)
    failure_block = workflow_text[start:end]

    assert '::warning::Проверка статуса PR завершилась с ошибкой' in failure_block
    assert 'echo "skip=true"' in failure_block
    assert 'echo "head_sha="' in failure_block
    assert failure_block.count('exit 0') >= 1


def test_pr_status_step_skips_pull_request_target() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    guard_snippet = 'GITHUB_EVENT_NAME:-}" = "pull_request_target"'
    notice_message = 'Workflow triggered for pull_request_target – пропускаю проверку PR'

    assert guard_snippet in workflow_text
    assert notice_message in workflow_text


def test_pr_status_step_handles_missing_repository_env() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert 'repo="${REPOSITORY:-}"' in workflow_text
    assert 'token="${GITHUB_TOKEN:-}"' in workflow_text
    assert 'Переменная REPOSITORY не задана – пропускаю проверку PR' in workflow_text


def test_review_job_runs_only_for_supported_events() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert "needs.evaluate.outputs.run_review == 'true'" in workflow_text
    assert "&& needs.evaluate.outputs.skip_reason == ''" in workflow_text
    assert "github.event_name != 'pull_request_target'" in workflow_text


def test_skip_job_covers_target_events() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    condition = (
        "needs.evaluate.outputs.skip_reason != '' || github.event_name == 'pull_request_target'"
    )

    assert condition in workflow_text


def test_pr_status_step_gates_supported_events() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert "Проверка статуса PR" in workflow_text
    assert "github.event_name != 'pull_request_target'" in workflow_text
    assert "github.event_name == 'pull_request'" in workflow_text
    assert "github.event_name == 'issue_comment'" in workflow_text


def test_issue_comment_requires_pull_request_reference() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert 'ISSUE_IS_PR: ${{ github.event.issue.pull_request != null }}' in workflow_text
    assert 'issue_is_pr="${ISSUE_IS_PR:-false}"' in workflow_text
    assert 'Комментарий не относится к pull request' in workflow_text


def test_issue_comment_command_is_case_insensitive() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert 'comment_body_lower="${comment_body,,}"' in workflow_text
    assert '[[ "$comment_body_lower" != *"/llm-review"* ]]' in workflow_text
