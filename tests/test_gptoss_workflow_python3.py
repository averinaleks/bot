from pathlib import Path

WORKFLOW_PATH = Path('.github/workflows/gptoss_review.yml')


def test_gptoss_workflow_uses_python3() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert 'python3 "${helpers_dir}/scripts/gptoss_mock_server.py"' in workflow_text
    assert 'python3 "${helpers_dir}/scripts/prepare_gptoss_diff.py"' in workflow_text
    assert 'python3 "${helpers_dir}/scripts/run_gptoss_review.py"' in workflow_text
    assert 'python "${helpers_dir}/scripts/gptoss_mock_server.py"' not in workflow_text
    assert 'python "${helpers_dir}/scripts/prepare_gptoss_diff.py"' not in workflow_text
    assert 'python "${helpers_dir}/scripts/run_gptoss_review.py"' not in workflow_text
    assert 'HELPERS_DIR: gptoss_helpers' in workflow_text
    assert 'path: ${{ env.HELPERS_DIR }}' in workflow_text


def test_pr_status_step_has_missing_script_guard() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert '${helpers_dir}/scripts/check_pr_status.py' in workflow_text
    assert '::notice::Скрипт ${helpers_dir}/scripts/check_pr_status.py не найден' in workflow_text


def test_pr_status_step_sets_outputs_on_failure() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    marker = 'if ! python3 "${helpers_dir}/scripts/check_pr_status.py"'
    next_step = '      - name: Checkout PR head'

    assert marker in workflow_text, 'missing PR status failure handler'
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
