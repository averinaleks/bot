from pathlib import Path

WORKFLOW_PATH = Path('.github/workflows/gptoss_review.yml')


def test_gptoss_workflow_uses_python3() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding='utf-8')

    assert 'python3 scripts/gptoss_mock_server.py' in workflow_text
    assert 'python3 scripts/prepare_gptoss_diff.py' in workflow_text
    assert 'python3 scripts/run_gptoss_review.py' in workflow_text
    assert 'python scripts/gptoss_mock_server.py' not in workflow_text
    assert 'python scripts/prepare_gptoss_diff.py' not in workflow_text
    assert 'python scripts/run_gptoss_review.py' not in workflow_text
