from __future__ import annotations

import json

import pytest

from scripts import trivy_sarif_summary


def test_count_results_counts_all_available_entries() -> None:
    runs = [
        {"results": [{"id": 1}, {"id": 2}]},
        {"results": ()},
        {"results": None},
        {"results": "should be ignored"},
        {"results": [{"id": 3}]},
    ]

    assert trivy_sarif_summary.count_results(runs) == 3


def test_main_with_valid_sarif(tmp_path, capsys) -> None:
    sarif_path = tmp_path / "report.sarif"
    sarif_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"results": [{"id": "one"}]},
                    {"results": [{"id": "two"}, {"id": "three"}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert trivy_sarif_summary.main([str(sarif_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "3"
    assert captured.err == ""


@pytest.mark.parametrize(
    "contents",
    ["", "not-json", json.dumps({"unexpected": "payload"})],
)
def test_main_handles_invalid_inputs_gracefully(tmp_path, capsys, contents) -> None:
    sarif_path = tmp_path / "invalid.sarif"
    sarif_path.write_text(contents, encoding="utf-8")

    assert trivy_sarif_summary.main([str(sarif_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "0"
    assert "warning:" in captured.err
