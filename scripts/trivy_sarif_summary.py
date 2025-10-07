#!/usr/bin/env python3
"""Utilities for summarizing Trivy SARIF reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from pathlib import Path

RunMapping = Mapping[str, object]


def _iter_results(runs: Iterable[RunMapping]) -> Iterator[object]:
    """Yield each result entry across the provided SARIF runs."""

    for run in runs:
        if not isinstance(run, Mapping):
            continue
        results = run.get("results")
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
            yield from results
        elif results is None:
            continue
        elif isinstance(results, Iterable):
            yield from results


def count_results(runs: Iterable[RunMapping]) -> int:
    """Return the number of results entries contained in ``runs``."""

    return sum(1 for _ in _iter_results(runs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Trivy SARIF output")
    parser.add_argument("sarif", type=Path, help="Path to the SARIF report produced by Trivy")
    args = parser.parse_args()

    with args.sarif.open("r", encoding="utf-8") as sarif_file:
        sarif = json.load(sarif_file)

    runs = sarif.get("runs", []) if isinstance(sarif, MutableMapping) else []
    print(count_results(runs))


if __name__ == "__main__":
    main()
