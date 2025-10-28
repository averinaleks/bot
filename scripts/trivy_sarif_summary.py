#!/usr/bin/env python3
"""Utilities for summarizing Trivy SARIF reports."""

from __future__ import annotations

import argparse
import json
import sys
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
        elif isinstance(results, Iterable) and not isinstance(
            results, (str, bytes, bytearray)
        ):
            yield from results


def count_results(runs: Iterable[RunMapping]) -> int:
    """Return the number of results entries contained in ``runs``."""

    return sum(1 for _ in _iter_results(runs))


def _load_sarif_runs(path: Path) -> Iterable[RunMapping]:
    """Return an iterable of SARIF runs extracted from *path*.

    The helper is intentionally defensive: empty files, invalid JSON payloads or
    unexpected top-level structures are treated as an absence of runs rather
    than hard failures.  A short explanation is written to ``stderr`` so the
    workflow logs still reveal the underlying issue without breaking the step
    that consumes the result count.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"warning: SARIF report not found at {path}", file=sys.stderr)
        return ()
    except OSError as exc:  # pragma: no cover - unexpected IO failure
        print(f"warning: failed to read SARIF report at {path}: {exc}", file=sys.stderr)
        return ()

    if not raw.strip():
        print(f"warning: SARIF report at {path} is empty", file=sys.stderr)
        return ()

    try:
        sarif = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"warning: SARIF report at {path} is not valid JSON: {exc.msg}",
            file=sys.stderr,
        )
        return ()

    if not isinstance(sarif, MutableMapping):
        print(
            f"warning: unexpected SARIF payload type {type(sarif).__name__} at {path}",
            file=sys.stderr,
        )
        return ()

    if "runs" not in sarif:
        print("warning: SARIF payload is missing 'runs' entry", file=sys.stderr)
    runs = sarif.get("runs", [])
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes, bytearray)):
        return (run for run in runs if isinstance(run, Mapping))
    if runs is None:
        return ()
    if isinstance(runs, Iterable):
        return (run for run in runs if isinstance(run, Mapping))

    print(
        f"warning: SARIF 'runs' entry has unexpected type {type(runs).__name__}",
        file=sys.stderr,
    )
    return ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Trivy SARIF output")
    parser.add_argument("sarif", type=Path, help="Path to the SARIF report produced by Trivy")
    args = parser.parse_args(argv)

    runs = _load_sarif_runs(args.sarif)
    print(count_results(runs))
    return 0

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
