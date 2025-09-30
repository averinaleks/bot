"""Validation helpers for sanitising feature and label payloads."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# Hard limits to avoid unbounded memory usage when parsing untrusted payloads.
MAX_FEATURES_PER_SAMPLE = 256
MAX_SAMPLES = 10_000


class FeatureValidationError(ValueError):
    """Raised when feature or label payloads fail validation."""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def coerce_float(value: Any) -> float:
    """Return ``value`` coerced to ``float`` ensuring it is finite."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - depends on input type
        raise FeatureValidationError("value must be numeric") from exc

    if not math.isfinite(result):
        raise FeatureValidationError("value must be finite")
    return result


def coerce_feature_vector(
    raw: Any,
    *,
    max_features: int = MAX_FEATURES_PER_SAMPLE,
) -> np.ndarray:
    """Return a ``(1, n)`` feature array built from *raw*.

    Scalars are accepted and treated as single-element vectors. Sequences must contain
    numeric items only.
    """

    if raw is None:
        raise FeatureValidationError("features are required")

    if _is_sequence(raw):
        values = [coerce_float(item) for item in raw]
    else:
        values = [coerce_float(raw)]

    if not values:
        raise FeatureValidationError("feature vector is empty")
    if len(values) > max_features:
        raise FeatureValidationError("too many feature values")

    return np.asarray([values], dtype=np.float32)


def coerce_feature_matrix(
    raw: Any,
    *,
    max_rows: int = MAX_SAMPLES,
    max_features: int = MAX_FEATURES_PER_SAMPLE,
) -> np.ndarray:
    """Return a ``(n, m)`` array from *raw* ensuring numeric content."""

    if raw is None:
        raise FeatureValidationError("features are required")

    if not _is_sequence(raw):
        raise FeatureValidationError("feature matrix must be a sequence")

    rows: list[list[float]] = []

    if not raw:
        raise FeatureValidationError("feature matrix is empty")

    # ``raw`` may be a flat sequence (representing a single row) or a sequence of
    # sequences. Handle both forms explicitly.
    if all(not _is_sequence(item) for item in raw):
        row = [coerce_float(item) for item in raw]
        if len(row) > max_features:
            raise FeatureValidationError("too many feature values")
        rows.append(row)
    else:
        for row in raw:
            if not _is_sequence(row):
                raise FeatureValidationError("feature rows must be sequences")
            values = [coerce_float(item) for item in row]
            if not values:
                raise FeatureValidationError("feature row is empty")
            if len(values) > max_features:
                raise FeatureValidationError("too many feature values")
            rows.append(values)

    if len(rows) > max_rows:
        raise FeatureValidationError("too many feature rows")

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    elif matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix


def coerce_label_vector(
    raw: Any,
    *,
    max_rows: int = MAX_SAMPLES,
) -> np.ndarray:
    """Return a 1D array of labels built from *raw* values."""

    if raw is None:
        raise FeatureValidationError("labels are required")
    if not _is_sequence(raw):
        raise FeatureValidationError("labels must be a sequence")

    values = [coerce_float(item) for item in raw]
    if not values:
        raise FeatureValidationError("label vector is empty")
    if len(values) > max_rows:
        raise FeatureValidationError("too many label values")

    return np.asarray(values, dtype=np.float32)


__all__ = [
    "FeatureValidationError",
    "MAX_FEATURES_PER_SAMPLE",
    "MAX_SAMPLES",
    "coerce_feature_matrix",
    "coerce_feature_vector",
    "coerce_label_vector",
    "coerce_float",
]
