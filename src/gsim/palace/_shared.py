"""Dependency-light helpers shared by Palace package internals.

This module owns only scalar coercion, JSON conversion, path formatting, and
small record helpers that are reused across Palace resolve and typed-data code.
It does not know about solver artifacts, problem reports, display policy,
physics columns, or handoff workflows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np


def as_mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` as a plain dict when it is mapping-like."""
    return dict(value) if isinstance(value, Mapping) else {}


def optional_float(value: Any) -> float | None:
    """Return a finite float or ``None`` when conversion is not meaningful."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def optional_int(value: Any) -> int | None:
    """Return a finite integer or ``None`` when conversion is not meaningful."""
    numeric = optional_float(value)
    return None if numeric is None else int(numeric)


def optional_str(value: Any) -> str | None:
    """Return ``value`` as a string unless it is missing."""
    if is_missing_value(value):
        return None
    return str(value)


def is_missing_value(value: Any) -> bool:
    """Return whether a scalar value should be treated as missing."""
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ImportError, TypeError, ValueError):
        pass
    try:
        if bool(np.isscalar(value)) and bool(np.isnan(value)):
            return True
    except (TypeError, ValueError):
        return False
    return False


def optional_str_pair(value: Any) -> tuple[str, str] | None:
    """Return a two-item sequence as a string pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return str(value[0]), str(value[1])


def finite_min(values: Any) -> float | None:
    """Return the finite minimum for numeric array-like values."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def int_tuple(value: Any) -> tuple[int, ...]:
    """Return integer items from a scalar or iterable, excluding booleans."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return (int(value),)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return ()
    return tuple(
        int(item)
        for item in value
        if isinstance(item, Integral) and not isinstance(item, bool)
    )


def json_ready(value: Any) -> Any:
    """Return a JSON-serializable version of a nested value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [json_ready(item) for item in value]
    return value


def path_value(value: Any) -> str:
    """Return a stable string representation for path-like sidecar fields."""
    return str(value)


def relative_path_or_name(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible, otherwise its name."""
    root = root if root.is_dir() or not root.suffix else root.parent
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_column_key(value: object) -> str:
    """Return a deterministic table column key for metric names."""
    key = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower()
    return key or "value"


def record_value(value: Any) -> Any:
    """Return a scalar or JSON string suitable for one CSV/JSONL record cell."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(json_ready(value), sort_keys=True)


def duplicate_values(values: Iterable[str]) -> tuple[str, ...]:
    """Return duplicate values in first-observed order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
