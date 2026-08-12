"""Write Resolve-owned Palace sweep sidecar metadata.

This module writes sweep point sidecars used by Resolve sweep summaries. The
sidecars describe sweep identity and path metadata that Resolve can audit later.

Run Stage handoff metadata, sbatch rendering, archive packaging, solver-result
physics parsing, and typed result data are owned by their respective run,
resolve, and results layers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gsim.palace._shared import duplicate_values, json_ready, optional_str
from gsim.palace.resolve.sources.sweep_models import (
    SWEEP_POINT_PATH_FIELDS,
    PalaceSweepPointSpec,
)


def write_palace_sweep_points(
    source: str | Path,
    points: Iterable[PalaceSweepPointSpec | Mapping[str, Any]],
    *,
    sweep_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    filename: str = "points.json",
) -> Path:
    """Write explicit point metadata for a point-local Palace sweep.

    This is the writer counterpart to :func:`load_palace_sweep_summary`. It
    records point identity and artifact locations only; it does not run Palace
    and does not inspect or summarize generated files.

    Args:
        source: Sweep root directory or direct JSON path.
        points: Point specs with explicit ``point_slug`` values.
        sweep_id: Optional stable sweep identifier.
        metadata: Additional JSON-friendly sweep metadata.
        filename: File name when ``source`` is a directory.

    Returns:
        Path to the written ``points.json`` file.
    """
    points_path = _sweep_points_path(source, filename=filename)
    payload: dict[str, Any] = {"schema_version": 1}
    if metadata is not None:
        for key, value in metadata.items():
            if key in {"schema_version", "points"}:
                msg = f"Sweep metadata key {key!r} is reserved"
                raise ValueError(msg)
            payload[str(key)] = json_ready(value)
    if sweep_id is not None:
        payload["sweep_id"] = str(sweep_id)
    point_rows = [_sweep_point_spec_row(point) for point in points]
    _raise_for_duplicate_sweep_point_slugs(point_rows)
    payload["points"] = point_rows

    points_path.parent.mkdir(parents=True, exist_ok=True)
    points_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return points_path


def _sweep_points_path(source: str | Path, *, filename: str) -> Path:
    """Resolve the sweep-points sidecar path."""
    path = Path(source)
    return path if path.suffix.lower() == ".json" else path / filename


def _sweep_point_spec_row(
    point: PalaceSweepPointSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize one sweep point specification to a JSON-ready row."""
    if isinstance(point, PalaceSweepPointSpec):
        row = point.to_dict()
    elif isinstance(point, Mapping):
        row = {str(key): json_ready(value) for key, value in point.items()}
    else:
        msg = "Sweep point specs must be mappings or PalaceSweepPointSpec instances"
        raise TypeError(msg)

    point_slug = optional_str(row.get("point_slug"))
    if not point_slug:
        msg = "Sweep point specs must include a non-empty 'point_slug'"
        raise ValueError(msg)
    row["point_slug"] = point_slug

    parameters = row.get("parameters", {})
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        msg = f"Sweep point {point_slug!r} field 'parameters' must be a mapping"
        raise TypeError(msg)
    row["parameters"] = {
        str(key): json_ready(value) for key, value in parameters.items()
    }

    for field_name in SWEEP_POINT_PATH_FIELDS:
        if row.get(field_name) is not None:
            row[field_name] = str(row[field_name])

    return row


def _raise_for_duplicate_sweep_point_slugs(point_rows: list[dict[str, Any]]) -> None:
    """Reject point rows with duplicate stable slugs."""
    duplicates = duplicate_values(str(row["point_slug"]) for row in point_rows)
    if duplicates:
        duplicate_text = ", ".join(repr(value) for value in duplicates)
        msg = f"Sweep point slugs must be unique; duplicates: {duplicate_text}"
        raise ValueError(msg)


__all__ = [
    "write_palace_sweep_points",
]
