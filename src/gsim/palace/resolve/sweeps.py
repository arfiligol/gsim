"""Public Resolve orchestration for Palace sweep summaries.

This module crosses two Resolve responsibilities on purpose: it loads
point-local source summaries from ``resolve.sources`` and, when requested,
attaches compact report metrics from ``resolve.assembly``. Keeping that
coordination here prevents ``resolve.sources`` from depending on report
construction while preserving one public sweep-summary API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gsim.palace._shared import as_mapping, optional_str, relative_path_or_name
from gsim.palace.resolve.assembly.sweep_metrics import load_sweep_point_report_metrics
from gsim.palace.resolve.sources.run_artifacts import find_sweep_handoff_metadata_json
from gsim.palace.resolve.sources.run_sidecars import summarize_handoff_metadata_json
from gsim.palace.resolve.sources.run_summary import load_palace_run_summary
from gsim.palace.resolve.sources.sweep_models import (
    PalaceSweepPointSummary,
    PalaceSweepResourceIndexResult,
    PalaceSweepSummary,
)
from gsim.palace.resolve.sources.sweep_records import (
    write_records_csv,
    write_records_jsonl,
)
from gsim.palace.resolve.sources.sweep_sources import (
    palace_sweep_point_source,
    sweep_point_slug,
)


def write_palace_sweep_resource_index(
    source: str | Path,
    *,
    include_hashes: bool = False,
    include_report_metrics: bool = False,
    records_dir: str | Path = "metadata/records",
    summary_filename: str = "sweep_resource_index.json",
    point_records_filename: str = "sweep_point_records.csv",
    resource_records_filename: str = "sweep_resource_records.csv",
    benchmark_jsonl_filename: str = "sweep_benchmark_index.jsonl",
) -> PalaceSweepResourceIndexResult:
    """Write sweep-level point/resource records and a benchmark JSONL index.

    This writer loads existing point-local run summaries from explicit
    ``points.json`` metadata, then writes table-friendly records under
    ``metadata/records`` by default. Optional report metrics are attached only
    through Resolve assembly. It does not run Palace, submit jobs, parse
    private profile catalogs, or infer sweep identity from folders.
    """
    source_path = Path(source)
    points_path = source_path if source_path.is_file() else source_path / "points.json"
    sweep_root = points_path.parent
    summary = load_palace_sweep_summary(
        points_path,
        include_hashes=include_hashes,
        include_report_metrics=include_report_metrics,
    )
    records_root = sweep_root / records_dir
    records_root.mkdir(parents=True, exist_ok=True)

    point_records = summary.to_point_records()
    resource_records = [
        record for record in point_records if record.get("resource_present") is True
    ]
    point_records_csv_path = records_root / point_records_filename
    resource_records_csv_path = records_root / resource_records_filename
    benchmark_jsonl_path = records_root / benchmark_jsonl_filename
    summary_path = records_root / summary_filename

    write_records_csv(point_records_csv_path, point_records)
    write_records_csv(resource_records_csv_path, resource_records)
    write_records_jsonl(benchmark_jsonl_path, point_records)

    payload = {
        "schema_version": 1,
        "sweep_id": summary.sweep_id,
        "source_path": relative_path_or_name(points_path, sweep_root),
        "point_count": summary.point_count,
        "resource_present_count": summary.resource_present_count,
        "runtime_present_count": summary.runtime_present_count,
        "complete_point_count": summary.complete_point_count,
        "problem_types": list(summary.problem_types),
        "parse_warnings": list(summary.parse_warnings),
        "metadata": dict(summary.metadata),
        "records": {
            "point_records_csv": relative_path_or_name(
                point_records_csv_path,
                sweep_root,
            ),
            "resource_records_csv": relative_path_or_name(
                resource_records_csv_path,
                sweep_root,
            ),
            "benchmark_jsonl": relative_path_or_name(
                benchmark_jsonl_path,
                sweep_root,
            ),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return PalaceSweepResourceIndexResult(
        summary_path=summary_path,
        point_records_csv_path=point_records_csv_path,
        resource_records_csv_path=resource_records_csv_path,
        benchmark_jsonl_path=benchmark_jsonl_path,
        point_count=summary.point_count,
        resource_present_count=summary.resource_present_count,
    )


def load_palace_sweep_summary(
    source: str | Path,
    *,
    include_hashes: bool = False,
    include_report_metrics: bool = False,
) -> PalaceSweepSummary:
    """Load a compact summary for a point-local Palace sweep.

    The sweep root must provide a ``points.json`` file. Each point row may
    contain ``point_slug``, ``parameters``, ``run_dir``, ``result_dir``,
    ``config_path``, and ``mesh_path`` fields. Point-local artifact loading
    reuses :func:`load_palace_run_summary`; optional report metrics are loaded
    by ``resolve.assembly``. This helper does not infer point identity from
    folder names alone and does not run Palace.

    Args:
        source: Sweep root directory or direct ``points.json`` path.
        include_hashes: Include SHA-256 checksums for present point files.
        include_report_metrics: Include compact physics/report metrics when the
            point has the result files required by its Palace problem type.

    Returns:
        :class:`PalaceSweepSummary` with one run summary per sweep point.
    """
    source_path = Path(source)
    points_path = source_path if source_path.is_file() else source_path / "points.json"
    if not points_path.exists():
        raise FileNotFoundError(points_path)

    sweep_root = points_path.parent
    payload = json.loads(points_path.read_text())
    if isinstance(payload, list):
        point_specs = payload
        metadata: dict[str, Any] = {}
        sweep_id = sweep_root.name
    else:
        metadata = {
            key: value for key, value in as_mapping(payload).items() if key != "points"
        }
        point_specs = payload.get("points", []) if isinstance(payload, dict) else []
        sweep_id = optional_str(metadata.get("sweep_id")) or sweep_root.name

    if not isinstance(point_specs, list):
        msg = "points.json field 'points' must be a list"
        raise TypeError(msg)

    points: list[PalaceSweepPointSummary] = []
    parse_warnings: list[str] = []
    seen_point_slugs: set[str] = set()
    for index, raw_point in enumerate(point_specs):
        if not isinstance(raw_point, dict):
            parse_warnings.append(f"Skipping non-object sweep point at index {index}")
            continue
        point_slug = sweep_point_slug(raw_point, index)
        if point_slug in seen_point_slugs:
            parse_warnings.append(
                f"Duplicate sweep point_slug {point_slug!r} at index {index}"
            )
        seen_point_slugs.add(point_slug)
        parameters = as_mapping(raw_point.get("parameters"))
        point_source = palace_sweep_point_source(
            sweep_root,
            raw_point,
            point_slug=point_slug,
        )
        run_summary = load_palace_run_summary(
            point_source,
            include_hashes=include_hashes,
        )
        report_problem_type = run_summary.problem_type or optional_str(
            parameters.get("problem_type")
        )
        report_metrics = (
            load_sweep_point_report_metrics(point_source, report_problem_type)
            if include_report_metrics
            else {}
        )
        points.append(
            PalaceSweepPointSummary(
                point_slug=point_slug,
                parameters=dict(parameters),
                source=point_source,
                run_summary=run_summary,
                report_metrics=report_metrics,
            )
        )

    return PalaceSweepSummary(
        sweep_id=sweep_id,
        source_path=points_path,
        points=tuple(points),
        handoff=summarize_handoff_metadata_json(
            find_sweep_handoff_metadata_json(points_path)
        ),
        metadata=metadata,
        parse_warnings=tuple(parse_warnings),
    )


__all__ = [
    "load_palace_sweep_summary",
    "write_palace_sweep_resource_index",
]
