"""Resolve-layer metadata models for point-local Palace sweeps.

Sweep models describe explicit point identities, artifact locations, and flat
records written by resolve. They are not problem reports and do not own
visualization; any semantic physics or performance display is represented by
typed data in ``gsim.palace.results``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gsim.palace._shared import (
    as_mapping,
    duplicate_values,
    json_ready,
    record_column_key,
    record_value,
)
from gsim.palace.resolve.sources.run_models import (
    PalaceArtifactStatus,
    PalaceRunSummary,
)


@dataclass(frozen=True)
class PalaceSweepPointSpec:
    """JSON-friendly metadata for one Palace sweep point."""

    point_slug: str
    parameters: dict[str, Any] = field(default_factory=dict)
    run_dir: str | Path | None = None
    result_dir: str | Path | None = None
    config_path: str | Path | None = None
    mesh_path: str | Path | None = None
    mesh_manifest_path: str | Path | None = None
    index_map_path: str | Path | None = None
    material_resolution_path: str | Path | None = None
    handoff_metadata_path: str | Path | None = None
    runtime_metadata_path: str | Path | None = None
    resource_record_path: str | Path | None = None
    port_information_path: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a ``points.json`` row."""
        row: dict[str, Any] = {
            "point_slug": self.point_slug,
            "parameters": json_ready(self.parameters),
        }
        for field_name in SWEEP_POINT_PATH_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                row[field_name] = str(value)
        return row


@dataclass(frozen=True)
class PalaceSweepPointSummary:
    """Resolved metadata for one point in a Palace sweep."""

    point_slug: str
    parameters: dict[str, Any]
    source: dict[str, Path] | Path
    run_summary: PalaceRunSummary
    report_metrics: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return one flat, table-friendly point metadata row."""
        summary = self.run_summary
        runtime = summary.runtime
        handoff = summary.handoff
        resource = summary.resource
        allocation = as_mapping(resource.get("allocation"))
        resource_runtime = as_mapping(resource.get("runtime"))
        model_size = as_mapping(resource.get("model_size"))
        memory = as_mapping(resource.get("memory"))
        scheduler = as_mapping(resource.get("scheduler"))
        handoff_profile = as_mapping(handoff.get("profile"))
        record = {
            "point_slug": self.point_slug,
            "problem_type": summary.problem_type,
            "complete": not summary.missing_artifacts,
            "missing_artifact_count": len(summary.missing_artifacts),
            "missing_artifacts": ",".join(summary.missing_artifacts),
            "core_artifact_count": count_present_artifacts(summary.artifacts),
            "core_artifact_bytes": sum_artifact_bytes(summary.artifacts),
            "result_count": count_present_artifacts(summary.results),
            "result_bytes": sum_artifact_bytes(summary.results),
            "runtime_present": runtime.get("present") is True,
            "runtime_status": runtime.get("status"),
            "runtime_return_code": runtime.get("return_code"),
            "runtime_elapsed_seconds": runtime.get("elapsed_seconds"),
            "runtime_output_count": runtime.get("output_count"),
            "runtime_output_bytes": runtime.get("output_bytes"),
            "handoff_present": handoff.get("present") is True,
            "handoff_status": handoff.get("status"),
            "handoff_profile_name": handoff_profile.get("name"),
            "handoff_script_present": handoff.get("script_present"),
            "handoff_archive_present": handoff.get("archive_present"),
            "handoff_archive_manifest_present": handoff.get("archive_manifest_present"),
            "resource_present": resource.get("present") is True,
            "resource_status": resource.get("status"),
            "resource_wall_time_seconds": resource_runtime.get("wall_time_seconds"),
            "resource_core_hours": resource_runtime.get("core_hours"),
            "resource_nodes": allocation.get("nodes"),
            "resource_num_processes": allocation.get("num_processes"),
            "resource_num_threads": allocation.get("num_threads"),
            "resource_peak_total_hwm_gib": memory.get("peak_total_hwm_gib"),
            "resource_global_unknowns": model_size.get("global_unknowns"),
            "resource_scheduler_kind": scheduler.get("kind"),
            "resource_scheduler_job_id": scheduler.get("job_id"),
            "resource_scheduler_job_state": scheduler.get("job_state"),
            "resource_scheduler_partition": scheduler.get("partition"),
            "config_material_count": summary.config.get("material_count"),
            "mesh_manifest_entry_count": summary.mesh_manifest.get("entry_count"),
            "index_map_entry_count": summary.index_map.get("entry_count"),
            "material_resolution_material_count": summary.material_resolution.get(
                "material_count"
            ),
            "material_resolution_interface_count": summary.material_resolution.get(
                "interface_count"
            ),
        }
        for key, value in sorted(self.report_metrics.items()):
            record[f"report_{record_column_key(key)}"] = record_value(value)
        for key, value in sorted(self.parameters.items()):
            record[f"parameter_{record_column_key(key)}"] = record_value(value)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly sweep point summary."""
        source: dict[str, str] | str
        if isinstance(self.source, dict):
            source = {name: str(path) for name, path in self.source.items()}
        else:
            source = str(self.source)
        return {
            "point_slug": self.point_slug,
            "parameters": dict(self.parameters),
            "source": source,
            "run_summary": self.run_summary.to_dict(),
            "report_metrics": dict(self.report_metrics),
            "record": self.to_record(),
            "missing_artifacts": list(self.run_summary.missing_artifacts),
        }


@dataclass(frozen=True)
class PalaceSweepSummary:
    """Resolved metadata for point-local Palace runs in one sweep folder."""

    sweep_id: str | None
    source_path: Path
    points: tuple[PalaceSweepPointSummary, ...]
    handoff: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parse_warnings: tuple[str, ...] = ()

    @property
    def point_count(self) -> int:
        """Number of point rows loaded from ``points.json``."""
        return len(self.points)

    @property
    def point_slugs(self) -> tuple[str, ...]:
        """Point slugs in sweep order."""
        return tuple(point.point_slug for point in self.points)

    @property
    def duplicate_point_slugs(self) -> tuple[str, ...]:
        """Point slugs that appear more than once."""
        return duplicate_values(self.point_slugs)

    @property
    def complete_point_count(self) -> int:
        """Points with all core handoff artifacts present."""
        return sum(not point.run_summary.missing_artifacts for point in self.points)

    @property
    def runtime_present_count(self) -> int:
        """Points with a runtime metadata sidecar."""
        return sum(
            point.run_summary.runtime.get("present") is True for point in self.points
        )

    @property
    def resource_present_count(self) -> int:
        """Points with a post-run resource record sidecar."""
        return sum(
            point.run_summary.resource.get("present") is True for point in self.points
        )

    @property
    def problem_types(self) -> tuple[str, ...]:
        """Sorted Palace problem types observed across point summaries."""
        return tuple(
            sorted(
                {
                    str(point.run_summary.problem_type)
                    for point in self.points
                    if point.run_summary.problem_type is not None
                }
            )
        )

    @property
    def total_runtime_elapsed_seconds(self) -> float | None:
        """Sum known point runtime durations, or ``None`` when none are present."""
        elapsed: list[float] = []
        for point in self.points:
            value = point.run_summary.runtime.get("elapsed_seconds")
            if value is not None:
                elapsed.append(float(value))
        if not elapsed:
            return None
        return sum(elapsed)

    def to_point_records(self) -> list[dict[str, Any]]:
        """Return flat, table-friendly point records for this sweep."""
        return [
            {"sweep_id": self.sweep_id, **point.to_record()} for point in self.points
        ]

    def to_dataframe(self):
        """Return sweep point records as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame.from_records(self.to_point_records())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly sweep summary."""
        return {
            "sweep_id": self.sweep_id,
            "source_path": str(self.source_path),
            "metadata": dict(self.metadata),
            "handoff": dict(self.handoff),
            "point_count": self.point_count,
            "point_slugs": list(self.point_slugs),
            "duplicate_point_slugs": list(self.duplicate_point_slugs),
            "complete_point_count": self.complete_point_count,
            "runtime_present_count": self.runtime_present_count,
            "resource_present_count": self.resource_present_count,
            "problem_types": list(self.problem_types),
            "total_runtime_elapsed_seconds": self.total_runtime_elapsed_seconds,
            "parse_warnings": list(self.parse_warnings),
            "point_records": self.to_point_records(),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class PalaceSweepResourceIndexResult:
    """Written sweep-level resource and benchmark index artifacts."""

    summary_path: Path
    point_records_csv_path: Path
    resource_records_csv_path: Path
    benchmark_jsonl_path: Path
    point_count: int
    resource_present_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result summary."""
        return {
            "summary_path": str(self.summary_path),
            "point_records_csv_path": str(self.point_records_csv_path),
            "resource_records_csv_path": str(self.resource_records_csv_path),
            "benchmark_jsonl_path": str(self.benchmark_jsonl_path),
            "point_count": self.point_count,
            "resource_present_count": self.resource_present_count,
        }


SWEEP_POINT_PATH_FIELDS = (
    "run_dir",
    "result_dir",
    "config_path",
    "mesh_path",
    "mesh_manifest_path",
    "index_map_path",
    "material_resolution_path",
    "handoff_metadata_path",
    "runtime_metadata_path",
    "resource_record_path",
    "port_information_path",
)


def count_present_artifacts(artifacts: dict[str, PalaceArtifactStatus]) -> int:
    """Return the number of present artifacts in an artifact-status mapping."""
    return sum(artifact.present for artifact in artifacts.values())


def sum_artifact_bytes(artifacts: dict[str, PalaceArtifactStatus]) -> int:
    """Return total byte size for present artifacts."""
    return sum(artifact.bytes for artifact in artifacts.values() if artifact.present)
