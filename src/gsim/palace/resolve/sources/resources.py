"""Write Palace resource-record sidecars.

This module composes parser output from Palace logs and optional Slurm sidecars
into the JSON/CSV resource-record artifacts consumed by run summaries and
benchmarks. Palace log syntax and Slurm syntax live in dedicated parser modules.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gsim.palace._shared import as_mapping, json_ready, relative_path_or_name
from gsim.palace.resolve.sources.resource_log import (
    PALACE_AMR_PASS_COLUMNS,
    PALACE_STAGE_MEMORY_COLUMNS,
    PALACE_STAGE_TIMING_COLUMNS,
    parse_palace_resource_log,
)
from gsim.palace.resolve.sources.slurm import parse_slurm_scontrol_job


def write_palace_resource_record(
    source: str | Path,
    *,
    status: str = "completed",
    sources: Mapping[str, Any] | None = None,
    launcher: Mapping[str, Any] | None = None,
    scheduler: Mapping[str, Any] | None = None,
    solver: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    model_size: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    tables: Mapping[str, Any] | None = None,
    missing_sources: Iterable[str] = (),
    parse_warnings: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    filename: str = "metadata/palace_resource_record.json",
) -> Path:
    """Write a Palace post-run resource record sidecar.

    The record is measured or caller-supplied post-run evidence. It is separate
    from ``palace_run_metadata.json`` execution metadata and from dry-run
    handoff intent.
    """
    record_path = _resource_record_path(source, filename=filename)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": str(status),
        "sources": json_ready(dict(sources or {})),
        "launcher": json_ready(dict(launcher or {})),
        "scheduler": json_ready(dict(scheduler or {})),
        "solver": json_ready(dict(solver or {})),
        "allocation": json_ready(dict(allocation or {})),
        "runtime": json_ready(dict(runtime or {})),
        "model_size": json_ready(dict(model_size or {})),
        "memory": json_ready(dict(memory or {})),
        "tables": json_ready(dict(tables or {})),
        "missing_sources": [str(value) for value in missing_sources],
        "parse_warnings": [str(value) for value in parse_warnings],
        "metadata": json_ready(dict(metadata or {})),
    }

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return record_path


def write_palace_resource_record_from_log(
    source: str | Path,
    log_path: str | Path,
    *,
    scontrol_path: str | Path | None = None,
    status: str = "completed",
    launcher: Mapping[str, Any] | None = None,
    scheduler: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    model_size: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    missing_sources: Iterable[str] = (),
    parse_warnings: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    filename: str = "metadata/palace_resource_record.json",
) -> Path:
    """Parse a Palace log and write a resource record plus CSV table sidecars.

    Caller-supplied ``scheduler``, ``allocation``, ``runtime``, ``model_size``,
    and ``memory`` values override parser-derived values so local runners can
    add scheduler or launcher facts without changing the log parser.
    """
    run_root = Path(source)
    record_path = _resource_record_path(run_root, filename=filename)
    records_dir = record_path.parent
    records_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_palace_resource_log(log_path)
    parsed_scontrol = (
        parse_slurm_scontrol_job(scontrol_path) if scontrol_path is not None else None
    )
    table_specs = {
        "amr_passes": (
            records_dir / "palace_amr_passes.csv",
            parsed["amr_passes"],
            PALACE_AMR_PASS_COLUMNS,
        ),
        "stage_timing": (
            records_dir / "palace_stage_timing.csv",
            parsed["stage_timing"],
            PALACE_STAGE_TIMING_COLUMNS,
        ),
        "stage_memory": (
            records_dir / "palace_stage_memory.csv",
            parsed["stage_memory"],
            PALACE_STAGE_MEMORY_COLUMNS,
        ),
    }
    table_refs: dict[str, dict[str, Any]] = {}
    for table_name, (path, rows, columns) in table_specs.items():
        _write_resource_record_csv(path, rows, columns)
        table_refs[table_name] = {
            "path": relative_path_or_name(path, run_root),
            "row_count": len(rows),
        }

    parsed_source = as_mapping(parsed.get("source"))
    sources = {
        "palace_log": {
            "path": relative_path_or_name(Path(log_path), run_root),
            "sha256": parsed_source.get("sha256"),
            "bytes": parsed_source.get("bytes"),
        }
    }
    if scontrol_path is not None and parsed_scontrol is not None:
        scontrol_source = as_mapping(parsed_scontrol.get("source"))
        sources["slurm_scontrol"] = {
            "path": relative_path_or_name(Path(scontrol_path), run_root),
            "sha256": scontrol_source.get("sha256"),
            "bytes": scontrol_source.get("bytes"),
        }

    parsed_scheduler = (
        as_mapping(parsed_scontrol.get("scheduler"))
        if parsed_scontrol is not None
        else {}
    )
    if parsed_scontrol is not None and launcher is None:
        launcher = {"kind": "slurm"}
    merged_scheduler = {
        **parsed_scheduler,
        **dict(scheduler or {}),
    }
    merged_allocation = {
        **as_mapping(parsed.get("allocation")),
        **(
            as_mapping(parsed_scontrol.get("allocation"))
            if parsed_scontrol is not None
            else {}
        ),
        **dict(allocation or {}),
    }

    return write_palace_resource_record(
        record_path,
        status=status,
        sources=sources,
        launcher=launcher,
        scheduler=merged_scheduler,
        solver=parsed["solver"],
        allocation=merged_allocation,
        runtime={**as_mapping(parsed.get("runtime")), **dict(runtime or {})},
        model_size={**as_mapping(parsed.get("model_size")), **dict(model_size or {})},
        memory={**as_mapping(parsed.get("memory")), **dict(memory or {})},
        tables=table_refs,
        missing_sources=missing_sources,
        parse_warnings=parse_warnings,
        metadata=metadata,
    )


def _resource_record_path(source: str | Path, *, filename: str) -> Path:
    """Resolve the resource-record path from a source path or directory."""
    path = Path(source)
    return path if path.suffix.lower() == ".json" else path / filename


def _write_resource_record_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> None:
    """Write one tabular resource record sidecar as CSV."""
    import pandas as pd

    frame = pd.DataFrame.from_records(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


__all__ = [
    "parse_palace_resource_log",
    "parse_slurm_scontrol_job",
    "write_palace_resource_record",
    "write_palace_resource_record_from_log",
]
