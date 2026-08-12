"""Summarize Palace JSON sidecars for run-level reports.

The functions here parse compact facts from config, mesh manifest, index-map,
material-resolution, handoff, runtime, and resource-record sidecars. They do
not discover files and do not load problem-specific physics tables.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gsim.palace._shared import (
    as_mapping as _as_mapping,
)
from gsim.palace._shared import (
    optional_float as _optional_float,
)


def summarize_config_json(path: Path | None) -> dict[str, Any]:
    """Return compact problem/material/postprocessing facts from ``config.json``."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    domains = _as_mapping(data.get("Domains"))
    boundaries = _as_mapping(data.get("Boundaries"))
    domain_postprocessing = _as_mapping(domains.get("Postprocessing"))
    boundary_postprocessing = _as_mapping(boundaries.get("Postprocessing"))
    materials = domains.get("Materials", [])
    return {
        "present": True,
        "problem_type": _as_mapping(data.get("Problem")).get("Type"),
        "material_count": len(materials) if isinstance(materials, list) else 0,
        "material_names": sorted(
            str(row.get("Name"))
            for row in materials
            if isinstance(row, dict) and row.get("Name") is not None
        ),
        "domain_postprocessing_keys": sorted(domain_postprocessing.keys()),
        "boundary_postprocessing_keys": sorted(boundary_postprocessing.keys()),
        "lumped_port_count": _len_list(boundaries.get("LumpedPort")),
        "terminal_count": _len_list(boundaries.get("Terminal")),
        "pec_count": _len_list(boundaries.get("PEC")),
    }


def summarize_mesh_manifest_json(path: Path | None) -> dict[str, Any]:
    """Return compact role and physical-name facts from ``mesh_manifest.json``."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    entries = _list_of_mappings(data.get("entries"))
    interface_entries = [
        entry
        for entry in entries
        if entry.get("interface_of") or "___" in str(entry.get("name", ""))
    ]
    return {
        "present": True,
        "schema_version": data.get("schema_version"),
        "entry_count": len(entries),
        "roles": _count_mapping_values(entries, "role"),
        "dimensions": _count_mapping_values(entries, "dimension"),
        "physical_name_count": sum(
            _len_list(entry.get("physical_names")) for entry in entries
        ),
        "interface_entry_count": len(interface_entries),
    }


def summarize_index_map_json(path: Path | None) -> dict[str, Any]:
    """Return compact section/role/name facts from ``palace_index_map.json``."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    entries = _list_of_mappings(data.get("entries"))
    return {
        "present": True,
        "schema_version": data.get("schema_version"),
        "entry_count": len(entries),
        "sections": _count_mapping_values(entries, "section"),
        "roles": _count_mapping_values(entries, "role"),
        "terminal_names": sorted(
            {
                str(entry["terminal_name"])
                for entry in entries
                if entry.get("terminal_name") is not None
            }
        ),
        "port_names": sorted(
            {
                str(_as_mapping(entry.get("metadata")).get("port"))
                for entry in entries
                if _as_mapping(entry.get("metadata")).get("port") is not None
            }
        ),
    }


def summarize_material_resolution_json(path: Path | None) -> dict[str, Any]:
    """Return compact material-overlay provenance facts from resolution sidecars."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    materials = _list_of_mappings(data.get("materials"))
    interfaces = _list_of_mappings(data.get("interfaces"))
    return {
        "present": True,
        "schema_version": data.get("schema_version"),
        "material_count": len(materials),
        "interface_count": len(interfaces),
        "material_model_sources": sorted(
            {
                str(row["model_source"])
                for row in materials
                if row.get("model_source") is not None
            }
        ),
        "interface_model_sources": sorted(
            {
                str(row["model_source"])
                for row in interfaces
                if row.get("model_source") is not None
            }
        ),
        "material_validity": _count_mapping_values(materials, "within_validity"),
        "interface_validity": _count_mapping_values(interfaces, "within_validity"),
    }


def summarize_handoff_metadata_json(path: Path | None) -> dict[str, Any]:
    """Return handoff package status and referenced artifact presence."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    script = _as_mapping(data.get("script"))
    archive = _as_mapping(data.get("archive"))
    script_path = _referenced_sidecar_path(path, script.get("path"))
    archive_path = _referenced_sidecar_path(path, archive.get("path"))
    archive_manifest_path = _referenced_sidecar_path(path, archive.get("manifest_path"))
    return {
        "present": True,
        "schema_version": data.get("schema_version"),
        "status": data.get("status"),
        "launcher": _as_mapping(data.get("launcher")),
        "profile": _as_mapping(data.get("profile")),
        "resources": _as_mapping(data.get("resources")),
        "script": script,
        "script_present": script_path is not None and script_path.exists(),
        "archive": archive,
        "archive_present": archive_path is not None and archive_path.exists(),
        "archive_manifest_present": (
            archive_manifest_path is not None and archive_manifest_path.exists()
        ),
        "command": _as_mapping(data.get("command")),
        "metadata": _as_mapping(data.get("metadata")),
        "path": str(path),
    }


def _referenced_sidecar_path(sidecar_path: Path, value: Any) -> Path | None:
    """Resolve an optional sidecar reference relative to its metadata root."""
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    reference_root = (
        sidecar_path.parent.parent
        if sidecar_path.parent.name == "metadata"
        else sidecar_path.parent
    )
    return reference_root / path


def summarize_runtime_metadata_json(path: Path | None) -> dict[str, Any]:
    """Return local runner status, elapsed time, and output counts."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    outputs = _as_mapping(data.get("outputs"))
    output_bytes = 0
    for output in outputs.values():
        if isinstance(output, dict):
            output_bytes += int(output.get("bytes", 0) or 0)
    return {
        "present": True,
        "schema_version": data.get("schema_version"),
        "status": data.get("status"),
        "return_code": data.get("return_code"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "launcher": _as_mapping(data.get("launcher")),
        "resources": _as_mapping(data.get("resources")),
        "command": _as_mapping(data.get("command")),
        "output_count": len(outputs),
        "output_bytes": output_bytes,
        "path": str(path),
    }


def summarize_resource_record_json(path: Path | None) -> dict[str, Any]:
    """Return post-run resource and scheduler facts from a resource record."""
    if path is None or not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    sources = _as_mapping(data.get("sources"))
    tables = _as_mapping(data.get("tables"))
    runtime = _as_mapping(data.get("runtime"))
    allocation = _as_mapping(data.get("allocation"))
    scheduler = _as_mapping(data.get("scheduler"))
    model_size = _as_mapping(data.get("model_size"))
    memory = _as_mapping(data.get("memory"))
    _normalize_resource_runtime(runtime, allocation)
    _normalize_resource_memory(memory)
    missing_sources = [str(value) for value in data.get("missing_sources") or []]
    parse_warnings = [str(value) for value in data.get("parse_warnings") or []]
    return {
        "present": True,
        "schema_version": data.get("schema_version"),
        "created_at_utc": data.get("created_at_utc"),
        "status": data.get("status"),
        "launcher": _as_mapping(data.get("launcher")),
        "scheduler": scheduler,
        "solver": _as_mapping(data.get("solver")),
        "allocation": allocation,
        "runtime": runtime,
        "model_size": model_size,
        "memory": memory,
        "sources": sources,
        "source_count": len(sources),
        "tables": tables,
        "table_count": len(tables),
        "missing_sources": missing_sources,
        "missing_source_count": len(missing_sources),
        "parse_warnings": parse_warnings,
        "parse_warning_count": len(parse_warnings),
        "metadata": _as_mapping(data.get("metadata")),
        "path": str(path),
    }


def _normalize_resource_runtime(
    runtime: dict[str, Any],
    allocation: Mapping[str, Any],
) -> None:
    """Fill normalized runtime fields from related allocation evidence."""
    wall_time_seconds = _first_optional_float(
        runtime,
        ("wall_time_seconds", "elapsed_seconds", "total_elapsed_seconds"),
    )
    if wall_time_seconds is not None and runtime.get("wall_time_seconds") is None:
        runtime["wall_time_seconds"] = wall_time_seconds
    if runtime.get("core_hours") is None and wall_time_seconds is not None:
        cores = _resource_core_count(allocation)
        if cores is not None:
            runtime["core_hours"] = wall_time_seconds * cores / 3600.0


def _normalize_resource_memory(memory: dict[str, Any]) -> None:
    """Fill normalized memory fields from byte-valued evidence."""
    if memory.get("peak_total_hwm_gib") is None:
        peak_bytes = _first_optional_float(
            memory,
            ("peak_total_hwm_bytes", "peak_total_memory_bytes"),
        )
        if peak_bytes is not None:
            memory["peak_total_hwm_gib"] = peak_bytes / (1024.0**3)


def _resource_core_count(allocation: Mapping[str, Any]) -> float | None:
    """Infer an allocated core count from supported allocation fields."""
    for key in ("cores", "num_cpus"):
        value = _optional_float(allocation.get(key))
        if value is not None:
            return value
    num_processes = _optional_float(allocation.get("num_processes"))
    num_threads = _optional_float(allocation.get("num_threads"))
    if num_processes is not None and num_threads is not None:
        return num_processes * num_threads
    return None


def _first_optional_float(
    mapping: Mapping[str, Any],
    keys: Iterable[str],
) -> float | None:
    """Return the first mapping value convertible to float."""
    for key in keys:
        value = _optional_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    """Return list members that are mapping records."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _len_list(value: Any) -> int:
    """Return a list length, treating non-lists as empty."""
    return len(value) if isinstance(value, list) else 0


def _count_mapping_values(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    """Count stringified values for one key across mapping rows."""
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value_key = str(value)
        counts[value_key] = counts.get(value_key, 0) + 1
    return dict(sorted(counts.items()))
