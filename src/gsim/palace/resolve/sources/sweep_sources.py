"""Locate point-local Palace artifacts for sweep summaries.

The sweep source layer translates a public ``points.json`` row into explicit
run artifact paths. It does not parse artifacts, compute metrics, or write
benchmark records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from gsim.palace.resolve.sources.run_artifacts import NON_RESULT_ARTIFACT_NAMES


def sweep_point_slug(point_spec: dict[str, Any], index: int) -> str:
    """Return the stable slug for one sweep point specification."""
    value = (
        point_spec.get("point_slug")
        or point_spec.get("run_id")
        or point_spec.get("name")
        or f"point_{index}"
    )
    return str(value)


def palace_sweep_point_source(
    sweep_root: Path,
    point_spec: dict[str, Any],
    *,
    point_slug: str,
) -> dict[str, Path] | Path:
    """Resolve point-local Palace artifacts from one public sweep point row."""
    run_dir = _resolve_sweep_path(
        sweep_root,
        _first_mapping_value(point_spec, ("run_dir", "point_dir", "output_dir")),
        default=Path("points") / point_slug,
    )
    result_dir = _resolve_sweep_path(
        sweep_root,
        _first_mapping_value(
            point_spec,
            ("result_dir", "results_dir", "palace_result_dir"),
        ),
        default=Path("results") / point_slug / "palace",
    )
    run_dir = cast(Path, run_dir)
    result_dir = cast(Path, result_dir)
    source: dict[str, Path] = {}

    _add_sweep_source_file(
        source,
        "config.json",
        _resolve_sweep_path(
            sweep_root,
            _first_mapping_value(point_spec, ("config_path", "config_file", "config")),
        ),
        run_dir / "config.json",
        result_dir / "config.json",
    )
    _add_sweep_source_file(
        source,
        "palace.msh",
        _resolve_sweep_path(
            sweep_root,
            _first_mapping_value(point_spec, ("mesh_path", "mesh_file", "mesh")),
        ),
        run_dir / "palace.msh",
        run_dir / "mesh.msh",
        result_dir / "palace.msh",
    )
    _add_sweep_source_file(
        source,
        "mesh_manifest.json",
        _resolve_sweep_path(
            sweep_root,
            _first_mapping_value(point_spec, ("mesh_manifest_path", "manifest_path")),
        ),
        run_dir / "metadata" / "mesh_manifest.json",
        result_dir / "mesh_manifest.json",
    )
    _add_sweep_source_file(
        source,
        "palace_index_map.json",
        _resolve_sweep_path(sweep_root, point_spec.get("index_map_path")),
        run_dir / "metadata" / "palace_index_map.json",
        result_dir / "palace_index_map.json",
    )
    _add_sweep_source_file(
        source,
        "palace_material_resolution.json",
        _resolve_sweep_path(sweep_root, point_spec.get("material_resolution_path")),
        run_dir / "metadata" / "palace_material_resolution.json",
        result_dir / "palace_material_resolution.json",
    )
    _add_sweep_source_file(
        source,
        "palace_handoff_metadata.json",
        _resolve_sweep_path(sweep_root, point_spec.get("handoff_metadata_path")),
        run_dir / "metadata" / "palace_handoff_metadata.json",
        result_dir / "palace_handoff_metadata.json",
    )
    _add_sweep_source_file(
        source,
        "palace_run_metadata.json",
        _resolve_sweep_path(sweep_root, point_spec.get("runtime_metadata_path")),
        run_dir / "metadata" / "palace_run_metadata.json",
        result_dir / "palace_run_metadata.json",
    )
    _add_sweep_source_file(
        source,
        "palace_resource_record.json",
        _resolve_sweep_path(sweep_root, point_spec.get("resource_record_path")),
        run_dir / "metadata" / "palace_resource_record.json",
        result_dir / "palace_resource_record.json",
    )
    _add_sweep_source_file(
        source,
        "port_information.json",
        _resolve_sweep_path(
            sweep_root,
            _first_mapping_value(
                point_spec,
                ("port_information_path", "port_info_path"),
            ),
        ),
        run_dir / "metadata" / "port_information.json",
        result_dir / "port_information.json",
    )

    _add_sweep_result_files(source, result_dir)
    _add_sweep_result_files(source, run_dir / "results" / "palace")

    return source or run_dir


def _resolve_sweep_path(
    sweep_root: Path,
    value: Any,
    *,
    default: Path | None = None,
) -> Path | None:
    """Resolve one optional sweep path relative to the sweep root."""
    if value is None:
        if default is None:
            return None
        path = default
    else:
        path = Path(str(value))
    return path if path.is_absolute() else sweep_root / path


def _first_mapping_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-null value among alternate mapping keys."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _add_sweep_source_file(
    source: dict[str, Path],
    name: str,
    *candidates: Path | None,
) -> None:
    """Record the first existing candidate for a named sweep artifact."""
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.exists() and candidate.is_file():
            source[name] = candidate
            return
    for candidate in candidates:
        if candidate is not None:
            source.setdefault(name, candidate)
            return


def _add_sweep_result_files(source: dict[str, Path], result_dir: Path | None) -> None:
    """Add visible result files from one optional Palace result directory."""
    if result_dir is None or not result_dir.exists() or not result_dir.is_dir():
        return
    for child in sorted(result_dir.iterdir()):
        if (
            child.is_file()
            and not child.name.startswith(".")
            and child.name not in NON_RESULT_ARTIFACT_NAMES
        ):
            source.setdefault(child.name, child)


__all__ = ["palace_sweep_point_source", "sweep_point_slug"]
