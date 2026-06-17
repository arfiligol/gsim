"""Resolve Palace run directories into artifact and benchmark summaries.

This module composes the run-level summary returned by ``resolve``. It owns the
public ``load_palace_run_summary`` entry point and delegates artifact discovery
and JSON sidecar interpretation to smaller resolve modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from gsim.palace._shared import sha256_file
from gsim.palace.resolve.sources.run_artifacts import (
    find_config_json,
    find_handoff_metadata_json,
    find_material_resolution_json,
    find_mesh_file,
    find_mesh_manifest_json,
    find_postprocessing_index_map,
    find_resource_record_json,
    find_runtime_metadata_json,
    palace_result_files,
)
from gsim.palace.resolve.sources.run_models import (
    PalaceArtifactStatus,
    PalaceRunSummary,
)
from gsim.palace.resolve.sources.run_sidecars import (
    summarize_config_json,
    summarize_handoff_metadata_json,
    summarize_index_map_json,
    summarize_material_resolution_json,
    summarize_mesh_manifest_json,
    summarize_resource_record_json,
    summarize_runtime_metadata_json,
)


def load_palace_run_summary(
    source: str | Path | dict,
    *,
    include_hashes: bool = False,
) -> PalaceRunSummary:
    """Load a compact Palace run artifact summary.

    This helper inspects existing generated files only. It does not run Palace
    and does not fabricate runtime timings when no timing artifact is present.

    Args:
        source: Simulation directory, Palace output directory, or results dict.
        include_hashes: Include SHA-256 checksums for present files.

    Returns:
        :class:`PalaceRunSummary` with core handoff artifacts, result files,
        and compact summaries of Palace config, mesh manifest, index-map, and
        material-resolution sidecars.
    """
    config_path = find_config_json(source)
    manifest_path = find_mesh_manifest_json(source)
    index_map_path = find_postprocessing_index_map(source)
    material_resolution_path = find_material_resolution_json(source)
    handoff_metadata_path = find_handoff_metadata_json(source)
    runtime_metadata_path = find_runtime_metadata_json(source)
    resource_record_path = find_resource_record_json(source)
    mesh_path = find_mesh_file(source)

    artifact_paths = {
        "palace.msh": mesh_path,
        "config.json": config_path,
        "mesh_manifest.json": manifest_path,
        "palace_index_map.json": index_map_path,
        "palace_material_resolution.json": material_resolution_path,
    }
    artifacts = {
        name: _artifact_status(name, path, include_hashes=include_hashes)
        for name, path in artifact_paths.items()
    }
    results = {
        name: _artifact_status(name, path, include_hashes=include_hashes)
        for name, path in palace_result_files(source).items()
    }
    config = summarize_config_json(config_path)
    return PalaceRunSummary(
        problem_type=cast("str | None", config.get("problem_type")),
        artifacts=artifacts,
        results=results,
        config=config,
        mesh_manifest=summarize_mesh_manifest_json(manifest_path),
        index_map=summarize_index_map_json(index_map_path),
        material_resolution=summarize_material_resolution_json(
            material_resolution_path
        ),
        handoff=summarize_handoff_metadata_json(handoff_metadata_path),
        runtime=summarize_runtime_metadata_json(runtime_metadata_path),
        resource=summarize_resource_record_json(resource_record_path),
    )


def _artifact_status(
    name: str,
    path: Path | None,
    *,
    include_hashes: bool,
) -> PalaceArtifactStatus:
    if path is None or not path.exists() or not path.is_file():
        return PalaceArtifactStatus(name=name, path=path, present=False)
    return PalaceArtifactStatus(
        name=name,
        path=path,
        present=True,
        bytes=int(path.stat().st_size),
        sha256=sha256_file(path) if include_hashes else None,
    )
