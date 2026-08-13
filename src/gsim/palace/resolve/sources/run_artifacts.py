"""Locate Palace run artifact files for the resolve pipeline.

This module owns filesystem discovery for core handoff artifacts and solver
result files. It does not parse file contents or build report objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from gsim.palace.run_folder import palace_run_folder

CORE_RUN_ARTIFACT_NAMES = (
    "palace.msh",
    "config.json",
    "mesh_manifest.json",
    "palace_index_map.json",
    "palace_material_resolution.json",
)
NON_RESULT_ARTIFACT_NAMES = (
    *CORE_RUN_ARTIFACT_NAMES,
    "palace_handoff_metadata.json",
    "palace_sweep_handoff_metadata.json",
    "palace_run_metadata.json",
    "palace_resource_record.json",
    "palace_handoff_archive_manifest.json",
    "palace_sweep_handoff_archive_manifest.json",
    "port_information.json",
    "run_palace.sbatch",
    "run_sweep_array.sbatch",
)


def find_postprocessing_index_map(source: str | Path | dict) -> Path | None:
    """Search canonical metadata/cloud locations for ``palace_index_map.json``."""
    return _find_metadata_sidecar(source, "palace_index_map.json")


def find_config_json(source: str | Path | dict) -> Path | None:
    """Search canonical run-root/cloud locations for Palace ``config.json``."""
    return _find_execution_input(source, "config.json")


def find_material_resolution_json(source: str | Path | dict) -> Path | None:
    """Search canonical metadata/cloud locations for material-resolution sidecars."""
    return _find_metadata_sidecar(source, "palace_material_resolution.json")


def find_mesh_manifest_json(source: str | Path | dict) -> Path | None:
    """Search canonical metadata/cloud locations for ``mesh_manifest.json``."""
    return _find_metadata_sidecar(source, "mesh_manifest.json")


def find_mesh_file(source: str | Path | dict) -> Path | None:
    """Search a configured directory mesh or legacy/cloud ``palace.msh``."""
    if isinstance(source, dict):
        config_path = source.get("config.json")
        if config_path is not None:
            configured_mesh = _explicit_configured_mesh(Path(config_path).parent)
            if configured_mesh is not None:
                return configured_mesh
        return _find_execution_input(source, "palace.msh")
    path = Path(source)
    root = path if path.is_dir() else path.parent
    for candidate_root in (root, root.parent, root.parent.parent):
        configured_mesh = _explicit_configured_mesh(candidate_root)
        if configured_mesh is not None:
            return configured_mesh
    return _find_execution_input(source, "palace.msh")


def _explicit_configured_mesh(root: Path) -> Path | None:
    """Return a Model.Mesh path only when the config explicitly owns it."""
    folder = palace_run_folder(root)
    if not folder.config_path.is_file():
        return None
    mesh_path = folder.mesh_path
    try:
        config = json.loads(folder.config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Palace config must be valid JSON: {folder.config_path}"
        ) from exc
    model = config.get("Model")
    return mesh_path if isinstance(model, Mapping) and "Mesh" in model else None


def find_runtime_metadata_json(source: str | Path | dict) -> Path | None:
    """Search canonical metadata/cloud locations for local runtime metadata."""
    return _find_metadata_sidecar(source, "palace_run_metadata.json")


def find_resource_record_json(source: str | Path | dict) -> Path | None:
    """Search canonical metadata/cloud locations for post-run resource records."""
    return _find_metadata_sidecar(source, "palace_resource_record.json")


def find_handoff_metadata_json(source: str | Path | dict) -> Path | None:
    """Search canonical metadata/cloud locations for handoff metadata."""
    return _find_metadata_sidecar(source, "palace_handoff_metadata.json")


def find_sweep_handoff_metadata_json(source: str | Path | dict) -> Path | None:
    """Search common sweep locations for handoff metadata."""
    return _find_metadata_sidecar(source, "palace_sweep_handoff_metadata.json")


def _candidate_roots(source: str | Path | dict, name: str) -> list[Path]:
    """Return plausible artifact roots for one source and filename."""
    if isinstance(source, dict):
        explicit = source.get(name)
        if explicit is not None:
            return [Path(explicit)]
        return [Path(value).parent for value in source.values()]

    path = Path(source)
    return [path if path.is_dir() else path.parent]


def _first_existing(candidates: list[Path]) -> Path | None:
    """Return the first existing path from ordered candidates."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_execution_input(source: str | Path | dict, name: str) -> Path | None:
    """Find an execution input across canonical run-folder locations."""
    if isinstance(source, dict) and source.get(name) is not None:
        return Path(source[name])

    for root in _candidate_roots(source, name):
        found = _first_existing(
            [
                root / name,
                root.parent / name,
                root.parent.parent / name,
                root / "input" / name,
                root.parent / "input" / name,
                root.parent.parent / "input" / name,
            ]
        )
        if found is not None:
            return found
    return None


def _find_metadata_sidecar(source: str | Path | dict, name: str) -> Path | None:
    """Find a metadata sidecar across canonical run-folder locations."""
    if isinstance(source, dict) and source.get(name) is not None:
        return Path(source[name])

    for root in _candidate_roots(source, name):
        found = _first_existing(
            [
                root / "metadata" / name,
                root.parent / "metadata" / name,
                root.parent.parent / "metadata" / name,
                root / name,
                root.parent / name,
                root.parent.parent / name,
                root / "input" / name,
                root.parent / "input" / name,
                root.parent.parent / "input" / name,
            ]
        )
        if found is not None:
            return found
    return None


def palace_result_files(source: str | Path | dict) -> dict[str, Path]:
    """Return solver result files while excluding handoff/run sidecars."""
    if isinstance(source, dict):
        excluded_names = {*NON_RESULT_ARTIFACT_NAMES}
        config_path = source.get("config.json")
        if config_path is not None:
            mesh_path = palace_run_folder(Path(config_path).parent).mesh_path
            excluded_names.update(
                {
                    mesh_path.name,
                    mesh_path.relative_to(Path(config_path).parent).as_posix(),
                }
            )
        return {
            str(name): Path(value)
            for name, value in sorted(source.items())
            if str(name) not in excluded_names and Path(value).is_file()
        }

    path = Path(source)
    root = path if path.is_dir() else path.parent
    mesh_path = find_mesh_file(source)
    excluded_names = {*NON_RESULT_ARTIFACT_NAMES}
    if mesh_path is not None:
        excluded_names.add(mesh_path.name)
    candidate_dirs = []
    if root.name == "palace" and root.parent.name == "results":
        candidate_dirs.append(root)
    candidate_dirs.extend(
        [
            root / "results" / "palace",
            root / "palace",
            root,
        ]
    )
    for candidate in candidate_dirs:
        if candidate.exists() and candidate.is_dir():
            files = {
                child.name: child
                for child in sorted(candidate.iterdir())
                if child.is_file()
                and not child.name.startswith(".")
                and child.name not in excluded_names
            }
            if files:
                return files
    return {}
