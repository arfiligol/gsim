"""Locate Palace run artifact files for the resolve pipeline.

This module owns filesystem discovery for core handoff artifacts and solver
result files. It does not parse file contents or build report objects.
"""

from __future__ import annotations

from pathlib import Path

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
    """Search canonical run-root/cloud locations for ``palace.msh``."""
    return _find_execution_input(source, "palace.msh")


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
        return {
            str(name): Path(value)
            for name, value in sorted(source.items())
            if str(name) not in NON_RESULT_ARTIFACT_NAMES and Path(value).is_file()
        }

    path = Path(source)
    root = path if path.is_dir() else path.parent
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
                and child.name not in NON_RESULT_ARTIFACT_NAMES
            }
            if files:
                return files
    return {}
