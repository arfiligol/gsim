"""Run-stage handoff assembly for Palace workflows.

This module provides the notebook-facing assembly step that turns a prepared
Palace run folder plus launcher/profile/resource metadata into a
``PalaceRunHandle`` and a handoff archive.

Mesh/config generation, Slurm profile schemas, sbatch rendering, tarball
implementation, Resolve/report construction, and display live outside this
assembly layer. Low-level script, metadata, and archive writers remain in
``gsim.palace.handoff``. After external solver execution, notebooks pass
``handle.run_folder`` into Resolve.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from gsim.palace.resolve.problem_types import canonical_problem_type
from gsim.palace.run_folder import (
    PalaceRunFolder,
    prepare_palace_run_folder,
    relative_to_run_folder,
)
from gsim.palace.run_stage import PalaceRunHandle


def _handoff_profile_payload(profile: Any) -> Mapping[str, Any] | None:
    """Return JSON-ready profile metadata from a mapping or resolved profile."""
    if profile is None:
        return None
    if isinstance(profile, Mapping):
        return profile
    payload = getattr(profile, "profile", None)
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("profile must be a mapping or resolved Palace Slurm profile")


def _handoff_profile_name(profile: Any) -> str | None:
    """Return the profile name used for a run-stage handle."""
    if profile is None:
        return None
    if isinstance(profile, Mapping):
        name = profile.get("name")
    else:
        name = getattr(profile, "name", None)
    return str(name) if name is not None else None


def _handoff_resources_payload(
    profile: Any,
    resources: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return explicit resources or derive them from a resolved profile."""
    if resources is not None:
        return resources
    profile_resources = getattr(profile, "resources", None)
    to_dict = getattr(profile_resources, "to_dict", None)
    if callable(to_dict):
        return {"requested": to_dict()}
    return None


def _handoff_kind(launcher: Mapping[str, Any] | None) -> Literal["handoff", "slurm"]:
    """Return the run-handle kind implied by handoff launcher metadata."""
    if launcher is not None and launcher.get("kind") == "slurm":
        return "slurm"
    return "handoff"


def _run_folder_path(root: Path, path: str | Path | None) -> Path | None:
    """Resolve a run-folder-relative path for a run-stage handle."""
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def generate_palace_handoff_package(
    run_folder: str | Path | PalaceRunFolder,
    *,
    simulation_type: str,
    include_hashes: bool = False,
    include_results: bool = False,
    status: str = "packaged",
    launcher: Mapping[str, Any] | None = None,
    script_path: str | Path | None = None,
    profile: Any | None = None,
    resources: Mapping[str, Any] | None = None,
    command: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    archive_path: str | Path | None = None,
) -> PalaceRunHandle:
    """Package a Palace run folder for external execution.

    This is the Run Stage assembly point used by ``PalaceSimBase``. It ensures
    the canonical directory skeleton exists, writes handoff metadata, creates
    the archive manifest and tarball through ``gsim.palace.handoff``, and
    returns a lightweight handle for later Resolve-stage loading.

    Args:
        run_folder: Existing Palace run folder or typed run-folder view.
        simulation_type: Problem type used to annotate the returned handle.
        include_hashes: Include SHA-256 checksums in the archive manifest.
        include_results: Include already-present solver outputs in the archive.
        status: Handoff metadata status recorded in the sidecar.
        launcher: Optional launcher metadata, for example Slurm submission
            intent. This function records it but does not submit jobs.
        script_path: Optional run-folder-relative launcher script path to
            include in metadata and the archive manifest.
        profile: Optional mapping or resolved Slurm profile metadata.
        resources: Optional resource metadata. If omitted and ``profile`` has
            resource details, those are recorded as requested resources.
        command: Optional redacted command metadata.
        metadata: Additional JSON-friendly package metadata.
        archive_path: Optional target tarball path. Relative paths are resolved
            beside the run folder by the lower-level handoff helper.

    Returns:
        Run-stage handle pointing at the packaged folder, metadata sidecar,
        archive manifest, optional script, and archive path.

    Raises:
        TypeError: If ``profile`` is neither a mapping nor a resolved profile.
        ValueError: If archive paths or generated manifest entries violate the
            lower-level handoff package contract.
    """
    folder = (
        prepare_palace_run_folder(run_folder.root)
        if isinstance(run_folder, PalaceRunFolder)
        else prepare_palace_run_folder(run_folder)
    )
    package_metadata = {
        "generated_by": "gsim.palace.PalaceSimBase.generate_handoff_package",
        "simulation_type": simulation_type,
        **dict(metadata or {}),
    }
    profile_payload = _handoff_profile_payload(profile)
    resources_payload = _handoff_resources_payload(profile, resources)
    script_reference: str | Path | None = script_path
    if script_path is not None:
        script_reference = relative_to_run_folder(Path(script_path), folder.root)
    launcher_payload = dict(launcher) if launcher is not None else None
    if (
        launcher_payload is None
        and script_reference is not None
        and profile is not None
    ):
        launcher_payload = {
            "kind": "slurm",
            "submission": "manual",
            "script_path": str(script_reference),
        }
    handoff_command = (
        dict(command)
        if command is not None
        else {"argv": ["sbatch", str(script_reference)], "redacted": True}
        if script_reference is not None and profile is not None
        else {"executable": "palace", "config_path": "config.json"}
    )

    from gsim.palace.handoff import (
        package_palace_run_handoff_archive,
        write_palace_handoff_metadata,
    )

    metadata_path = write_palace_handoff_metadata(
        folder.root,
        status=status,
        launcher=launcher_payload,
        script_path=script_reference,
        profile=profile_payload,
        resources=resources_payload,
        command=handoff_command,
        metadata=package_metadata,
    )
    archive_result = package_palace_run_handoff_archive(
        folder.root,
        archive_path=archive_path,
        metadata=package_metadata,
        include_results=include_results,
        include_hashes=include_hashes,
    )

    return PalaceRunHandle(
        run_folder=folder.root,
        kind=_handoff_kind(launcher_payload),
        status=status,
        problem_type=canonical_problem_type(simulation_type),
        profile_name=_handoff_profile_name(profile),
        script_path=_run_folder_path(folder.root, script_reference),
        archive_path=archive_result.archive_path,
        metadata_path=metadata_path,
        archive_manifest_path=archive_result.manifest_path,
        metadata=package_metadata,
    )


__all__ = [
    "generate_palace_handoff_package",
]
