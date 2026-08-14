"""Palace handoff helpers for Run Stage packaging.

This module contains the Slurm profile resolution, sbatch rendering, handoff
metadata writing, archive manifest generation, and tarball packaging helpers
used by prepared Palace run folders.

Resolve-stage result auditing, typed result parsing, and report construction
remain in the Resolve/results pipeline. Handoff code records how a run folder
should travel to an external execution environment; it does not interpret solver
outputs.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from gsim.palace._shared import (
    as_mapping as _as_mapping,
)
from gsim.palace._shared import (
    json_ready as _json_ready,
)
from gsim.palace._shared import (
    optional_str as _optional_string,
)
from gsim.palace._shared import (
    sha256_file as _sha256_file,
)
from gsim.palace.run_folder import (
    prepare_palace_run_folder,
)

_SBATCH_TOKEN_FORBIDDEN = set(" \t\r\n\"'`$;&|<>")
_SBATCH_JOB_NAME_FORBIDDEN = set(" \t\r\n\"'`$;&|<>/#")
_WALL_TIME_PATTERN = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")

DEFAULT_PALACE_PETSC_OPTIONS = (
    "-ksp_monitor",
    "-ksp_converged_reason",
    "-eps_converged_reason",
    "-log_view",
)


def _default_palace_handoff_archive_path(root: str | Path) -> Path:
    """Return the default handoff archive path beside a Palace run folder."""
    run_root = Path(root)
    return run_root.parent / f"{run_root.name}-palace.tar.gz"


def write_palace_handoff_metadata(
    source: str | Path,
    *,
    status: str = "planned",
    launcher: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    script_path: str | Path | None = None,
    archive_path: str | Path | None = None,
    archive_manifest_path: str | Path | None = None,
    command: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    filename: str = "palace_handoff_metadata.json",
) -> Path:
    """Write Run Stage handoff metadata beside a Palace run or sweep point.

    The sidecar records intent and provenance for external execution. It does
    not claim that a scheduler submission happened and does not inspect Palace
    result files.

    Args:
        source: Run directory, sweep directory, or explicit JSON sidecar path.
        status: Producer-owned handoff status string.
        launcher: Optional launcher metadata such as Slurm/manual handoff
            intent.
        profile: Optional caller-owned profile metadata.
        resources: Optional requested/resolved resource metadata.
        script_path: Optional launcher script path recorded relative to the
            run or sweep root by callers.
        archive_path: Optional handoff archive path reference.
        archive_manifest_path: Optional archive manifest path reference.
        command: Optional redacted command metadata.
        metadata: Additional JSON-friendly metadata.
        filename: Sidecar filename. The default writes under ``metadata/`` for
            run folders; sweep metadata names write at the sweep root.

    Returns:
        Path to the written JSON sidecar.
    """
    source_path = Path(source)
    if source_path.suffix.lower() == ".json":
        sidecar_path = source_path
    elif filename == "palace_handoff_metadata.json":
        sidecar_path = source_path / "metadata" / filename
    else:
        sidecar_path = source_path / filename

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": str(status),
    }
    if launcher is not None:
        payload["launcher"] = _json_ready(dict(launcher))
    if profile is not None:
        payload["profile"] = _json_ready(dict(profile))
    if resources is not None:
        payload["resources"] = _json_ready(dict(resources))
    if script_path is not None:
        payload["script"] = {"path": str(script_path)}
    if archive_path is not None or archive_manifest_path is not None:
        archive = {}
        if archive_path is not None:
            archive["path"] = str(archive_path)
        if archive_manifest_path is not None:
            archive["manifest_path"] = str(archive_manifest_path)
        payload["archive"] = archive
    if command is not None:
        payload["command"] = _json_ready(dict(command))
    if metadata is not None:
        payload["metadata"] = _json_ready(dict(metadata))

    _write_json(sidecar_path, payload)
    return sidecar_path


@dataclass(frozen=True)
class PalaceSlurmResourceSpec:
    """Resolved Slurm resources for a Palace handoff script.

    This model is the scheduler-resource boundary for handoff rendering. It is
    intentionally explicit and validates Slurm-facing tokens eagerly so invalid
    profile catalogs fail before an sbatch script is written.
    """

    account: str
    partition: str
    wall_time: str
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    memory_mb: int | None = None
    gres: str | None = None

    def __post_init__(self) -> None:
        """Validate scheduler-facing resource values at construction."""
        _validate_sbatch_token("account", self.account)
        _validate_sbatch_token("partition", self.partition)
        if _WALL_TIME_PATTERN.match(self.wall_time) is None:
            raise ValueError("wall_time must use HH:MM:SS or D-HH:MM:SS")
        _validate_positive_int("nodes", self.nodes)
        _validate_positive_int("ntasks_per_node", self.ntasks_per_node)
        _validate_positive_int("cpus_per_task", self.cpus_per_task)
        if self.memory_mb is not None:
            _validate_positive_int("memory_mb", self.memory_mb)
        if self.gres is not None:
            _validate_sbatch_token("gres", self.gres)

    @property
    def num_processes(self) -> int:
        """Total MPI process count implied by Slurm tasks."""
        return self.nodes * self.ntasks_per_node

    @property
    def num_threads(self) -> int:
        """OpenMP thread count implied by ``--cpus-per-task``."""
        return self.cpus_per_task

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready resource record."""
        return {
            "account": self.account,
            "partition": self.partition,
            "wall_time": self.wall_time,
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks_per_node,
            "cpus_per_task": self.cpus_per_task,
            "memory_mb": self.memory_mb,
            "gres": self.gres,
            "num_processes": self.num_processes,
            "num_threads": self.num_threads,
        }


@dataclass(frozen=True)
class PalaceSlurmLauncherSpec:
    """Optional launch hints attached to a caller-supplied Slurm profile.

    Launcher hints describe how the generated script should invoke Palace on a
    site. They are not a submission API and do not encode private cluster
    policy beyond caller-provided executable, setup, PETSc, and ``srun``
    tokens.
    """

    palace_executable: str | None = None
    command_style: Literal["binary", "wrapper"] | None = None
    setup_commands: tuple[str, ...] | None = None
    petsc_options: tuple[str, ...] | None = None
    srun_args: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate optional launcher hints at construction."""
        if self.palace_executable is not None and (
            not self.palace_executable or "\n" in self.palace_executable
        ):
            raise ValueError("palace_executable must be single-line text")
        if self.command_style is not None and self.command_style not in {
            "binary",
            "wrapper",
        }:
            raise ValueError("command_style must be 'binary' or 'wrapper'")
        if self.setup_commands is not None:
            _validate_setup_commands(self.setup_commands)
        if self.petsc_options is not None:
            _validate_shell_tokens("petsc_options", self.petsc_options)
        if self.srun_args is not None:
            _validate_shell_tokens("srun_args", self.srun_args)

    def to_sbatch_kwargs(self) -> dict[str, Any]:
        """Return kwargs that can be passed to Slurm handoff spec classes."""
        payload: dict[str, Any] = {}
        if self.palace_executable is not None:
            payload["palace_executable"] = self.palace_executable
        if self.command_style is not None:
            payload["command_style"] = self.command_style
        if self.setup_commands is not None:
            payload["setup_commands"] = self.setup_commands
        if self.petsc_options is not None:
            payload["petsc_options"] = self.petsc_options
        if self.srun_args is not None:
            payload["srun_args"] = self.srun_args
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready launcher-hint record."""
        payload = self.to_sbatch_kwargs()
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


@dataclass(frozen=True)
class PalaceSlurmProfileSpec:
    """Caller-supplied Slurm profile with explicit Palace resources.

    ``gsim`` treats profiles as external configuration. The model validates the
    fields it understands, preserves JSON-friendly metadata for sidecars, and
    refuses unknown profile/resource/solver keys instead of silently accepting
    site-specific semantics.
    """

    name: str
    resources: PalaceSlurmResourceSpec
    source: str = "caller-supplied"
    description: str | None = None
    launcher: PalaceSlurmLauncherSpec = field(default_factory=PalaceSlurmLauncherSpec)
    solver: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate profile-owned resources, launcher, and metadata."""
        _validate_sbatch_token("profile name", self.name)
        if not isinstance(self.resources, PalaceSlurmResourceSpec):
            raise TypeError("resources must be a PalaceSlurmResourceSpec")
        _validate_single_line_text("profile source", self.source)
        if self.description is not None:
            _validate_single_line_text("profile description", self.description)
        if not isinstance(self.launcher, PalaceSlurmLauncherSpec):
            raise TypeError("launcher must be a PalaceSlurmLauncherSpec")
        _validate_profile_solver(self.solver)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

    def to_profile_metadata(
        self,
        *,
        resource_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return JSON-ready profile metadata for handoff sidecars."""
        payload: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
        }
        if self.description is not None:
            payload["description"] = self.description
        if launcher := self.launcher.to_dict():
            payload["launcher"] = launcher
        if self.solver:
            payload["solver"] = dict(self.solver)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if resource_overrides:
            payload["resource_overrides"] = dict(resource_overrides)
        return payload


@dataclass(frozen=True)
class PalaceSlurmProfileResolution:
    """Resolved Slurm resources and profile metadata for a handoff.

    A resolution is the immutable product of selecting one caller-owned
    profile and applying explicit resource overrides. It can feed both Palace
    config hints and sbatch rendering without re-reading the original catalog.
    """

    name: str
    resources: PalaceSlurmResourceSpec
    launcher: PalaceSlurmLauncherSpec
    solver: Mapping[str, Any]
    profile: Mapping[str, Any]
    resource_overrides: Mapping[str, Any] = field(default_factory=dict)

    def to_palace_config_hints(self) -> dict[str, Any]:
        """Return Palace config hints derived from profile solver metadata."""
        solver_hints = _palace_slurm_solver_config_hints(self.solver)
        return {"Solver": solver_hints} if solver_hints else {}

    def to_sbatch_spec(
        self,
        *,
        job_name: str,
        **sbatch_kwargs: Any,
    ) -> PalaceSlurmSbatchSpec:
        """Return a render-ready sbatch spec for this resolved profile."""
        return PalaceSlurmSbatchSpec(
            job_name=job_name,
            resources=self.resources,
            **self.launcher.to_sbatch_kwargs(),
            **sbatch_kwargs,
        )


@dataclass(frozen=True)
class PalaceSlurmSbatchSpec:
    """Render-ready Slurm sbatch specification for one Palace run folder.

    The spec owns script rendering only. It assumes the run folder contains
    ``config.json`` and ``palace.msh`` and emits a manual-submission script
    that writes logs/results under the canonical run-folder layout.
    """

    job_name: str
    resources: PalaceSlurmResourceSpec
    config_path: str = "config.json"
    mesh_path: str = "palace.msh"
    stdout_path: str = "logs/%x-%j.out"
    stderr_path: str = "logs/%x-%j.err"
    palace_executable: str = "palace"
    command_style: Literal["binary", "wrapper"] = "binary"
    setup_commands: tuple[str, ...] = ()
    petsc_options: tuple[str, ...] = DEFAULT_PALACE_PETSC_OPTIONS
    srun_args: tuple[str, ...] = ()
    mail_user: str | None = None
    mail_type: tuple[str, ...] = ("BEGIN", "END", "FAIL", "TIME_LIMIT")

    def __post_init__(self) -> None:
        """Validate a render-ready sbatch specification."""
        _validate_job_name(self.job_name)
        _validate_relative_path("config_path", self.config_path)
        _validate_relative_path("mesh_path", self.mesh_path)
        _validate_relative_path("stdout_path", self.stdout_path)
        _validate_relative_path("stderr_path", self.stderr_path)
        if not self.palace_executable or "\n" in self.palace_executable:
            raise ValueError("palace_executable must be single-line text")
        if self.command_style not in {"binary", "wrapper"}:
            raise ValueError("command_style must be 'binary' or 'wrapper'")
        _validate_setup_commands(self.setup_commands)
        _validate_shell_tokens("petsc_options", self.petsc_options)
        _validate_shell_tokens("srun_args", self.srun_args)
        if self.mail_user is not None:
            _validate_sbatch_token("mail_user", self.mail_user)
        if not self.mail_type:
            raise ValueError("mail_type must not be empty")
        _validate_shell_tokens("mail_type", self.mail_type)

    @property
    def num_processes(self) -> int:
        """Total Palace MPI process count."""
        return self.resources.num_processes

    @property
    def num_threads(self) -> int:
        """Palace OpenMP thread count."""
        return self.resources.num_threads

    def render(self) -> str:
        """Render the specification into a Slurm sbatch script."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --account={self.resources.account}",
            f"#SBATCH --partition={self.resources.partition}",
            f"#SBATCH --nodes={self.resources.nodes}",
            f"#SBATCH --ntasks-per-node={self.resources.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={self.resources.cpus_per_task}",
        ]
        if self.resources.gres is not None:
            lines.append(f"#SBATCH --gres={self.resources.gres}")
        if self.resources.memory_mb is not None:
            lines.append(f"#SBATCH --mem={self.resources.memory_mb}M")
        lines.extend(
            [
                f"#SBATCH --time={self.resources.wall_time}",
                f"#SBATCH --output={self.stdout_path}",
                f"#SBATCH --error={self.stderr_path}",
            ]
        )
        if self.mail_user is not None:
            lines.append(f"#SBATCH --mail-user={self.mail_user}")
            lines.append(f"#SBATCH --mail-type={','.join(self.mail_type)}")
        lines.extend(
            [
                "",
                "set -euo pipefail",
                "",
                'cd "${SLURM_SUBMIT_DIR:-$PWD}"',
                "",
                f"PALACE_CONFIG={shlex.quote(self.config_path)}",
                f"PALACE_MESH={shlex.quote(self.mesh_path)}",
                f"PALACE_EXECUTABLE={shlex.quote(self.palace_executable)}",
                f"PALACE_NUM_PROCESSES={self.num_processes}",
                f"PALACE_NUM_THREADS={self.num_threads}",
                "",
                "mkdir -p logs results/palace metadata",
                "",
                'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$PALACE_NUM_THREADS}"',
                "export OMPI_MCA_coll_hcoll_enable=0",
            ]
        )
        lines.extend(_render_petsc_options(self.petsc_options))
        lines.extend(
            [
                "",
                'echo "===== Slurm Info ====="',
                'echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"',
                'echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-unset}"',
                'echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"',
                'echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"',
                'echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"',
                'echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"',
                "",
                'if [[ ! -f "$PALACE_CONFIG" ]]; then',
                '  echo "Missing Palace config: $PALACE_CONFIG" >&2',
                "  exit 2",
                "fi",
                'if [[ ! -f "$PALACE_MESH" ]]; then',
                '  echo "Missing Palace mesh: $PALACE_MESH" >&2',
                "  exit 2",
                "fi",
                "",
            ]
        )
        lines.extend(_render_setup_commands(self.setup_commands))
        if self.setup_commands:
            lines.append("")
        lines.extend(
            [
                'echo "===== Runtime Info ====="',
                'echo "PALACE_EXECUTABLE=$PALACE_EXECUTABLE"',
                "command -v srun || true",
                '"$PALACE_EXECUTABLE" --version || true',
                "",
                'echo "===== Run Palace ====="',
                self._render_palace_command(),
                "",
            ]
        )
        return "\n".join(lines)

    def _render_palace_command(self) -> str:
        """Render the Palace launch command for a single Slurm run."""
        executable = '"$PALACE_EXECUTABLE"'
        config = '"$PALACE_CONFIG"'
        command = ["srun", *self.srun_args, executable]
        if self.command_style == "wrapper":
            command.extend(
                [
                    "-np",
                    '"$PALACE_NUM_PROCESSES"',
                    "-nt",
                    '"$PALACE_NUM_THREADS"',
                ]
            )
        command.append(config)
        log_path = '"logs/palace-${SLURM_JOB_ID:-manual}.log"'
        return f"{' '.join(command)} 2>&1 | tee {log_path}"


@dataclass(frozen=True)
class PalaceSlurmSweepArraySpec:
    """Render-ready Slurm array specification for a Palace sweep folder.

    The sweep script consumes the existing gsim ``points.json`` contract and a
    generated CSV lookup table. Point identity remains owned by sweep metadata;
    this class only controls scheduler array shape and Palace invocation.
    """

    job_name: str
    resources: PalaceSlurmResourceSpec
    max_parallel: int | None = None
    points_path: str = "points.json"
    points_csv_path: str = "points.csv"
    stdout_path: str = "logs/%x-%A_%a.out"
    stderr_path: str = "logs/%x-%A_%a.err"
    palace_executable: str = "palace"
    command_style: Literal["binary", "wrapper"] = "binary"
    setup_commands: tuple[str, ...] = ()
    petsc_options: tuple[str, ...] = DEFAULT_PALACE_PETSC_OPTIONS
    srun_args: tuple[str, ...] = ()
    mail_user: str | None = None
    mail_type: tuple[str, ...] = ("BEGIN", "END", "FAIL", "TIME_LIMIT")

    def __post_init__(self) -> None:
        """Validate a render-ready Slurm array specification."""
        _validate_job_name(self.job_name)
        if self.max_parallel is not None:
            _validate_positive_int("max_parallel", self.max_parallel)
        _validate_relative_path("points_path", self.points_path)
        _validate_relative_path("points_csv_path", self.points_csv_path)
        _validate_relative_path("stdout_path", self.stdout_path)
        _validate_relative_path("stderr_path", self.stderr_path)
        if not self.palace_executable or "\n" in self.palace_executable:
            raise ValueError("palace_executable must be single-line text")
        if self.command_style not in {"binary", "wrapper"}:
            raise ValueError("command_style must be 'binary' or 'wrapper'")
        _validate_setup_commands(self.setup_commands)
        _validate_shell_tokens("petsc_options", self.petsc_options)
        _validate_shell_tokens("srun_args", self.srun_args)
        if self.mail_user is not None:
            _validate_sbatch_token("mail_user", self.mail_user)
        if not self.mail_type:
            raise ValueError("mail_type must not be empty")
        _validate_shell_tokens("mail_type", self.mail_type)

    def resolved_max_parallel(self, point_count: int) -> int:
        """Return bounded Slurm array parallelism for ``point_count`` points."""
        _validate_positive_int("point_count", point_count)
        if self.max_parallel is None:
            return point_count
        return max(1, min(self.max_parallel, point_count))


@dataclass(frozen=True)
class PalaceSlurmHandoffResult:
    """Files written by a dry-run Palace Slurm handoff.

    The result names local artifacts only. It is not evidence that ``sbatch``
    was invoked or accepted by a cluster scheduler.
    """

    script_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PalaceSlurmSweepHandoffResult:
    """Files written by a dry-run Palace Slurm sweep-array handoff.

    The result names the generated script, handoff sidecar, and array lookup
    table. Scheduler submission remains a manual or caller-owned step.
    """

    script_path: Path
    metadata_path: Path
    points_csv_path: Path


@dataclass(frozen=True)
class PalaceHandoffArchiveManifestResult:
    """Files and counts written for a generated handoff archive manifest.

    ``archive_path`` is populated only when a tarball was actually written.
    Manifest-only helpers leave it unset unless the caller supplied a path to
    record in metadata.
    """

    manifest_path: Path
    metadata_path: Path | None
    file_count: int
    total_bytes: int
    archive_path: Path | None = None


def resolve_palace_slurm_profile(
    profiles: Mapping[str, PalaceSlurmProfileSpec | Mapping[str, Any]],
    name: str,
    *,
    resource_overrides: Mapping[str, Any] | None = None,
) -> PalaceSlurmProfileResolution:
    """Resolve a caller-supplied Slurm profile into Palace handoff resources.

    ``gsim`` intentionally does not ship private site catalogs or submit jobs.
    Callers provide a named profile mapping, and this helper validates that the
    selected profile can be converted to ``PalaceSlurmResourceSpec``.

    Args:
        profiles: Caller-owned profile catalog keyed by profile name.
        name: Profile key to resolve.
        resource_overrides: Explicit resource-field overrides applied after
            catalog validation.

    Returns:
        Immutable profile resolution with rendered metadata and resource
        objects ready for config hints or sbatch rendering.

    Raises:
        KeyError: If ``name`` is not present in ``profiles``.
        TypeError: If profile, launcher, resource, solver, or metadata fields
            have unsupported shapes.
        ValueError: If unknown fields or invalid Slurm-facing tokens are found.
    """
    _validate_sbatch_token("profile name", name)
    if name not in profiles:
        raise KeyError(f"Unknown Slurm profile: {name}")

    profile = _normalize_slurm_profile_spec(name, profiles[name])
    overrides = dict(resource_overrides or {})
    if overrides:
        allowed_fields = {field.name for field in fields(PalaceSlurmResourceSpec)}
        unknown_fields = sorted(set(overrides) - allowed_fields)
        if unknown_fields:
            msg = "Unknown Slurm resource override field(s): "
            msg += ", ".join(unknown_fields)
            raise ValueError(msg)
    resources = replace(profile.resources, **overrides)
    return PalaceSlurmProfileResolution(
        name=profile.name,
        resources=resources,
        launcher=profile.launcher,
        solver=profile.solver,
        profile=profile.to_profile_metadata(resource_overrides=overrides),
        resource_overrides=overrides,
    )


def load_palace_slurm_profile_catalog(
    path: str | Path,
) -> dict[str, PalaceSlurmProfileSpec]:
    """Load a caller-owned Slurm profile catalog from JSON.

    The JSON file may be either a direct mapping of profile names to profile
    specs or an envelope with ``schema_version: 1`` and a ``profiles`` mapping.

    Args:
        path: JSON catalog path.

    Returns:
        Validated profile specs keyed by profile name.

    Raises:
        FileNotFoundError: If the catalog path does not exist.
        TypeError: If the catalog payload is not a JSON object.
        ValueError: If the catalog extension, schema version, or profile fields
            do not match the supported contract.
    """
    catalog_path = Path(path)
    if catalog_path.suffix.lower() != ".json":
        raise ValueError("Slurm profile catalogs must be JSON files")
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Slurm profile catalog must contain a JSON object")
    profiles_payload = _slurm_profile_catalog_profiles(payload)
    return {
        str(name): _normalize_slurm_profile_spec(str(name), profile)
        for name, profile in profiles_payload.items()
    }


def _palace_slurm_solver_config_hints(
    solver: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Convert Slurm profile solver metadata into Palace ``Solver`` hints."""
    if solver is None:
        return {}
    normalized = _normalize_slurm_profile_solver(solver)
    hints: dict[str, Any] = {}
    if device := normalized.get("device"):
        hints["Device"] = device
    if backend := normalized.get("backend"):
        hints["Backend"] = backend
    return hints


def write_palace_slurm_sbatch_handoff(
    source: str | Path,
    spec: PalaceSlurmSbatchSpec,
    *,
    script_path: str | Path = "run_palace.sbatch",
    profile: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    validate_inputs: bool = True,
) -> PalaceSlurmHandoffResult:
    """Write a Slurm sbatch script and Palace handoff metadata without submitting.

    The helper is intentionally a dry-run handoff writer. It prepares the
    canonical run-folder directories, renders an executable script, and records
    manual-submission intent in ``metadata/palace_handoff_metadata.json``.

    Args:
        source: Palace run directory containing ``config.json`` and
            ``palace.msh``.
        spec: Render-ready Slurm sbatch specification.
        script_path: Relative path for the generated sbatch script.
        profile: Optional caller-supplied profile metadata. ``gsim`` records it
            but does not resolve site-specific profile catalogs.
        metadata: Additional JSON-friendly handoff metadata.
        validate_inputs: When true, require the referenced config and mesh files.

    Returns:
        Paths to the generated script and metadata sidecar.

    Raises:
        FileNotFoundError: If validation is enabled and required Palace inputs
            are absent.
        ValueError: If ``source`` is not a run directory path or generated
            paths are not run-folder-relative.
    """
    run_dir = Path(source)
    if run_dir.suffix:
        raise ValueError("source must be a run directory")
    run_folder = prepare_palace_run_folder(run_dir)
    run_dir = run_folder.root
    configured_mesh_path = run_folder.mesh_path
    configured_mesh = configured_mesh_path.relative_to(run_dir).as_posix()
    if spec.mesh_path not in {"palace.msh", configured_mesh}:
        raise ValueError(
            "PalaceSlurmSbatchSpec.mesh_path conflicts with config Model.Mesh"
        )
    spec = replace(spec, mesh_path=configured_mesh)
    _validate_relative_path("script_path", str(script_path))
    output_script_path = run_dir / script_path
    if validate_inputs:
        _require_file(run_dir / spec.config_path, "Palace config")
        _require_file(run_dir / spec.mesh_path, "Palace mesh")

    for log_path in (spec.stdout_path, spec.stderr_path):
        (run_dir / log_path).parent.mkdir(parents=True, exist_ok=True)
    output_script_path.parent.mkdir(parents=True, exist_ok=True)
    output_script_path.write_text(spec.render(), encoding="utf-8")
    output_script_path.chmod(output_script_path.stat().st_mode | 0o755)

    user_metadata = dict(metadata or {})
    user_metadata.setdefault("script_schema_version", 1)
    user_metadata.setdefault("command_style", spec.command_style)
    sidecar_path = write_palace_handoff_metadata(
        run_dir,
        status="scripted",
        launcher={
            "kind": "slurm",
            "submission": "manual",
            "dry_run": True,
        },
        profile=profile,
        resources={
            "requested": spec.resources.to_dict(),
            "resolved": spec.resources.to_dict(),
        },
        script_path=script_path,
        command={"argv": ["sbatch", str(script_path)], "redacted": True},
        metadata=user_metadata,
    )
    return PalaceSlurmHandoffResult(
        script_path=output_script_path,
        metadata_path=sidecar_path,
    )


def write_palace_slurm_sweep_array_handoff(
    source: str | Path,
    spec: PalaceSlurmSweepArraySpec,
    *,
    script_path: str | Path = "run_sweep_array.sbatch",
    profile: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    validate_inputs: bool = True,
) -> PalaceSlurmSweepHandoffResult:
    """Write a Slurm array script and sweep handoff metadata without submitting.

    The sweep must already provide ``points.json``. Point rows are converted to
    a Slurm array lookup table, keeping point identity in the existing gsim
    sweep metadata instead of introducing a private campaign format.

    Args:
        source: Sweep root containing ``points.json``.
        spec: Render-ready Slurm array specification.
        script_path: Relative path for the generated array script.
        profile: Optional caller-supplied profile metadata.
        metadata: Additional JSON-friendly handoff metadata.
        validate_inputs: When true, require every referenced point config and
            mesh before writing the script.

    Returns:
        Paths to the generated script, sweep handoff sidecar, and CSV lookup
        table.

    Raises:
        FileNotFoundError: If ``points.json`` or required point artifacts are
            missing.
        TypeError: If ``points.json`` does not contain the supported object or
            list shape.
        ValueError: If point identity, relative paths, or scheduler parameters
            violate the handoff contract.
    """
    sweep_root = Path(source)
    if sweep_root.suffix:
        raise ValueError("source must be a sweep directory")
    _validate_relative_path("script_path", str(script_path))
    points_path = sweep_root / spec.points_path
    if not points_path.is_file():
        raise FileNotFoundError(points_path)

    points_payload = json.loads(points_path.read_text(encoding="utf-8"))
    point_specs = _sweep_point_specs(points_payload)
    if not point_specs:
        raise ValueError("points.json must contain at least one point")

    point_rows = _sweep_array_rows(sweep_root, point_specs)
    if validate_inputs:
        for row in point_rows:
            _require_file(sweep_root / str(row["config_path"]), "Palace config")
            _require_file(sweep_root / str(row["mesh_path"]), "Palace mesh")

    points_csv_path = sweep_root / spec.points_csv_path
    _write_sweep_array_points_csv(points_csv_path, point_rows)
    max_parallel = spec.resolved_max_parallel(len(point_rows))
    output_script_path = sweep_root / script_path
    output_script_path.parent.mkdir(parents=True, exist_ok=True)
    output_script_path.write_text(
        _render_sweep_array_sbatch(spec, point_rows, max_parallel),
        encoding="utf-8",
    )
    output_script_path.chmod(output_script_path.stat().st_mode | 0o755)

    user_metadata = dict(metadata or {})
    user_metadata.setdefault("script_schema_version", 1)
    user_metadata.setdefault("command_style", spec.command_style)
    user_metadata.setdefault("points_path", spec.points_path)
    user_metadata.setdefault("points_csv_path", spec.points_csv_path)
    sidecar_path = write_palace_handoff_metadata(
        sweep_root,
        status="scripted",
        launcher={
            "kind": "slurm",
            "array": True,
            "submission": "manual",
            "dry_run": True,
        },
        profile=profile,
        resources={
            "requested": spec.resources.to_dict(),
            "resolved": spec.resources.to_dict(),
            "array": {
                "point_count": len(point_rows),
                "max_parallel": max_parallel,
            },
        },
        script_path=script_path,
        command={"argv": ["sbatch", str(script_path)], "redacted": True},
        metadata=user_metadata,
        filename="palace_sweep_handoff_metadata.json",
    )
    return PalaceSlurmSweepHandoffResult(
        script_path=output_script_path,
        metadata_path=sidecar_path,
        points_csv_path=points_csv_path,
    )


def write_palace_run_handoff_archive_manifest(
    source: str | Path,
    *,
    manifest_path: str | Path = "metadata/palace_handoff_archive_manifest.json",
    archive_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_results: bool = False,
    include_hashes: bool = True,
    update_handoff_metadata: bool = True,
    handoff_metadata_filename: str = "palace_handoff_metadata.json",
) -> PalaceHandoffArchiveManifestResult:
    """Write a reviewable manifest for a Palace run handoff archive.

    The helper records which generated files would be packaged. It does not
    create an archive, submit a job, or infer site policy.

    Args:
        source: Palace run directory.
        manifest_path: Run-folder-relative manifest path.
        archive_path: Optional archive path reference recorded in handoff
            metadata and manifest payload.
        metadata: Additional JSON-friendly manifest metadata.
        include_results: Include already-present solver results in the
            manifest.
        include_hashes: Include SHA-256 checksums for present files.
        update_handoff_metadata: Update or create the handoff sidecar with the
            manifest reference.
        handoff_metadata_filename: Handoff sidecar filename to update/read.

    Returns:
        Manifest path, optional metadata path, file count, total bytes, and
        optional archive reference.
    """
    run_dir = Path(source)
    if run_dir.suffix:
        raise ValueError("source must be a run directory")
    run_dir = prepare_palace_run_folder(run_dir).root
    _validate_relative_path("manifest_path", str(manifest_path))
    output_manifest_path = run_dir / manifest_path
    metadata_path = (
        _record_handoff_archive_manifest(
            run_dir,
            filename=handoff_metadata_filename,
            manifest_path=manifest_path,
            archive_path=archive_path,
        )
        if update_handoff_metadata
        else _existing_handoff_metadata_path(run_dir, handoff_metadata_filename)
    )

    files = _run_archive_manifest_entries_from_folder(
        run_dir,
        include_results=include_results,
        include_hashes=include_hashes,
        excluded_paths=(output_manifest_path,),
    )
    payload = _archive_manifest_payload(
        source_kind="run",
        files=files,
        include_results=include_results,
        archive_path=archive_path,
        metadata=metadata,
    )
    _write_json(output_manifest_path, payload)
    return PalaceHandoffArchiveManifestResult(
        manifest_path=output_manifest_path,
        metadata_path=metadata_path,
        file_count=int(payload["file_count"]),
        total_bytes=int(payload["total_bytes"]),
        archive_path=None if archive_path is None else Path(archive_path),
    )


def package_palace_run_handoff_archive(
    source: str | Path,
    *,
    archive_path: str | Path | None = None,
    manifest_path: str | Path = "metadata/palace_handoff_archive_manifest.json",
    metadata: Mapping[str, Any] | None = None,
    include_results: bool = False,
    include_hashes: bool = True,
) -> PalaceHandoffArchiveManifestResult:
    """Write the manifest and package a canonical Palace run folder as tar.gz.

    The archive root is the run-folder name. By default, existing result and
    log contents are excluded while the ``results/palace`` and ``logs``
    directories remain present for HPC handoff.

    Args:
        source: Palace run directory.
        archive_path: Optional tarball target. When omitted, the archive is
            written beside the run folder.
        manifest_path: Run-folder-relative manifest path.
        metadata: Additional JSON-friendly manifest metadata.
        include_results: Include already-present solver results and logs in the
            archive.
        include_hashes: Include SHA-256 checksums in the manifest.

    Returns:
        Manifest/archive summary with ``archive_path`` set to the written
        tarball.

    Raises:
        ValueError: If ``archive_path`` resolves inside the run folder.
    """
    run_folder = prepare_palace_run_folder(source)
    output_archive_path = _resolve_handoff_archive_path(
        run_folder.root,
        archive_path,
    )
    archive_reference = _archive_path_reference(
        run_folder.root,
        output_archive_path,
    )
    manifest_result = write_palace_run_handoff_archive_manifest(
        run_folder.root,
        manifest_path=manifest_path,
        archive_path=archive_reference,
        metadata=metadata,
        include_results=include_results,
        include_hashes=include_hashes,
    )
    output_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_archive_path, "w:gz") as archive:
        archive.add(
            run_folder.root,
            arcname=run_folder.root.name,
            recursive=True,
            filter=lambda info: _filter_run_handoff_tarinfo(
                info,
                run_folder.root.name,
                include_results=include_results,
            ),
        )
    return PalaceHandoffArchiveManifestResult(
        manifest_path=manifest_result.manifest_path,
        metadata_path=manifest_result.metadata_path,
        file_count=manifest_result.file_count,
        total_bytes=manifest_result.total_bytes,
        archive_path=output_archive_path,
    )


def write_palace_sweep_handoff_archive_manifest(
    source: str | Path,
    *,
    manifest_path: str | Path = "palace_sweep_handoff_archive_manifest.json",
    archive_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_point_files: bool = True,
    include_results: bool = False,
    include_hashes: bool = True,
    update_handoff_metadata: bool = True,
    handoff_metadata_filename: str = "palace_sweep_handoff_metadata.json",
) -> PalaceHandoffArchiveManifestResult:
    """Write a reviewable manifest for a Palace sweep handoff archive.

    The manifest can include sweep-level handoff artifacts and, by default, the
    already-prepared point run folders referenced by ``points.json``. It does
    not create a tarball or submit scheduler work.

    Args:
        source: Sweep root or explicit ``points.json`` path.
        manifest_path: Sweep-root-relative manifest path.
        archive_path: Optional archive path reference recorded in metadata.
        metadata: Additional JSON-friendly manifest metadata.
        include_point_files: Include artifacts for each point run folder.
        include_results: Include already-present point solver outputs.
        include_hashes: Include SHA-256 checksums for present files.
        update_handoff_metadata: Update or create sweep handoff metadata with
            the manifest reference.
        handoff_metadata_filename: Sweep handoff sidecar filename to
            update/read.

    Returns:
        Manifest path, optional metadata path, file count, and total bytes.
    """
    source_path = Path(source)
    points_path = source_path if source_path.is_file() else source_path / "points.json"
    if not points_path.is_file():
        raise FileNotFoundError(points_path)
    sweep_root = points_path.parent
    _validate_relative_path("manifest_path", str(manifest_path))
    output_manifest_path = sweep_root / manifest_path
    metadata_path = (
        _record_handoff_archive_manifest(
            sweep_root,
            filename=handoff_metadata_filename,
            manifest_path=manifest_path,
            archive_path=archive_path,
        )
        if update_handoff_metadata
        else _existing_handoff_metadata_path(sweep_root, handoff_metadata_filename)
    )

    points_payload = json.loads(points_path.read_text(encoding="utf-8"))
    point_specs = _sweep_point_specs(points_payload)
    point_rows = _sweep_array_rows(sweep_root, point_specs)
    files: list[dict[str, Any]] = []
    _append_manifest_entry(
        files,
        sweep_root,
        points_path,
        role="sweep_points",
        name=points_path.name,
        include_hashes=include_hashes,
    )
    if metadata_path is not None:
        handoff_payload = _read_json_mapping(metadata_path)
        handoff = {
            "present": True,
            "path": metadata_path,
            **handoff_payload,
        }
        metadata_payload = _as_mapping(handoff.get("metadata"))
        points_csv_path = metadata_payload.get("points_csv_path")
        if points_csv_path is not None:
            _append_manifest_entry(
                files,
                sweep_root,
                _resolve_sidecar_reference(points_path, points_csv_path),
                role="sweep_points_table",
                name=Path(str(points_csv_path)).name,
                include_hashes=include_hashes,
            )
        _extend_handoff_reference_entries(
            files,
            sweep_root,
            handoff,
            include_hashes=include_hashes,
            role_prefix="sweep_",
        )
    if include_point_files:
        for point in point_rows:
            files.extend(
                _run_archive_manifest_entries_from_folder(
                    sweep_root / str(point["run_dir"]),
                    include_results=include_results,
                    include_hashes=include_hashes,
                    manifest_root=sweep_root,
                    point_slug=str(point["point_slug"]),
                )
            )
    files = _deduplicate_manifest_entries(files)
    payload = _archive_manifest_payload(
        source_kind="sweep",
        files=files,
        include_results=include_results,
        archive_path=archive_path,
        metadata={
            "include_point_files": include_point_files,
            **dict(metadata or {}),
        },
    )
    _write_json(output_manifest_path, payload)
    return PalaceHandoffArchiveManifestResult(
        manifest_path=output_manifest_path,
        metadata_path=metadata_path,
        file_count=int(payload["file_count"]),
        total_bytes=int(payload["total_bytes"]),
    )


def _render_sweep_array_sbatch(
    spec: PalaceSlurmSweepArraySpec,
    point_rows: Sequence[Mapping[str, str | int]],
    max_parallel: int,
) -> str:
    """Render a Slurm array script for resolved Palace sweep points."""
    array_last_index = len(point_rows) - 1
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --account={spec.resources.account}",
        f"#SBATCH --partition={spec.resources.partition}",
        f"#SBATCH --nodes={spec.resources.nodes}",
        f"#SBATCH --ntasks-per-node={spec.resources.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={spec.resources.cpus_per_task}",
    ]
    if spec.resources.gres is not None:
        lines.append(f"#SBATCH --gres={spec.resources.gres}")
    if spec.resources.memory_mb is not None:
        lines.append(f"#SBATCH --mem={spec.resources.memory_mb}M")
    lines.extend(
        [
            f"#SBATCH --time={spec.resources.wall_time}",
            f"#SBATCH --array=0-{array_last_index}%{max_parallel}",
            f"#SBATCH --output={spec.stdout_path}",
            f"#SBATCH --error={spec.stderr_path}",
        ]
    )
    if spec.mail_user is not None:
        lines.append(f"#SBATCH --mail-user={spec.mail_user}")
        lines.append(f"#SBATCH --mail-type={','.join(spec.mail_type)}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            'cd "${SLURM_SUBMIT_DIR:-$PWD}"',
            "",
            f"POINTS_CSV={shlex.quote(spec.points_csv_path)}",
            f"PALACE_EXECUTABLE={shlex.quote(spec.palace_executable)}",
            f"PALACE_NUM_PROCESSES={spec.resources.num_processes}",
            f"PALACE_NUM_THREADS={spec.resources.num_threads}",
            "",
            "mkdir -p logs results metadata",
            "",
            'if [[ ! -f "$POINTS_CSV" ]]; then',
            '  echo "Missing sweep points table: $POINTS_CSV" >&2',
            "  exit 2",
            "fi",
            "",
            'POINT_ROW="$(awk -F, -v idx="$SLURM_ARRAY_TASK_ID" '
            "'NR > 1 && $1 == idx { print; found=1 } "
            'END { if (!found) exit 1 }\' "$POINTS_CSV")"',
            "IFS=, read -r ARRAY_INDEX POINT_SLUG RUN_DIR CONFIG_PATH MESH_PATH "
            'LOG_DIR RESULT_DIR <<< "$POINT_ROW"',
            "",
            'if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then',
            '  echo "Missing run directory for array task '
            '$SLURM_ARRAY_TASK_ID: $RUN_DIR" >&2',
            "  exit 2",
            "fi",
            'if [[ ! -f "$CONFIG_PATH" ]]; then',
            '  echo "Missing Palace config for array task '
            '$SLURM_ARRAY_TASK_ID: $CONFIG_PATH" >&2',
            "  exit 2",
            "fi",
            'if [[ ! -f "$MESH_PATH" ]]; then',
            '  echo "Missing Palace mesh for array task '
            '$SLURM_ARRAY_TASK_ID: $MESH_PATH" >&2',
            "  exit 2",
            "fi",
            "",
            'echo "===== Sweep Point ====="',
            'echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"',
            'echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"',
            'echo "POINT_SLUG=$POINT_SLUG"',
            'echo "RUN_DIR=$RUN_DIR"',
            'echo "CONFIG_PATH=$CONFIG_PATH"',
            'echo "MESH_PATH=$MESH_PATH"',
            'echo "LOG_DIR=$LOG_DIR"',
            'echo "RESULT_DIR=$RESULT_DIR"',
            "",
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$PALACE_NUM_THREADS}"',
            "export OMPI_MCA_coll_hcoll_enable=0",
        ]
    )
    lines.extend(_render_petsc_options(spec.petsc_options))
    lines.extend(
        [
            "",
            'echo "===== Slurm Info ====="',
            'echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"',
            'echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"',
            'echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"',
            'echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"',
            "",
            'mkdir -p "$LOG_DIR" "$RESULT_DIR" "metadata/$POINT_SLUG"',
            "",
        ]
    )
    lines.extend(_render_setup_commands(spec.setup_commands))
    if spec.setup_commands:
        lines.append("")
    lines.extend(
        [
            'echo "===== Runtime Info ====="',
            'echo "PALACE_EXECUTABLE=$PALACE_EXECUTABLE"',
            "command -v srun || true",
            '"$PALACE_EXECUTABLE" --version || true',
            "",
            'echo "===== Run Palace ====="',
            _render_sweep_palace_command(spec),
            "",
        ]
    )
    return "\n".join(lines)


def _render_sweep_palace_command(spec: PalaceSlurmSweepArraySpec) -> str:
    """Render the Palace command executed by one Slurm array task."""
    executable = '"$PALACE_EXECUTABLE"'
    config = '"$CONFIG_PATH"'
    command = ["srun", *spec.srun_args, executable]
    if spec.command_style == "wrapper":
        command.extend(
            [
                "-np",
                '"$PALACE_NUM_PROCESSES"',
                "-nt",
                '"$PALACE_NUM_THREADS"',
            ]
        )
    command.append(config)
    log_path = '"$LOG_DIR/palace-${SLURM_ARRAY_TASK_ID:-manual}.log"'
    return f"{' '.join(command)} 2>&1 | tee {log_path}"


def _normalize_slurm_profile_spec(
    name: str,
    profile: PalaceSlurmProfileSpec | Mapping[str, Any],
) -> PalaceSlurmProfileSpec:
    """Normalize one caller profile into its validated typed representation."""
    if isinstance(profile, PalaceSlurmProfileSpec):
        if profile.name != name:
            raise ValueError("profile mapping key must match profile.name")
        return profile
    if not isinstance(profile, Mapping):
        raise TypeError("profile must be a PalaceSlurmProfileSpec or mapping")

    allowed_fields = {
        "description",
        "launcher",
        "metadata",
        "name",
        "resources",
        "solver",
        "source",
    }
    unknown_fields = sorted(set(profile) - allowed_fields)
    if unknown_fields:
        msg = "Unknown Slurm profile field(s): "
        msg += ", ".join(str(field) for field in unknown_fields)
        raise ValueError(msg)

    profile_name = str(profile.get("name", name))
    if profile_name != name:
        raise ValueError("profile mapping key must match profile name")
    if "resources" not in profile:
        raise ValueError("profile must include resources")
    resources = _normalize_slurm_profile_resources(profile["resources"])
    launcher = _normalize_slurm_profile_launcher(profile.get("launcher", {}))
    solver = _normalize_slurm_profile_solver(profile.get("solver", {}))
    metadata = profile.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("profile metadata must be a mapping")
    description = profile.get("description")
    source = str(profile.get("source", "caller-supplied"))
    return PalaceSlurmProfileSpec(
        name=profile_name,
        resources=resources,
        source=source,
        description=None if description is None else str(description),
        launcher=launcher,
        solver=solver,
        metadata=metadata,
    )


def _normalize_slurm_profile_resources(
    resources: PalaceSlurmResourceSpec | Mapping[str, Any],
) -> PalaceSlurmResourceSpec:
    """Normalize caller resource data into a validated resource specification."""
    if isinstance(resources, PalaceSlurmResourceSpec):
        return resources
    if not isinstance(resources, Mapping):
        raise TypeError(
            "profile resources must be a PalaceSlurmResourceSpec or mapping"
        )
    allowed_fields = {field.name for field in fields(PalaceSlurmResourceSpec)}
    unknown_fields = sorted(set(resources) - allowed_fields)
    if unknown_fields:
        msg = "Unknown Slurm resource field(s): "
        msg += ", ".join(str(field) for field in unknown_fields)
        raise ValueError(msg)
    return PalaceSlurmResourceSpec(**dict(resources))


def _normalize_slurm_profile_launcher(
    launcher: PalaceSlurmLauncherSpec | Mapping[str, Any],
) -> PalaceSlurmLauncherSpec:
    """Normalize caller launcher data into a validated launcher specification."""
    if isinstance(launcher, PalaceSlurmLauncherSpec):
        return launcher
    if not isinstance(launcher, Mapping):
        raise TypeError("profile launcher must be a PalaceSlurmLauncherSpec or mapping")
    allowed_fields = {
        "command_style",
        "palace_executable",
        "petsc_options",
        "setup_commands",
        "srun_args",
    }
    unknown_fields = sorted(set(launcher) - allowed_fields)
    if unknown_fields:
        msg = "Unknown Slurm launcher field(s): "
        msg += ", ".join(str(field) for field in unknown_fields)
        raise ValueError(msg)
    command_style_value = _optional_string(launcher.get("command_style"))
    if command_style_value is None:
        command_style = None
    elif command_style_value == "binary":
        command_style = "binary"
    elif command_style_value == "wrapper":
        command_style = "wrapper"
    else:
        raise ValueError("command_style must be 'binary' or 'wrapper'")
    return PalaceSlurmLauncherSpec(
        palace_executable=_optional_string(launcher.get("palace_executable")),
        command_style=command_style,
        setup_commands=_optional_tuple(launcher.get("setup_commands")),
        petsc_options=_optional_tuple(launcher.get("petsc_options")),
        srun_args=_optional_tuple(launcher.get("srun_args")),
    )


def _normalize_slurm_profile_solver(solver: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize supported Slurm solver hint fields."""
    if not isinstance(solver, Mapping):
        raise TypeError("profile solver must be a mapping")
    allowed_fields = {"backend", "device"}
    unknown_fields = sorted(set(solver) - allowed_fields)
    if unknown_fields:
        msg = "Unknown Slurm profile solver field(s): "
        msg += ", ".join(str(field) for field in unknown_fields)
        raise ValueError(msg)
    normalized: dict[str, Any] = {}
    for key, value in solver.items():
        if value is None:
            continue
        normalized[key] = str(value)
        _validate_single_line_text(f"solver {key}", normalized[key])
    return normalized


def _slurm_profile_catalog_profiles(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Extract profile mappings from a supported Slurm catalog payload."""
    if "schema_version" not in payload and "profiles" not in payload:
        return payload

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("Slurm profile catalog schema_version must be 1")
    allowed_fields = {"metadata", "profiles", "schema_version"}
    unknown_fields = sorted(set(payload) - allowed_fields)
    if unknown_fields:
        msg = "Unknown Slurm profile catalog field(s): "
        msg += ", ".join(str(field) for field in unknown_fields)
        raise ValueError(msg)
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping):
        raise TypeError("Slurm profile catalog field 'profiles' must be a mapping")
    return profiles


def _sweep_point_specs(payload: Any) -> list[Mapping[str, Any]]:
    """Extract and validate point mappings from a sweep payload."""
    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        points = payload.get("points", [])
    else:
        raise TypeError("points.json must contain an object or list")
    if not isinstance(points, list):
        raise TypeError("points.json field 'points' must be a list")
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise TypeError(f"points.json point at index {index} must be an object")
    return points


def _sweep_array_rows(
    sweep_root: Path,
    point_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, str | int]]:
    """Build validated Slurm array rows from sweep point specifications."""
    rows: list[dict[str, str | int]] = []
    seen_slugs: set[str] = set()
    for index, point in enumerate(point_specs):
        point_slug = str(point.get("point_slug") or f"point_{index:04d}")
        _validate_csv_field("point_slug", point_slug)
        if point_slug in seen_slugs:
            msg = f"points.json contains duplicate point_slug {point_slug!r}"
            raise ValueError(msg)
        seen_slugs.add(point_slug)

        run_dir = str(point.get("run_dir") or Path("points") / point_slug)
        config_path = str(point.get("config_path") or Path(run_dir) / "config.json")
        mesh_path = str(point.get("mesh_path") or Path(run_dir) / "palace.msh")
        log_dir = str(point.get("log_dir") or Path("logs") / point_slug)
        result_dir = str(
            point.get("result_dir") or Path("results") / point_slug / "palace"
        )
        for label, value in (
            ("run_dir", run_dir),
            ("config_path", config_path),
            ("mesh_path", mesh_path),
            ("log_dir", log_dir),
            ("result_dir", result_dir),
        ):
            _validate_relative_path(label, value)
            _validate_csv_field(label, value)
        rows.append(
            {
                "array_index": index,
                "point_slug": point_slug,
                "run_dir": _normalize_sweep_path(sweep_root, run_dir),
                "config_path": _normalize_sweep_path(sweep_root, config_path),
                "mesh_path": _normalize_sweep_path(sweep_root, mesh_path),
                "log_dir": _normalize_sweep_path(sweep_root, log_dir),
                "result_dir": _normalize_sweep_path(sweep_root, result_dir),
            }
        )
    return rows


def _write_sweep_array_points_csv(
    path: Path,
    rows: Sequence[Mapping[str, str | int]],
) -> None:
    """Write the deterministic point lookup CSV used by a Slurm array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames: tuple[str, ...] = (
            "array_index",
            "point_slug",
            "run_dir",
            "config_path",
            "mesh_path",
            "log_dir",
            "result_dir",
        )
        writer: csv.DictWriter[str] = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _normalize_sweep_path(sweep_root: Path, value: str) -> str:
    """Return a sweep-relative POSIX path for a point artifact."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve().relative_to(sweep_root.resolve()).as_posix()
    return path.as_posix()


def _render_petsc_options(options: Sequence[str]) -> list[str]:
    """Render shell statements that append configured PETSc options."""
    if not options:
        return []
    option_string = shlex.quote(" ".join(options))
    return [
        "",
        f"PALACE_PETSC_OPTIONS={option_string}",
        'if [[ -n "${PETSC_OPTIONS:-}" ]]; then',
        '  export PETSC_OPTIONS="$PETSC_OPTIONS $PALACE_PETSC_OPTIONS"',
        "else",
        '  export PETSC_OPTIONS="$PALACE_PETSC_OPTIONS"',
        "fi",
        'echo "PETSC_OPTIONS=$PETSC_OPTIONS"',
    ]


def _render_setup_commands(commands: Sequence[str]) -> list[str]:
    """Render sourced vendor environments without leaking the nounset override."""
    lines: list[str] = []
    for command in commands:
        if command.lstrip().startswith((". ", "source ")):
            lines.extend(("set +u", command, "set -u"))
        else:
            lines.append(command)
    return lines


def _validate_sbatch_token(label: str, value: str) -> None:
    """Reject empty or shell-unsafe Slurm directive tokens."""
    if not value:
        raise ValueError(f"{label} must not be empty")
    if any(char in _SBATCH_TOKEN_FORBIDDEN for char in value):
        raise ValueError(f"{label} must not contain whitespace or shell metacharacters")


def _validate_job_name(value: str) -> None:
    """Reject job names unsafe for a Slurm directive."""
    if not value:
        raise ValueError("job_name must not be empty")
    if any(char in _SBATCH_JOB_NAME_FORBIDDEN for char in value):
        raise ValueError("job_name must be a Slurm-safe token")


def _validate_single_line_text(label: str, value: str) -> None:
    """Reject empty or multiline text fields."""
    if not value:
        raise ValueError(f"{label} must not be empty")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be single-line text")


def _optional_tuple(value: Any) -> tuple[str, ...] | None:
    """Normalize an optional launcher scalar or sequence to string tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError("launcher tuple fields must be strings or sequences")
    return tuple(str(item) for item in value)


def _validate_profile_solver(value: Mapping[str, Any]) -> None:
    """Validate supported Slurm profile solver hints."""
    _normalize_slurm_profile_solver(value)


def _validate_shell_tokens(label: str, values: Sequence[str]) -> None:
    """Validate each token intended for shell rendering."""
    for value in values:
        _validate_sbatch_token(label, value)


def _validate_setup_commands(commands: Sequence[str]) -> None:
    """Validate one nonempty single-line shell command per setup entry."""
    for command in commands:
        if not command.strip():
            raise ValueError("setup_commands must not contain empty commands")
        if "\n" in command or "\r" in command:
            raise ValueError("setup_commands must be one shell command per entry")


def _validate_relative_path(label: str, value: str) -> None:
    """Reject paths outside the run directory or containing line breaks."""
    if not value:
        raise ValueError(f"{label} must not be empty")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative to the run directory")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be single-line text")


def _validate_csv_field(label: str, value: str) -> None:
    """Reject values that cannot occupy one unquoted CSV field."""
    if "," in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must not contain CSV separators")


def _validate_positive_int(label: str, value: int) -> None:
    """Reject nonpositive integer settings."""
    if value < 1:
        raise ValueError(f"{label} must be positive")


def _require_file(path: Path, label: str) -> None:
    """Raise a contextual error unless a required file exists."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _record_handoff_archive_manifest(
    source: Path,
    *,
    filename: str,
    manifest_path: str | Path,
    archive_path: str | Path | None,
) -> Path:
    """Attach an archive-manifest reference to existing handoff metadata."""
    existing_path = _existing_handoff_metadata_path(source, filename)
    existing = (
        json.loads(existing_path.read_text(encoding="utf-8"))
        if existing_path is not None
        else {}
    )
    script = _as_mapping(existing.get("script"))
    archive = _as_mapping(existing.get("archive"))
    return write_palace_handoff_metadata(
        source,
        status=str(existing.get("status") or "manifested"),
        launcher=_optional_dict(existing.get("launcher")),
        profile=_optional_dict(existing.get("profile")),
        resources=_optional_dict(existing.get("resources")),
        script_path=script.get("path"),
        archive_path=archive_path if archive_path is not None else archive.get("path"),
        archive_manifest_path=manifest_path,
        command=_optional_dict(existing.get("command")),
        metadata=_optional_dict(existing.get("metadata")),
        filename=filename,
    )


def _existing_handoff_metadata_path(source: Path, filename: str) -> Path | None:
    """Return an existing handoff metadata file in supported locations."""
    candidates = [source / "metadata" / filename, source / filename]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _run_archive_manifest_entries_from_folder(
    root: Path,
    *,
    include_results: bool,
    include_hashes: bool,
    manifest_root: Path | None = None,
    point_slug: str | None = None,
    excluded_paths: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Collect every file the run-handoff archive includes, except exclusions."""
    files: list[dict[str, Any]] = []
    entry_root = root if manifest_root is None else manifest_root
    excluded = {path.resolve() for path in excluded_paths}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        relative = path.relative_to(root)
        if not _include_run_handoff_relative_path(
            relative, include_results=include_results
        ):
            continue
        _append_manifest_entry(
            files,
            entry_root,
            path,
            role=_run_handoff_artifact_role(relative),
            name=relative.as_posix(),
            include_hashes=include_hashes,
            point_slug=point_slug,
        )
    return files


def _include_run_handoff_relative_path(
    relative_path: Path,
    *,
    include_results: bool,
) -> bool:
    """Return whether a run-relative file belongs in a handoff archive."""
    parts = relative_path.parts
    if not parts:
        return True
    return include_results or not (
        parts[0] == "logs" or (len(parts) > 1 and parts[:2] == ("results", "palace"))
    )


def _run_handoff_artifact_role(relative_path: Path) -> str:
    """Classify one archived file without deciding archive membership."""
    parts = relative_path.parts
    if parts[:2] == ("results", "palace"):
        return "result_artifact"
    if parts and parts[0] == "metadata":
        return "metadata_artifact"
    if parts and parts[0] == "geometry":
        return "geometry_artifact"
    if relative_path.name == "config.json" or relative_path.suffix == ".msh":
        return "core_artifact"
    return "handoff_artifact"


def _run_archive_manifest_entries(
    root: Path,
    summary: Any,
    *,
    include_results: bool,
    include_hashes: bool,
    point_slug: str | None = None,
) -> list[dict[str, Any]]:
    """Collect archive-manifest entries represented by a loaded run summary."""
    files: list[dict[str, Any]] = []
    for artifact in summary.artifacts.values():
        _append_status_manifest_entry(
            files,
            root,
            artifact,
            role="core_artifact",
            include_hashes=include_hashes,
            point_slug=point_slug,
        )
    if include_results:
        for artifact in summary.results.values():
            _append_status_manifest_entry(
                files,
                root,
                artifact,
                role="result_artifact",
                include_hashes=include_hashes,
                point_slug=point_slug,
            )
    handoff = _as_mapping(summary.handoff)
    _extend_handoff_reference_entries(
        files,
        root,
        handoff,
        include_hashes=include_hashes,
        point_slug=point_slug,
    )
    runtime = _as_mapping(summary.runtime)
    runtime_path = runtime.get("path")
    if runtime.get("present") is True and runtime_path is not None:
        _append_manifest_entry(
            files,
            root,
            Path(str(runtime_path)),
            role="runtime_metadata",
            name="palace_run_metadata.json",
            include_hashes=include_hashes,
            point_slug=point_slug,
        )
    return _deduplicate_manifest_entries(files)


def _extend_handoff_reference_entries(
    files: list[dict[str, Any]],
    root: Path,
    handoff: Mapping[str, Any],
    *,
    include_hashes: bool,
    role_prefix: str = "",
    point_slug: str | None = None,
) -> None:
    """Append handoff metadata and referenced script entries when present."""
    handoff_path_value = handoff.get("path")
    if handoff.get("present") is not True or handoff_path_value is None:
        return
    handoff_path = Path(str(handoff_path_value))
    _append_manifest_entry(
        files,
        root,
        handoff_path,
        role=f"{role_prefix}handoff_metadata",
        name=handoff_path.name,
        include_hashes=include_hashes,
        point_slug=point_slug,
    )
    script = _as_mapping(handoff.get("script"))
    if script.get("path") is not None:
        script_path = _resolve_sidecar_reference(handoff_path, script["path"])
        _append_manifest_entry(
            files,
            root,
            script_path,
            role=f"{role_prefix}handoff_script",
            name=Path(str(script["path"])).name,
            include_hashes=include_hashes,
            point_slug=point_slug,
        )


def _append_status_manifest_entry(
    files: list[dict[str, Any]],
    root: Path,
    artifact: Any,
    *,
    role: str,
    include_hashes: bool,
    point_slug: str | None = None,
) -> None:
    """Append one present run-summary artifact as a manifest entry."""
    if not artifact.present or artifact.path is None:
        return
    entry = _manifest_entry(
        root,
        artifact.path,
        role=role,
        name=artifact.name,
        include_hashes=include_hashes,
        point_slug=point_slug,
    )
    if include_hashes and artifact.sha256 is not None:
        entry["sha256"] = artifact.sha256
    files.append(entry)


def _append_manifest_entry(
    files: list[dict[str, Any]],
    root: Path,
    path: Path,
    *,
    role: str,
    name: str,
    include_hashes: bool,
    point_slug: str | None = None,
) -> None:
    """Append a manifest entry only when its referenced file exists."""
    if path.is_file():
        files.append(
            _manifest_entry(
                root,
                path,
                role=role,
                name=name,
                include_hashes=include_hashes,
                point_slug=point_slug,
            )
        )


def _manifest_entry(
    root: Path,
    path: Path,
    *,
    role: str,
    name: str,
    include_hashes: bool,
    point_slug: str | None,
) -> dict[str, Any]:
    """Build one archive manifest entry for a file below its root."""
    resolved_path = path.resolve()
    root_path = root.resolve()
    try:
        relative_path = resolved_path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(
            "archive manifest entries must live under the run or sweep root"
        ) from exc
    entry: dict[str, Any] = {
        "role": role,
        "name": name,
        "path": relative_path,
        "bytes": int(path.stat().st_size),
    }
    if point_slug is not None:
        entry["point_slug"] = point_slug
    if include_hashes:
        entry["sha256"] = _sha256_file(path)
    return entry


def _resolve_handoff_archive_path(
    run_dir: Path,
    archive_path: str | Path | None,
) -> Path:
    """Resolve an archive path and reject paths inside the run directory."""
    path = (
        _default_palace_handoff_archive_path(run_dir)
        if archive_path is None
        else Path(archive_path)
    )
    if not path.is_absolute() and archive_path is not None:
        path = run_dir.parent / path
    resolved_path = path.resolve()
    resolved_run_dir = run_dir.resolve()
    try:
        resolved_path.relative_to(resolved_run_dir)
    except ValueError:
        return path
    raise ValueError("archive_path must be outside the Palace run directory")


def _archive_path_reference(run_dir: Path, archive_path: Path) -> str:
    """Return an archive path expressed relative to its run directory."""
    return Path(
        os.path.relpath(archive_path.resolve(), start=run_dir.resolve())
    ).as_posix()


def _filter_run_handoff_tarinfo(
    info: tarfile.TarInfo,
    archive_root_name: str,
    *,
    include_results: bool,
) -> tarfile.TarInfo | None:
    """Exclude result and log members when a compact archive is requested."""
    if info.isdir():
        return info
    parts = Path(info.name).parts
    relative_parts = parts[1:] if parts and parts[0] == archive_root_name else parts
    return (
        info
        if _include_run_handoff_relative_path(
            Path(*relative_parts), include_results=include_results
        )
        else None
    )


def _archive_manifest_payload(
    *,
    source_kind: Literal["run", "sweep"],
    files: Sequence[Mapping[str, Any]],
    include_results: bool,
    archive_path: str | Path | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the schema-v1 archive-manifest payload."""
    archive = {}
    if archive_path is not None:
        archive["path"] = str(archive_path)
    return {
        "schema_version": 1,
        "kind": "palace_handoff_archive_manifest",
        "source_kind": source_kind,
        "status": "manifested",
        "root": ".",
        "include_results": include_results,
        "archive": archive,
        "file_count": len(files),
        "total_bytes": sum(int(row.get("bytes", 0) or 0) for row in files),
        "files": list(files),
        "metadata": _json_ready(dict(metadata or {})),
    }


def _deduplicate_manifest_entries(
    files: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return manifest rows deduplicated by relative path in first-seen order."""
    deduplicated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in files:
        path = str(row.get("path"))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        deduplicated.append(dict(row))
    return deduplicated


def _resolve_sidecar_reference(sidecar_path: Path, value: Any) -> Path:
    """Resolve a sidecar-relative artifact reference to a filesystem path."""
    path = Path(str(value))
    if path.is_absolute():
        return path
    reference_root = (
        sidecar_path.parent.parent
        if sidecar_path.parent.name == "metadata"
        else sidecar_path.parent
    )
    return reference_root / path


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON file as a mapping, returning empty for non-mappings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _relative_path(root: Path, path: Path) -> str:
    """Return a resolved relative path when possible, otherwise the input path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a mapping as stable indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _optional_dict(value: Any) -> dict[str, Any] | None:
    """Return a nonempty mapping copy or ``None``."""
    data = _as_mapping(value)
    return data or None
