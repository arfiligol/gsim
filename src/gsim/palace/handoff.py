"""Dry-run Palace handoff helpers."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from gsim.palace.results import write_palace_handoff_metadata

_SBATCH_TOKEN_FORBIDDEN = set(" \t\r\n\"'`$;&|<>")
_SBATCH_JOB_NAME_FORBIDDEN = set(" \t\r\n\"'`$;&|<>/#")
_WALL_TIME_PATTERN = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")

DEFAULT_PALACE_PETSC_OPTIONS = (
    "-ksp_monitor",
    "-ksp_converged_reason",
    "-eps_converged_reason",
    "-log_view",
)


@dataclass(frozen=True)
class PalaceSlurmResourceSpec:
    """Resolved Slurm resources for a Palace handoff script."""

    account: str
    partition: str
    wall_time: str
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    memory_mb: int | None = None
    gres: str | None = None

    def __post_init__(self) -> None:
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
class PalaceSlurmSbatchSpec:
    """Render-ready Slurm sbatch specification for a Palace run directory."""

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
        lines.extend(self.setup_commands)
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
class PalaceSlurmHandoffResult:
    """Files written by a dry-run Palace Slurm handoff."""

    script_path: Path
    metadata_path: Path
    messages: tuple[str, ...] = field(default_factory=tuple)


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
    """
    run_dir = Path(source)
    if run_dir.suffix:
        raise ValueError("source must be a run directory")
    _validate_relative_path("script_path", str(script_path))
    output_script_path = run_dir / script_path
    if validate_inputs:
        _require_file(run_dir / spec.config_path, "Palace config")
        _require_file(run_dir / spec.mesh_path, "Palace mesh")

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
        messages=(f"Wrote Palace Slurm sbatch handoff: {output_script_path}",),
    )


def _render_petsc_options(options: Sequence[str]) -> list[str]:
    if not options:
        return []
    option_string = shlex.quote(" ".join(options))
    return [
        "",
        f"PALACE_PETSC_OPTIONS={option_string}",
        'if [[ -v PETSC_OPTIONS && -n "$PETSC_OPTIONS" ]]; then',
        '  export PETSC_OPTIONS="$PETSC_OPTIONS $PALACE_PETSC_OPTIONS"',
        "else",
        '  export PETSC_OPTIONS="$PALACE_PETSC_OPTIONS"',
        "fi",
        'echo "PETSC_OPTIONS=$PETSC_OPTIONS"',
    ]


def _validate_sbatch_token(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if any(char in _SBATCH_TOKEN_FORBIDDEN for char in value):
        raise ValueError(f"{label} must not contain whitespace or shell metacharacters")


def _validate_job_name(value: str) -> None:
    if not value:
        raise ValueError("job_name must not be empty")
    if any(char in _SBATCH_JOB_NAME_FORBIDDEN for char in value):
        raise ValueError("job_name must be a Slurm-safe token")


def _validate_shell_tokens(label: str, values: Sequence[str]) -> None:
    for value in values:
        _validate_sbatch_token(label, value)


def _validate_setup_commands(commands: Sequence[str]) -> None:
    for command in commands:
        if not command.strip():
            raise ValueError("setup_commands must not contain empty commands")
        if "\n" in command or "\r" in command:
            raise ValueError("setup_commands must be one shell command per entry")


def _validate_relative_path(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative to the run directory")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be single-line text")


def _validate_positive_int(label: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{label} must be positive")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
