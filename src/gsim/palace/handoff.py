"""Dry-run Palace handoff helpers."""

from __future__ import annotations

import csv
import json
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
class PalaceSlurmSweepArraySpec:
    """Render-ready Slurm array specification for a Palace sweep folder."""

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
    """Files written by a dry-run Palace Slurm handoff."""

    script_path: Path
    metadata_path: Path
    messages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PalaceSlurmSweepHandoffResult:
    """Files written by a dry-run Palace Slurm sweep-array handoff."""

    script_path: Path
    metadata_path: Path
    points_csv_path: Path
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
            _require_file(sweep_root / row["config_path"], "Palace config")
            _require_file(sweep_root / row["mesh_path"], "Palace mesh")

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
        messages=(f"Wrote Palace Slurm sweep handoff: {output_script_path}",),
    )


def _render_sweep_array_sbatch(
    spec: PalaceSlurmSweepArraySpec,
    point_rows: Sequence[Mapping[str, str | int]],
    max_parallel: int,
) -> str:
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
    lines.extend(spec.setup_commands)
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


def _sweep_point_specs(payload: Any) -> list[Mapping[str, Any]]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "array_index",
                "point_slug",
                "run_dir",
                "config_path",
                "mesh_path",
                "log_dir",
                "result_dir",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _normalize_sweep_path(sweep_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return path.resolve().relative_to(sweep_root.resolve()).as_posix()
    return path.as_posix()


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


def _validate_csv_field(label: str, value: str) -> None:
    if "," in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must not contain CSV separators")


def _validate_positive_int(label: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{label} must be positive")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
