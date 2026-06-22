"""Local Palace execution for the Run Stage.

This module builds and runs a local Palace command from a canonical run folder,
including optional caller-provided setup commands, process execution,
local-run metadata, and immediate result discovery.

Simulation model state, mesh/config generation, Slurm handoff archives, cloud
submission, Resolve/report construction, typed report display, and PDK-specific
runtime policy are owned by adjacent layers. Local execution consumes a folder
with ``config.json`` and ``palace.msh``, writes raw solver outputs under
``results/palace`` and metadata under ``metadata``, then leaves the folder ready
for ``resolve_palace_result(run_folder, ...)``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from gsim.palace.resolve.sources.resources import write_palace_resource_record_from_log
from gsim.palace.run_folder import (
    palace_run_folder,
    prepare_palace_run_folder,
    relative_to_run_folder,
)

logger = logging.getLogger(__name__)

PALACE_VERSION_LINE_PATTERN = re.compile(
    r"\bpalace\b.*?(?<![-_/@])v?(\d+\.\d+\.\d+)\b",
    re.IGNORECASE,
)
PALACE_PACKAGE_PATH_VERSION_PATTERN = re.compile(
    r"\bpalace-(\d+\.\d+\.\d+)-",
    re.IGNORECASE,
)


def redact_local_palace_command(
    cmd: list[str],
    *,
    use_apptainer: bool,
) -> list[str]:
    """Redact host-specific executable paths from local-run metadata."""
    redacted: list[str] = []
    for index, part in enumerate(cmd):
        if index == 0 or (use_apptainer and index == 2):
            redacted.append(Path(part).name)
        else:
            redacted.append(part)
    return redacted


def validate_local_setup_commands(
    setup_commands: Sequence[str] | None,
) -> tuple[str, ...]:
    """Normalize caller-provided shell setup commands for direct execution."""
    if setup_commands is None:
        return ()
    if isinstance(setup_commands, str):
        raise TypeError("setup_commands must be a sequence, not a single string")
    commands = tuple(str(command) for command in setup_commands)
    for command in commands:
        if not command.strip():
            raise ValueError("setup_commands must not contain empty commands")
        if "\n" in command or "\r" in command:
            raise ValueError("setup_commands must be one shell command per entry")
    return commands


def build_local_setup_shell_command(
    setup_commands: Sequence[str],
    palace_cmd: Sequence[str],
) -> list[str]:
    """Wrap a Palace command in one shell session with setup commands."""
    executable = palace_cmd[0]
    script_lines = ["set -eo pipefail", *setup_commands]
    if "/" not in executable:
        script_lines.append(f"command -v {shlex.quote(executable)} >/dev/null")
    script_lines.append(f"exec {shlex.join(palace_cmd)}")
    return ["/bin/bash", "-lc", "\n".join(script_lines)]


def parse_palace_runtime_version(output: str) -> str | None:
    """Extract a Palace semantic version from executable output.

    The local runner often invokes Palace through wrappers, MPI launchers, or
    Spack setup commands. Those layers can print dependency versions such as
    ``Open MPI 5.0.8`` before Palace itself prints anything. Palace's own
    ``--version`` output may report only a commit hash, so Spack package paths
    like ``palace-0.16.0-...`` are the fallback source for config-version audit.
    """
    for line in output.splitlines():
        match = PALACE_VERSION_LINE_PATTERN.search(line)
        if match is not None:
            return match.group(1)
    package_match = PALACE_PACKAGE_PATH_VERSION_PATTERN.search(output)
    if package_match is not None:
        return package_match.group(1)
    return None


def detect_palace_runtime_version(
    version_cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    setup_commands: Sequence[str],
) -> tuple[str | None, str]:
    """Run a best-effort Palace version command and return version plus output."""
    run_cmd = (
        build_local_setup_shell_command(setup_commands, version_cmd)
        if setup_commands
        else list(version_cmd)
    )
    try:
        result = subprocess.run(  # noqa: S603
            run_cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=dict(env),
            timeout=30,
        )
    except Exception as exc:
        return None, str(exc)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    runtime_version = parse_palace_runtime_version(output)
    if runtime_version is None:
        runtime_version = parse_palace_runtime_version(shlex.join(version_cmd))
    return runtime_version, output.strip()


def write_local_run_metadata(
    *,
    output_dir: Path,
    postpro_dir: Path,
    cmd: list[str],
    elapsed_seconds: float,
    returncode: int,
    use_apptainer: bool,
    executable_mode: Literal["wrapper", "binary"],
    executable_path: Path | None,
    sif_path: Path | None,
    num_processes: int,
    num_threads: int | None,
    serial: bool,
    omp_num_threads: str | None,
    files: dict[str, Path],
    target_palace_version: str | None,
    runtime_palace_version: str | None,
    runtime_version_output: str | None,
    runtime_version_check: str,
    palace_log_path: Path | None = None,
    resource_record_path: Path | None = None,
) -> Path:
    """Write ``metadata/palace_run_metadata.json`` for a completed local run."""
    launcher: dict[str, object]
    if use_apptainer:
        launcher = {
            "kind": "apptainer",
            "palace_sif_configured": sif_path is not None,
            "palace_sif_name": None if sif_path is None else sif_path.name,
        }
    else:
        launcher = {
            "kind": "executable",
            "executable_mode": executable_mode,
            "serial": serial,
            "palace_executable_configured": executable_path is not None,
            "palace_executable_name": (
                None if executable_path is None else executable_path.name
            ),
        }

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "completed",
        "return_code": returncode,
        "elapsed_seconds": elapsed_seconds,
        "launcher": launcher,
        "resources": {
            "num_processes": num_processes,
            "num_threads": num_threads,
            "omp_num_threads": omp_num_threads,
        },
        "command": {
            "argv": redact_local_palace_command(
                cmd,
                use_apptainer=use_apptainer,
            )
        },
        "paths": {
            "config": "config.json",
            "mesh": "palace.msh",
            "postprocessing_output": relative_to_run_folder(
                postpro_dir,
                output_dir,
            ),
        },
        "outputs": {
            name: {
                "path": relative_to_run_folder(path, output_dir),
                "bytes": int(path.stat().st_size),
            }
            for name, path in sorted(files.items())
        },
    }
    if target_palace_version is not None or runtime_palace_version is not None:
        metadata["palace_version"] = {
            "target": target_palace_version,
            "runtime": runtime_palace_version,
            "check": runtime_version_check,
        }
        if runtime_version_output:
            metadata["palace_version"]["output"] = runtime_version_output
    if palace_log_path is not None:
        metadata["paths"]["palace_log"] = relative_to_run_folder(
            palace_log_path,
            output_dir,
        )
    if resource_record_path is not None:
        metadata["paths"]["resource_record"] = relative_to_run_folder(
            resource_record_path,
            output_dir,
        )
    metadata_path = palace_run_folder(output_dir).local_run_metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def run_palace_local(
    output_dir: str | Path,
    *,
    palace_sif_path: str | Path | None = None,
    palace_executable: str | Path | None = None,
    use_apptainer: bool = True,
    executable_mode: Literal["wrapper", "binary"] = "wrapper",
    num_processes: int | None = None,
    num_threads: int | None = None,
    serial: bool = False,
    setup_commands: Sequence[str] | None = None,
    verbose: bool = True,
    prepare_run_folder: bool = True,
    palace_version: str | None = None,
    check_runtime_version: bool = True,
) -> dict[str, Path]:
    """Run Palace locally against a prepared canonical run folder."""
    if executable_mode not in {"wrapper", "binary"}:
        raise ValueError("executable_mode must be 'wrapper' or 'binary'")

    run_folder = (
        prepare_palace_run_folder(output_dir)
        if prepare_run_folder
        else palace_run_folder(output_dir)
    )
    output_path = run_folder.root
    config_path = run_folder.config_path
    mesh_path = run_folder.mesh_path

    if num_processes is None:
        num_processes = (
            1
            if not use_apptainer and executable_mode == "binary"
            else os.cpu_count() or 1
        )
    run_env = os.environ.copy()
    normalized_setup_commands = validate_local_setup_commands(setup_commands)
    if normalized_setup_commands and use_apptainer:
        raise ValueError("setup_commands are supported only with use_apptainer=False")
    sif_path: Path | None = None
    resolved_exe_path: Path | None = None

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Call write_config() first."
        )

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}. Call mesh() first.")

    if use_apptainer:
        if palace_sif_path is None:
            palace_sif_path = os.environ.get("PALACE_SIF")
            if palace_sif_path is None:
                raise ValueError(
                    "Palace SIF path not specified. Either set PALACE_SIF "
                    "environment variable or pass palace_sif_path parameter."
                )
            if verbose:
                logger.info("Using PALACE_SIF from environment: %s", palace_sif_path)

        sif_path = Path(palace_sif_path).expanduser().resolve()

        if not sif_path.exists():
            raise FileNotFoundError(
                f"Palace SIF file not found: {sif_path}. "
                "Install Palace via Apptainer or provide correct path."
            )

        if shutil.which("apptainer") is None:
            raise RuntimeError(
                "Apptainer not found. Install Apptainer to run local simulations "
                "with use_apptainer=True."
            )

        cmd = [
            "apptainer",
            "run",
            str(sif_path),
            "-np",
            str(num_processes),
        ]
        version_cmd = ["apptainer", "run", str(sif_path), "--version"]

    else:
        if palace_executable is None:
            palace_executable = os.environ.get("PALACE_EXECUTABLE", "palace")
            if verbose:
                logger.info(
                    "Using Palace executable from environment/default: %s",
                    palace_executable,
                )

        exe_path = Path(palace_executable).expanduser()

        if normalized_setup_commands:
            resolved_exe_path = exe_path
        elif not exe_path.exists():
            resolved = shutil.which(str(exe_path))
            if resolved is None:
                raise FileNotFoundError(
                    f"Palace executable not found: {exe_path}. "
                    "Install Palace directly or provide correct path via "
                    "palace_executable parameter."
                )
            resolved_exe_path = Path(resolved)
        else:
            resolved_exe_path = exe_path

        if executable_mode == "binary":
            if num_processes != 1:
                raise ValueError(
                    "executable_mode='binary' runs a single-process Palace "
                    "binary; use executable_mode='wrapper' for -np support."
                )
            if num_threads is not None:
                run_env["OMP_NUM_THREADS"] = str(num_threads)
            cmd = [str(resolved_exe_path)]
        else:
            cmd = [str(resolved_exe_path)]
            if serial:
                cmd.append("-serial")
            cmd.extend(["-np", str(num_processes)])
        version_cmd = [str(resolved_exe_path), "--version"]

    if num_threads is not None and not (
        not use_apptainer and executable_mode == "binary"
    ):
        cmd.extend(["-nt", str(num_threads)])
    runtime_version, runtime_version_output = detect_palace_runtime_version(
        version_cmd,
        cwd=output_path,
        env=run_env,
        setup_commands=normalized_setup_commands,
    )
    runtime_version_check = "not_requested"
    if palace_version is not None:
        if runtime_version is None:
            runtime_version_check = "unknown"
            if verbose:
                logger.warning(
                    "Could not detect Palace runtime version before local run."
                )
        elif runtime_version != palace_version:
            runtime_version_check = "mismatch"
            message = (
                "Palace runtime version "
                f"{runtime_version!r} does not match target config version "
                f"{palace_version!r}."
            )
            if check_runtime_version:
                raise RuntimeError(message)
            logger.warning(message)
        else:
            runtime_version_check = "matched"
    cmd.extend(["config.json"])
    run_cmd = (
        build_local_setup_shell_command(normalized_setup_commands, cmd)
        if normalized_setup_commands
        else cmd
    )

    def emit_info(msg: str, *args: object) -> None:
        if logger.isEnabledFor(logging.INFO):
            logger.info(msg, *args)
        else:
            logger.warning(msg, *args)

    def emit_warning(msg: str, *args: object) -> None:
        logger.warning(msg, *args)

    if verbose:
        if use_apptainer:
            emit_info("Running Palace simulation in %s via Apptainer", output_path)
        else:
            emit_info("Running Palace simulation in %s directly", output_path)
        emit_info("Command: %s", " ".join(cmd))
        if normalized_setup_commands:
            emit_info("Setup commands: %d", len(normalized_setup_commands))
        emit_info("Processes: %d", num_processes)

    started = time.perf_counter()
    returncode: int | None = None
    palace_log_path = run_folder.logs_dir / "palace-local.log"
    palace_log_path.parent.mkdir(parents=True, exist_ok=True)
    palace_log_path.write_text("", encoding="utf-8")
    try:
        if verbose:
            streamed_lines: list[str] = []
            with palace_log_path.open("w", encoding="utf-8") as log_stream:
                process_context = subprocess.Popen(  # noqa: S603
                    run_cmd,
                    cwd=output_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=run_env,
                )
                with process_context as process:
                    if process.stdout is not None:
                        for line in process.stdout:
                            log_stream.write(line)
                            log_stream.flush()
                            line = line.rstrip("\n")
                            streamed_lines.append(line)
                            if line:
                                emit_info(line)
                    returncode = process.wait()

            if returncode != 0:
                tail = "\n".join(streamed_lines[-200:])
                error_msg = f"Palace simulation failed with return code {returncode}"
                if tail:
                    error_msg += f"\n\nOutput (tail):\n{tail}"
                raise RuntimeError(error_msg)
        else:
            result = subprocess.run(  # noqa: S603
                run_cmd,
                cwd=output_path,
                check=True,
                capture_output=True,
                text=True,
                env=run_env,
            )
            returncode = result.returncode
            if result.stdout:
                logger.debug(result.stdout)
            if result.stderr:
                emit_warning(result.stderr)
            log_output = "".join(
                part for part in (result.stdout, result.stderr) if part
            )
            if log_output:
                palace_log_path.write_text(log_output, encoding="utf-8")
    except FileNotFoundError as e:
        if use_apptainer:
            raise RuntimeError(
                "Apptainer not found. Install Apptainer to run local simulations "
                "with use_apptainer=True."
            ) from e
        raise RuntimeError(
            "Palace executable not found. Install Palace directly or provide "
            "correct path via palace_executable parameter, "
            "or set PALACE_EXECUTABLE environment variable."
        ) from e

    if verbose:
        emit_info("Simulation completed successfully")

    postpro_dir = run_folder.palace_results_dir

    if verbose:
        emit_info("Results saved to %s", postpro_dir)

    files = {
        file.name: file
        for file in postpro_dir.iterdir()
        if file.is_file() and not file.name.startswith(".")
    }
    elapsed_seconds = time.perf_counter() - started
    resource_record_path: Path | None = None
    if palace_log_path.stat().st_size > 0:
        allocation: dict[str, int] = {"num_processes": num_processes}
        if num_threads is not None:
            allocation["num_threads"] = num_threads
        resource_record_path = write_palace_resource_record_from_log(
            output_path,
            palace_log_path,
            launcher={"kind": "local"},
            allocation=allocation,
            runtime={"elapsed_seconds": elapsed_seconds},
            metadata={"source": "run_palace_local"},
        )
    write_local_run_metadata(
        output_dir=output_path,
        postpro_dir=postpro_dir,
        cmd=cmd,
        elapsed_seconds=elapsed_seconds,
        returncode=0 if returncode is None else returncode,
        use_apptainer=use_apptainer,
        executable_mode=executable_mode,
        executable_path=resolved_exe_path,
        sif_path=sif_path,
        num_processes=num_processes,
        num_threads=num_threads,
        serial=serial,
        omp_num_threads=run_env.get("OMP_NUM_THREADS"),
        files=files,
        target_palace_version=palace_version,
        runtime_palace_version=runtime_version,
        runtime_version_output=runtime_version_output,
        runtime_version_check=runtime_version_check,
        palace_log_path=palace_log_path,
        resource_record_path=resource_record_path,
    )

    return files


__all__ = ["run_palace_local"]
