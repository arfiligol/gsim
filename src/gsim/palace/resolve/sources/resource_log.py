"""Parse Palace solver resource logs into public-safe records.

This module owns Palace log syntax: AMR passes, timing tables, memory tables,
PETSc summary snippets, and solver/resource metadata. It does not inspect Slurm
sidecars or write the final resource-record JSON.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gsim.palace._shared import sha256_file
from gsim.palace.resolve.sources.resource_common import parse_resource_memory_bytes

_RESOURCE_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PALACE_VERSION_RE = re.compile(r"Git changeset ID:\s*(?P<version>\S+)")
_PALACE_MPI_RE = re.compile(
    r"Running with (?P<mpi>\d+) MPI processes, (?P<threads>\d+) OpenMP threads"
)
_PALACE_DEVICE_RE = re.compile(r"Device configuration:\s*(?P<value>.+)")
_PALACE_MEMORY_RE = re.compile(r"Memory configuration:\s*(?P<value>.+)")
_PALACE_LIBCEED_RE = re.compile(r"libCEED backend:\s*(?P<value>.+)")
_PALACE_COMPLETED_AMR_RE = re.compile(
    r"Completed (?P<count>\d+) iterations of adaptive mesh refinement \(AMR\)"
)
_PALACE_AMR_ITERATION_RE = re.compile(
    r"Adaptive mesh refinement \(AMR\) iteration (?P<iteration>\d+):"
)
_PALACE_INDICATOR_RE = re.compile(
    rf"Indicator norm = (?P<indicator>{_RESOURCE_NUMBER_RE}), "
    rf"global unknowns = (?P<unknowns>\d+)"
)
_PALACE_AMR_LIMITS_RE = re.compile(
    rf"Max\. iterations = (?P<max_iterations>\d+), "
    rf"tol\. = (?P<tolerance>{_RESOURCE_NUMBER_RE}), "
    rf"max\. size = (?P<max_size>\d+)"
)
_PALACE_MARKED_RE = re.compile(
    rf"Marked (?P<marked>\d+)/(?P<total>\d+) elements for refinement "
    rf"\((?P<error_percent>{_RESOURCE_NUMBER_RE})% of the error, "
    rf".* = (?P<theta>{_RESOURCE_NUMBER_RE})\)"
)
_PALACE_REFINEMENT_RE = re.compile(
    r"Conforming mesh refinement added (?P<added>\d+) elements "
    r"\(initial = (?P<initial>\d+), final = (?P<final>\d+)\)"
)
_PALACE_ESTIMATED_MEMORY_RE = re.compile(
    r"Estimated peak per-(?P<scope>rank|node) memory usage is:\s*"
    r"Min\. (?P<min>\S+), Max\. (?P<max>\S+), "
    r"Avg\. (?P<avg>\S+), Total (?P<total>\S+)"
)
_PALACE_PETSC_HEADER_RE = re.compile(
    r"(?P<binary>\S*palace\S*) on a\s+(?P<name>\S*)\s+named (?P<node>\S+) "
    r"with (?P<processes>\d+) processes, by (?P<user>\S+) on (?P<date>.+)"
)
_PALACE_PETSC_VERSION_RE = re.compile(r"Using PETSc Release Version (?P<version>[^,]+)")
_PALACE_PETSC_TIME_RE = re.compile(
    rf"Time \(sec\):\s+(?P<max>{_RESOURCE_NUMBER_RE})\s+"
    rf"(?P<max_min>{_RESOURCE_NUMBER_RE})\s+(?P<avg>{_RESOURCE_NUMBER_RE})"
)
_RESOURCE_TABLE_ROW_RE = re.compile(
    rf"^(?P<stage>[A-Za-z][A-Za-z ]*[A-Za-z])\s+"
    rf"(?P<a>{_RESOURCE_NUMBER_RE}[KMGTPE]?)\s+"
    rf"(?P<b>{_RESOURCE_NUMBER_RE}[KMGTPE]?)\s+"
    rf"(?P<c>{_RESOURCE_NUMBER_RE}[KMGTPE]?)$"
)
PALACE_AMR_PASS_COLUMNS = (
    "pass_index",
    "indicator_norm",
    "global_unknowns",
    "max_iterations",
    "tolerance",
    "max_size",
    "marked_elements",
    "total_elements",
    "marked_error_percent",
    "theta",
    "refined_added_elements",
    "initial_elements",
    "final_elements",
)
PALACE_STAGE_TIMING_COLUMNS = (
    "pass_index",
    "stage",
    "min_seconds",
    "max_seconds",
    "avg_seconds",
)
PALACE_STAGE_MEMORY_COLUMNS = (
    "pass_index",
    "stage",
    "per_node",
    "total",
    "total_hwm",
    "per_node_bytes",
    "total_bytes",
    "total_hwm_bytes",
)


def parse_palace_resource_log(log_path: str | Path) -> dict[str, Any]:
    """Parse public-safe Palace solver resource details from a log file.

    The parser keeps solver/resource facts, AMR rows, and timing/memory tables.
    It intentionally omits PETSc user, node, and executable path fields so the
    parsed payload can be committed or shared as a publication-safe sidecar.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    solver: dict[str, Any] = {}
    allocation: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    model_size: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    estimated_peak_memory: dict[str, Any] = {}
    petsc_summary: dict[str, Any] = {}
    amr_passes: list[dict[str, Any]] = []
    stage_timing: list[dict[str, Any]] = []
    stage_memory: list[dict[str, Any]] = []

    current_pass_index = 1
    current_amr_pass: dict[str, Any] | None = None
    current_table: str | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                current_table = None
                continue

            if match := re.search(
                r"Proceeding with solve/estimate iteration (\d+)",
                stripped,
            ):
                current_pass_index = int(match.group(1))
                continue

            if match := _PALACE_VERSION_RE.search(stripped):
                solver["palace_git_changeset"] = match.group("version")
                continue
            if match := _PALACE_MPI_RE.search(stripped):
                allocation["num_processes"] = int(match.group("mpi"))
                allocation["num_threads"] = int(match.group("threads"))
                continue
            if match := _PALACE_DEVICE_RE.search(stripped):
                solver["device_configuration"] = match.group("value")
                continue
            if match := _PALACE_MEMORY_RE.search(stripped):
                solver["memory_configuration"] = match.group("value")
                continue
            if match := _PALACE_LIBCEED_RE.search(stripped):
                solver["libceed_backend"] = match.group("value")
                continue

            if match := _PALACE_AMR_ITERATION_RE.search(stripped):
                if current_amr_pass:
                    amr_passes.append(_ordered_amr_pass_row(current_amr_pass))
                current_amr_pass = {
                    "pass_index": int(match.group("iteration")),
                }
                current_pass_index = int(match.group("iteration"))
                continue

            if match := _PALACE_COMPLETED_AMR_RE.search(stripped):
                model_size["completed_amr_iterations"] = int(match.group("count"))
                continue
            if match := _PALACE_INDICATOR_RE.search(stripped):
                indicator_norm = float(match.group("indicator"))
                global_unknowns = int(match.group("unknowns"))
                model_size["final_indicator_norm"] = indicator_norm
                model_size["global_unknowns"] = global_unknowns
                if current_amr_pass is not None:
                    current_amr_pass["indicator_norm"] = indicator_norm
                    current_amr_pass["global_unknowns"] = global_unknowns
                continue
            if match := _PALACE_AMR_LIMITS_RE.search(stripped):
                if current_amr_pass is not None:
                    current_amr_pass.update(
                        {
                            "max_iterations": int(match.group("max_iterations")),
                            "tolerance": float(match.group("tolerance")),
                            "max_size": int(match.group("max_size")),
                        }
                    )
                continue
            if match := _PALACE_MARKED_RE.search(stripped):
                if current_amr_pass is not None:
                    current_amr_pass.update(
                        {
                            "marked_elements": int(match.group("marked")),
                            "total_elements": int(match.group("total")),
                            "marked_error_percent": float(match.group("error_percent")),
                            "theta": float(match.group("theta")),
                        }
                    )
                continue
            if match := _PALACE_REFINEMENT_RE.search(stripped):
                if current_amr_pass is not None:
                    current_amr_pass.update(
                        {
                            "refined_added_elements": int(match.group("added")),
                            "initial_elements": int(match.group("initial")),
                            "final_elements": int(match.group("final")),
                        }
                    )
                    amr_passes.append(_ordered_amr_pass_row(current_amr_pass))
                    current_amr_pass = None
                continue

            if match := _PALACE_ESTIMATED_MEMORY_RE.search(stripped):
                scope = match.group("scope")
                estimated_peak_memory[scope] = {
                    key: match.group(key) for key in ("min", "max", "avg", "total")
                } | {
                    f"{key}_bytes": parse_resource_memory_bytes(match.group(key))
                    for key in ("min", "max", "avg", "total")
                }
                if scope == "node" and not any(
                    row["pass_index"] == current_pass_index and row["stage"] == "Total"
                    for row in stage_memory
                ):
                    total_bytes = parse_resource_memory_bytes(match.group("total"))
                    row = {
                        "pass_index": current_pass_index,
                        "stage": "Total",
                        "per_node": match.group("max"),
                        "total": match.group("total"),
                        "total_hwm": match.group("total"),
                        "per_node_bytes": parse_resource_memory_bytes(
                            match.group("max")
                        ),
                        "total_bytes": total_bytes,
                        "total_hwm_bytes": total_bytes,
                    }
                    stage_memory.append(row)
                    memory["peak_total_memory_bytes"] = row["total_bytes"]
                    memory["peak_total_hwm_bytes"] = row["total_hwm_bytes"]
                continue

            if "Elapsed Time Report (s)" in stripped:
                current_table = "timing"
                continue
            if stripped.startswith("Peak Memory"):
                current_table = "memory"
                continue

            if current_table and (match := _RESOURCE_TABLE_ROW_RE.match(stripped)):
                stage = match.group("stage")
                if current_table == "timing":
                    row = {
                        "pass_index": current_pass_index,
                        "stage": stage,
                        "min_seconds": float(match.group("a")),
                        "max_seconds": float(match.group("b")),
                        "avg_seconds": float(match.group("c")),
                    }
                    stage_timing.append(row)
                    if stage == "Total":
                        runtime["wall_time_seconds"] = row["max_seconds"]
                else:
                    row = {
                        "pass_index": current_pass_index,
                        "stage": stage,
                        "per_node": match.group("a"),
                        "total": match.group("b"),
                        "total_hwm": match.group("c"),
                        "per_node_bytes": parse_resource_memory_bytes(match.group("a")),
                        "total_bytes": parse_resource_memory_bytes(match.group("b")),
                        "total_hwm_bytes": parse_resource_memory_bytes(
                            match.group("c")
                        ),
                    }
                    stage_memory.append(row)
                    if stage == "Total":
                        memory["peak_total_memory_bytes"] = row["total_bytes"]
                        memory["peak_total_hwm_bytes"] = row["total_hwm_bytes"]
                continue

            if match := _PALACE_PETSC_HEADER_RE.search(stripped):
                petsc_summary["processes"] = int(match.group("processes"))
                continue
            if match := _PALACE_PETSC_VERSION_RE.search(stripped):
                solver["petsc_version"] = match.group("version").strip()
                continue
            if match := _PALACE_PETSC_TIME_RE.search(stripped):
                petsc_summary["time_seconds"] = {
                    "max": float(match.group("max")),
                    "max_min_ratio": float(match.group("max_min")),
                    "avg": float(match.group("avg")),
                }

    if current_amr_pass:
        amr_passes.append(_ordered_amr_pass_row(current_amr_pass))

    return {
        "schema_version": 1,
        "source": {
            "path": str(log_path),
            "sha256": sha256_file(log_path),
            "bytes": int(log_path.stat().st_size),
        },
        "solver": solver,
        "allocation": allocation,
        "runtime": runtime,
        "model_size": model_size,
        "memory": memory,
        "estimated_peak_memory": estimated_peak_memory,
        "petsc_summary": petsc_summary,
        "amr_passes": amr_passes,
        "stage_timing": stage_timing,
        "stage_memory": stage_memory,
    }


def _ordered_amr_pass_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one AMR pass row into the canonical column order."""
    return {column: row.get(column) for column in PALACE_AMR_PASS_COLUMNS}
