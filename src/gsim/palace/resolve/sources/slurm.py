"""Parse Slurm job-accounting sidecars for Palace runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gsim.palace._shared import optional_int, sha256_file
from gsim.palace.resolve.sources.resource_common import parse_resource_memory_bytes

_SLURM_KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_:/]+)=(?P<value>\S+)")
_SLURM_TRES_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_:/]+)=(?P<value>[^,]+)")


def parse_slurm_scontrol_job(path: str | Path) -> dict[str, Any]:
    """Parse sanitized Slurm ``scontrol show job`` evidence.

    The returned payload intentionally excludes raw scheduler text, account,
    user, node, job-name, command, stdout/stderr, and working-directory fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    values = {
        match.group("key"): match.group("value")
        for match in _SLURM_KEY_VALUE_RE.finditer(raw_text)
    }
    tres = _parse_slurm_tres(values.get("TRES"))

    scheduler = {
        "kind": "slurm",
        "job_id": optional_int(values.get("JobId")),
        "job_state": values.get("JobState"),
        "partition": values.get("Partition"),
        "submit_time": values.get("SubmitTime"),
        "start_time": values.get("StartTime"),
        "end_time": values.get("EndTime"),
        "time_limit": values.get("TimeLimit"),
        "time_limit_seconds": _parse_slurm_duration_seconds(values.get("TimeLimit")),
        "run_time": values.get("RunTime"),
        "run_time_seconds": _parse_slurm_duration_seconds(values.get("RunTime")),
    }
    scheduler = {key: value for key, value in scheduler.items() if value is not None}

    allocation = {
        "nodes": optional_int(values.get("NumNodes") or tres.get("node")),
        "num_cpus": optional_int(values.get("NumCPUs") or tres.get("cpu")),
        "num_tasks": optional_int(values.get("NumTasks")),
        "cpus_per_task": optional_int(values.get("CPUs/Task")),
    }
    allocation["num_processes"] = allocation.get("num_tasks")
    allocation["num_threads"] = allocation.get("cpus_per_task")
    allocation["cores"] = allocation.get("num_cpus")
    requested_memory = tres.get("mem") or values.get("MinMemoryNode")
    if requested_memory is not None:
        allocation["requested_memory"] = requested_memory
        allocation["requested_memory_bytes"] = parse_resource_memory_bytes(
            requested_memory
        )
    allocation = {key: value for key, value in allocation.items() if value is not None}

    return {
        "schema_version": 1,
        "source": {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        },
        "scheduler": scheduler,
        "allocation": allocation,
    }


def _parse_slurm_tres(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    return {
        match.group("key"): match.group("value")
        for match in _SLURM_TRES_VALUE_RE.finditer(str(value))
    }


def _parse_slurm_duration_seconds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "UNLIMITED", "NOT_SET", "Unknown"}:
        return None
    day_part = 0
    if "-" in text:
        days, text = text.split("-", 1)
        day_part = optional_int(days) or 0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(part) for part in parts)
        else:
            return None
    except ValueError:
        return None
    return ((day_part * 24 + hours) * 60 + minutes) * 60 + seconds
