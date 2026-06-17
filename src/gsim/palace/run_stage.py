"""Run-stage models for Palace execution and handoff workflows.

Responsibility:
Owns lightweight handles returned by Palace run-stage APIs after a run folder
has been prepared, executed, or packaged.

Does not own:
Resolve/report loading, typed result data, Slurm rendering, cloud submission,
or local process execution.

Inputs:
Run-stage APIs provide canonical run-folder paths and optional launcher/archive
references.

Outputs:
Notebook-friendly handles that can be passed into Resolve by using
``handle.run_folder``.

Pipeline position:
``sim.write_config() / sim.generate_handoff_package()`` ->
``PalaceRunHandle`` -> ``resolve_palace_result(handle.run_folder)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class PalaceRunHandle:
    """Run-stage handle for a prepared, submitted, or completed Palace run.

    The handle is intentionally not a report bundle. It records execution and
    handoff artifacts only; callers enter the Resolve stage explicitly through
    ``resolve_palace_result(handle.run_folder, ...)``.
    """

    run_folder: Path
    kind: Literal["handoff", "slurm", "local", "cloud"]
    status: str
    problem_type: str | None = None
    profile_name: str | None = None
    script_path: Path | None = None
    archive_path: Path | None = None
    metadata_path: Path | None = None
    archive_manifest_path: Path | None = None
    cloud_job_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly run-stage handle summary."""
        return {
            "run_folder": self.run_folder.as_posix(),
            "kind": self.kind,
            "status": self.status,
            "problem_type": self.problem_type,
            "profile_name": self.profile_name,
            "script_path": None
            if self.script_path is None
            else self.script_path.as_posix(),
            "archive_path": None
            if self.archive_path is None
            else self.archive_path.as_posix(),
            "metadata_path": None
            if self.metadata_path is None
            else self.metadata_path.as_posix(),
            "archive_manifest_path": None
            if self.archive_manifest_path is None
            else self.archive_manifest_path.as_posix(),
            "cloud_job_id": self.cloud_job_id,
            "metadata": dict(self.metadata),
        }


__all__ = ["PalaceRunHandle"]
