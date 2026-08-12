"""Resolve-layer metadata models for one Palace run.

These models describe artifact presence and sidecar summaries discovered from a
canonical Palace run folder. They are resolve metadata, not problem report data:
source checks, missing artifacts, config facts, and handoff/runtime/resource
dictionaries stay here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PalaceArtifactStatus:
    """Presence, size, and optional checksum for one Palace artifact."""

    name: str
    path: Path | None
    present: bool
    bytes: int = 0
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly artifact row."""
        return {
            "name": self.name,
            "path": None if self.path is None else str(self.path),
            "present": self.present,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PalaceRunSummary:
    """Resolved metadata for one Palace run directory or result mapping."""

    problem_type: str | None
    artifacts: dict[str, PalaceArtifactStatus]
    results: dict[str, PalaceArtifactStatus]
    config: dict[str, Any]
    mesh_manifest: dict[str, Any]
    index_map: dict[str, Any]
    material_resolution: dict[str, Any]
    handoff: dict[str, Any]
    runtime: dict[str, Any]
    resource: dict[str, Any]

    @property
    def missing_artifacts(self) -> tuple[str, ...]:
        """Core handoff artifacts that were expected but absent."""
        return tuple(
            name for name, artifact in self.artifacts.items() if not artifact.present
        )

    @property
    def result_names(self) -> tuple[str, ...]:
        """Present Palace result artifact names."""
        return tuple(
            name for name, artifact in self.results.items() if artifact.present
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly run summary."""
        return {
            "problem_type": self.problem_type,
            "artifacts": {
                name: artifact.to_dict() for name, artifact in self.artifacts.items()
            },
            "results": {
                name: artifact.to_dict() for name, artifact in self.results.items()
            },
            "config": dict(self.config),
            "mesh_manifest": dict(self.mesh_manifest),
            "index_map": dict(self.index_map),
            "material_resolution": dict(self.material_resolution),
            "handoff": dict(self.handoff),
            "runtime": dict(self.runtime),
            "resource": dict(self.resource),
            "missing_artifacts": list(self.missing_artifacts),
        }
