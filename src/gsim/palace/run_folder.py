"""Canonical Palace run-folder contract.

This module owns only the filesystem shape shared by Palace execution,
handoff packaging, and Resolve source discovery. It does not write Palace
configs, parse solver outputs, build reports, or define visualization policy.

Run folder boundary:
``config.json`` and ``palace.msh`` are Palace execution inputs at the run root;
``metadata/`` stores gsim semantic sidecars; ``results/palace/`` stores raw
Palace solver outputs; ``logs/`` stores runtime logs; ``geometry/`` stores an
optional review snapshot such as ``design.gds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PalaceRunFolder:
    """Typed paths for a canonical Palace run folder."""

    root: Path

    @property
    def config_path(self) -> Path:
        """Return the Palace config path."""
        return self.root / "config.json"

    @property
    def mesh_path(self) -> Path:
        """Return the Palace mesh path."""
        return self.root / "palace.msh"

    @property
    def sbatch_path(self) -> Path:
        """Return the optional Slurm launcher path."""
        return self.root / "run_palace.sbatch"

    @property
    def geometry_dir(self) -> Path:
        """Return the review-snapshot geometry directory."""
        return self.root / "geometry"

    @property
    def design_gds_path(self) -> Path:
        """Return the optional run-local GDS snapshot path."""
        return self.geometry_dir / "design.gds"

    @property
    def metadata_dir(self) -> Path:
        """Return the semantic sidecar metadata directory."""
        return self.root / "metadata"

    @property
    def logs_dir(self) -> Path:
        """Return the runtime log directory."""
        return self.root / "logs"

    @property
    def results_dir(self) -> Path:
        """Return the raw result root directory."""
        return self.root / "results"

    @property
    def palace_results_dir(self) -> Path:
        """Return the raw Palace solver-output directory."""
        return self.results_dir / "palace"

    @property
    def mesh_manifest_path(self) -> Path:
        """Return the mesh manifest sidecar path."""
        return self.metadata_dir / "mesh_manifest.json"

    @property
    def index_map_path(self) -> Path:
        """Return the Palace postprocessing index-map sidecar path."""
        return self.metadata_dir / "palace_index_map.json"

    @property
    def material_resolution_path(self) -> Path:
        """Return the Palace material-resolution sidecar path."""
        return self.metadata_dir / "palace_material_resolution.json"

    @property
    def port_information_path(self) -> Path:
        """Return the port-information sidecar path."""
        return self.metadata_dir / "port_information.json"

    @property
    def handoff_metadata_path(self) -> Path:
        """Return the Palace handoff metadata sidecar path."""
        return self.metadata_dir / "palace_handoff_metadata.json"

    @property
    def handoff_archive_manifest_path(self) -> Path:
        """Return the handoff archive manifest sidecar path."""
        return self.metadata_dir / "palace_handoff_archive_manifest.json"

    @property
    def local_run_metadata_path(self) -> Path:
        """Return the local-run runtime metadata sidecar path."""
        return self.metadata_dir / "palace_run_metadata.json"

    @property
    def resource_record_path(self) -> Path:
        """Return the post-run resource record sidecar path."""
        return self.metadata_dir / "palace_resource_record.json"

    @property
    def directories(self) -> tuple[Path, ...]:
        """Return canonical directories that must exist before packaging."""
        return (
            self.root,
            self.geometry_dir,
            self.metadata_dir,
            self.logs_dir,
            self.results_dir,
            self.palace_results_dir,
        )


def palace_run_folder(root: str | Path) -> PalaceRunFolder:
    """Return typed paths for ``root`` without mutating the filesystem."""
    return PalaceRunFolder(root=Path(root))


def prepare_palace_run_folder(root: str | Path) -> PalaceRunFolder:
    """Create canonical Palace run-folder directories idempotently."""
    folder = palace_run_folder(root)
    for directory in folder.directories:
        directory.mkdir(parents=True, exist_ok=True)
    return folder


def relative_to_run_folder(path: Path, root: Path) -> str:
    """Return a stable JSON path relative to a run folder when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "PalaceRunFolder",
    "palace_run_folder",
    "prepare_palace_run_folder",
    "relative_to_run_folder",
]
