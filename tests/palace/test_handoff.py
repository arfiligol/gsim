from __future__ import annotations

import json
import tarfile
from pathlib import Path

from gsim.palace.handoff import (
    PalaceSlurmResourceSpec,
    PalaceSlurmSbatchSpec,
    package_palace_run_handoff_archive,
    write_palace_slurm_sbatch_handoff,
)


def test_slurm_handoff_archive_contains_log_output_parents(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"Model": {"Mesh": "palace.msh"}}), encoding="utf-8"
    )
    (run_dir / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")
    spec = PalaceSlurmSbatchSpec(
        job_name="palace_handoff",
        resources=PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="00:30:00",
        ),
        stdout_path="logs/stdout/%x-%j.out",
        stderr_path="logs/stderr/%x-%j.err",
    )
    handoff = write_palace_slurm_sbatch_handoff(run_dir, spec)
    log_paths = [
        Path(line.split("=", 1)[1])
        for line in handoff.script_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("#SBATCH --output=", "#SBATCH --error="))
    ]
    package = package_palace_run_handoff_archive(
        run_dir, archive_path=tmp_path / "handoff.tar.gz"
    )

    assert package.archive_path is not None
    extract_dir = tmp_path / "extracted"
    with tarfile.open(package.archive_path) as archive:
        assert f"{run_dir.name}/logs" in archive.getnames()
        archive.extractall(extract_dir, filter="data")

    extracted_run = extract_dir / run_dir.name
    for log_path in log_paths:
        assert (extracted_run / log_path).parent.is_dir()
