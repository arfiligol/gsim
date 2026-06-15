from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from gsim.palace import (
    PalaceSlurmResourceSpec,
    PalaceSlurmSbatchSpec,
    load_palace_run_summary,
    write_palace_slurm_sbatch_handoff,
)


def _write_minimal_palace_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"Problem": {"Type": "Electrostatic"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")


def test_write_palace_slurm_sbatch_handoff_round_trips_summary(
    tmp_path: Path,
) -> None:
    _write_minimal_palace_run(tmp_path)
    spec = PalaceSlurmSbatchSpec(
        job_name="palace_public",
        resources=PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="2-00:00:00",
            nodes=2,
            ntasks_per_node=4,
            cpus_per_task=3,
            memory_mb=64000,
        ),
        setup_commands=("module load palace",),
        petsc_options=("-log_view",),
        srun_args=("--mpi=pmix",),
        mail_user="user@example.org",
    )

    result = write_palace_slurm_sbatch_handoff(
        tmp_path,
        spec,
        profile={"name": "public-slurm:cpu", "source": "caller-supplied"},
        metadata={"fixture": "public"},
    )

    script = result.script_path.read_text(encoding="utf-8")
    assert result.script_path.name == "run_palace.sbatch"
    assert result.script_path.stat().st_mode & stat.S_IXUSR
    assert "#SBATCH --account=public_alloc" in script
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --time=2-00:00:00" in script
    assert "#SBATCH --mem=64000M" in script
    assert "#SBATCH --mail-user=user@example.org" in script
    assert "module load palace" in script
    assert 'srun --mpi=pmix "$PALACE_EXECUTABLE" "$PALACE_CONFIG"' in script
    assert "sbatch" not in script

    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["status"] == "scripted"
    assert payload["launcher"] == {
        "dry_run": True,
        "kind": "slurm",
        "submission": "manual",
    }
    assert payload["resources"]["resolved"]["num_processes"] == 8
    assert payload["resources"]["resolved"]["num_threads"] == 3
    assert payload["script"]["path"] == "run_palace.sbatch"
    assert payload["command"] == {
        "argv": ["sbatch", "run_palace.sbatch"],
        "redacted": True,
    }

    summary = load_palace_run_summary(tmp_path)

    assert summary.handoff["present"] is True
    assert summary.handoff["status"] == "scripted"
    assert summary.handoff["profile"]["name"] == "public-slurm:cpu"
    assert summary.handoff["script_present"] is True
    assert summary.handoff["archive_present"] is False
    assert summary.handoff["resources"]["requested"]["nodes"] == 2
    assert summary.handoff["resources"]["resolved"]["ntasks_per_node"] == 4
    assert summary.handoff["metadata"]["fixture"] == "public"
    assert "run_palace.sbatch" not in summary.results


def test_write_palace_slurm_sbatch_handoff_supports_wrapper_flags(
    tmp_path: Path,
) -> None:
    _write_minimal_palace_run(tmp_path)
    spec = PalaceSlurmSbatchSpec(
        job_name="palace_wrapper",
        resources=PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="00:30:00",
        ),
        command_style="wrapper",
    )

    result = write_palace_slurm_sbatch_handoff(tmp_path, spec)

    script = result.script_path.read_text(encoding="utf-8")
    assert (
        'srun "$PALACE_EXECUTABLE" -np "$PALACE_NUM_PROCESSES" '
        '-nt "$PALACE_NUM_THREADS" "$PALACE_CONFIG"'
    ) in script


def test_write_palace_slurm_sbatch_handoff_validates_inputs(
    tmp_path: Path,
) -> None:
    spec = PalaceSlurmSbatchSpec(
        job_name="palace_missing",
        resources=PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="00:30:00",
        ),
    )

    with pytest.raises(FileNotFoundError, match="Palace config"):
        write_palace_slurm_sbatch_handoff(tmp_path, spec)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wall_time": "2 days"}, "wall_time"),
        ({"account": "bad account"}, "account"),
        ({"partition": "cpu;rm"}, "partition"),
    ],
)
def test_palace_slurm_resource_spec_rejects_unsafe_values(
    kwargs: dict[str, str],
    match: str,
) -> None:
    values = {
        "account": "public_alloc",
        "partition": "cpu",
        "wall_time": "00:30:00",
        **kwargs,
    }

    with pytest.raises(ValueError, match=match):
        PalaceSlurmResourceSpec(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"job_name": "bad/name"}, "job_name"),
        ({"srun_args": ("--mpi pmix",)}, "srun_args"),
        ({"petsc_options": ("-ksp_type gmres",)}, "petsc_options"),
        ({"setup_commands": ("module load palace\nrm -rf /",)}, "setup_commands"),
    ],
)
def test_palace_slurm_sbatch_spec_rejects_unsafe_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = {
        "job_name": "palace_public",
        "resources": PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="00:30:00",
        ),
        **kwargs,
    }

    with pytest.raises(ValueError, match=match):
        PalaceSlurmSbatchSpec(**values)
