from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from gsim.palace import (
    PalaceSlurmProfileSpec,
    PalaceSlurmResourceSpec,
    PalaceSlurmSbatchSpec,
    PalaceSlurmSweepArraySpec,
    PalaceSweepPointSpec,
    load_palace_run_summary,
    load_palace_sweep_summary,
    resolve_palace_slurm_profile,
    write_palace_run_handoff_archive_manifest,
    write_palace_slurm_sbatch_handoff,
    write_palace_slurm_sweep_array_handoff,
    write_palace_sweep_handoff_archive_manifest,
    write_palace_sweep_points,
)


def _write_minimal_palace_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"Problem": {"Type": "Electrostatic"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")


def test_resolve_palace_slurm_profile_accepts_mapping_and_overrides() -> None:
    resolution = resolve_palace_slurm_profile(
        {
            "public-slurm:cpu": {
                "source": "caller-supplied test fixture",
                "description": "Public CPU dry-run profile",
                "metadata": {"cluster": "public"},
                "resources": {
                    "account": "public_alloc",
                    "partition": "cpu",
                    "wall_time": "00:30:00",
                    "nodes": 1,
                    "ntasks_per_node": 2,
                    "cpus_per_task": 4,
                },
            }
        },
        "public-slurm:cpu",
        resource_overrides={"memory_mb": 64000, "wall_time": "01:00:00"},
    )

    assert resolution.name == "public-slurm:cpu"
    assert resolution.resources.num_processes == 2
    assert resolution.resources.num_threads == 4
    assert resolution.resources.memory_mb == 64000
    assert resolution.resources.wall_time == "01:00:00"
    assert resolution.profile == {
        "name": "public-slurm:cpu",
        "source": "caller-supplied test fixture",
        "description": "Public CPU dry-run profile",
        "metadata": {"cluster": "public"},
        "resource_overrides": {
            "memory_mb": 64000,
            "wall_time": "01:00:00",
        },
    }


def test_resolve_palace_slurm_profile_accepts_spec_objects() -> None:
    resolution = resolve_palace_slurm_profile(
        {
            "public-slurm:cpu": PalaceSlurmProfileSpec(
                name="public-slurm:cpu",
                resources=PalaceSlurmResourceSpec(
                    account="public_alloc",
                    partition="cpu",
                    wall_time="00:30:00",
                ),
            )
        },
        "public-slurm:cpu",
    )

    assert resolution.resources.account == "public_alloc"
    assert resolution.profile == {
        "name": "public-slurm:cpu",
        "source": "caller-supplied",
    }


def test_resolve_palace_slurm_profile_validates_inputs() -> None:
    profiles = {
        "public-slurm:cpu": {
            "resources": {
                "account": "public_alloc",
                "partition": "cpu",
                "wall_time": "00:30:00",
            }
        }
    }

    with pytest.raises(KeyError, match="Unknown Slurm profile"):
        resolve_palace_slurm_profile(profiles, "missing-profile")

    with pytest.raises(ValueError, match="Unknown Slurm resource override"):
        resolve_palace_slurm_profile(
            profiles,
            "public-slurm:cpu",
            resource_overrides={"queue": "cpu"},
        )

    with pytest.raises(ValueError, match="Unknown Slurm profile field"):
        resolve_palace_slurm_profile(
            {
                "public-slurm:cpu": {
                    "resources": {
                        "account": "public_alloc",
                        "partition": "cpu",
                        "wall_time": "00:30:00",
                    },
                    "private_site_default": True,
                }
            },
            "public-slurm:cpu",
        )


def test_write_palace_slurm_sbatch_handoff_round_trips_summary(
    tmp_path: Path,
) -> None:
    _write_minimal_palace_run(tmp_path)
    resolution = resolve_palace_slurm_profile(
        {
            "public-slurm:cpu": {
                "source": "caller-supplied test fixture",
                "resources": {
                    "account": "public_alloc",
                    "partition": "cpu",
                    "wall_time": "2-00:00:00",
                    "nodes": 2,
                    "ntasks_per_node": 4,
                    "cpus_per_task": 3,
                },
            }
        },
        "public-slurm:cpu",
        resource_overrides={"memory_mb": 64000},
    )
    spec = PalaceSlurmSbatchSpec(
        job_name="palace_public",
        resources=resolution.resources,
        setup_commands=("module load palace",),
        petsc_options=("-log_view",),
        srun_args=("--mpi=pmix",),
        mail_user="user@example.org",
    )

    result = write_palace_slurm_sbatch_handoff(
        tmp_path,
        spec,
        profile=resolution.profile,
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
    assert summary.handoff["profile"]["resource_overrides"] == {"memory_mb": 64000}
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


def test_write_palace_run_handoff_archive_manifest_round_trips_summary(
    tmp_path: Path,
) -> None:
    _write_minimal_palace_run(tmp_path)
    write_palace_slurm_sbatch_handoff(
        tmp_path,
        PalaceSlurmSbatchSpec(
            job_name="palace_manifest",
            resources=PalaceSlurmResourceSpec(
                account="public_alloc",
                partition="cpu",
                wall_time="00:30:00",
            ),
            petsc_options=(),
        ),
    )

    result = write_palace_run_handoff_archive_manifest(
        tmp_path,
        metadata={"workflow": "public-test"},
    )

    assert result.manifest_path.name == "palace_handoff_archive_manifest.json"
    assert result.metadata_path == tmp_path / "palace_handoff_metadata.json"
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source_kind"] == "run"
    assert payload["status"] == "manifested"
    assert payload["include_results"] is False
    assert payload["metadata"] == {"workflow": "public-test"}
    assert payload["file_count"] == result.file_count == 4
    assert payload["total_bytes"] == result.total_bytes
    assert {row["path"] for row in payload["files"]} == {
        "config.json",
        "palace.msh",
        "palace_handoff_metadata.json",
        "run_palace.sbatch",
    }
    assert {row["role"] for row in payload["files"]} == {
        "core_artifact",
        "handoff_metadata",
        "handoff_script",
    }
    assert all("sha256" in row for row in payload["files"])

    summary = load_palace_run_summary(tmp_path)
    assert summary.handoff["archive"] == {
        "manifest_path": "palace_handoff_archive_manifest.json"
    }
    assert summary.handoff["archive_present"] is False
    assert summary.handoff["archive_manifest_present"] is True
    assert "palace_handoff_archive_manifest.json" not in summary.results


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


def test_write_palace_slurm_sweep_array_handoff_round_trips_summary(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_minimal_palace_run(sweep_root / "points" / "gap_6um")
    _write_minimal_palace_run(sweep_root / "points" / "gap_8um")
    write_palace_sweep_points(
        sweep_root,
        [
            PalaceSweepPointSpec(
                point_slug="gap_6um",
                parameters={"gap_um": 6.0},
                run_dir="points/gap_6um",
            ),
            PalaceSweepPointSpec(
                point_slug="gap_8um",
                parameters={"gap_um": 8.0},
                run_dir="points/gap_8um",
            ),
        ],
        sweep_id="gap_sweep",
    )
    spec = PalaceSlurmSweepArraySpec(
        job_name="palace_gap_sweep",
        resources=PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="00:30:00",
            nodes=1,
            ntasks_per_node=2,
            cpus_per_task=2,
        ),
        max_parallel=8,
        petsc_options=(),
        srun_args=("--mpi=pmix",),
    )

    result = write_palace_slurm_sweep_array_handoff(
        sweep_root,
        spec,
        profile={"name": "public-slurm:sweep", "source": "caller-supplied"},
        metadata={"campaign": "public"},
    )

    script = result.script_path.read_text(encoding="utf-8")
    assert result.script_path.name == "run_sweep_array.sbatch"
    assert result.script_path.stat().st_mode & stat.S_IXUSR
    assert "#SBATCH --array=0-1%2" in script
    assert "#SBATCH --account=public_alloc" in script
    assert "POINTS_CSV=points.csv" in script
    assert 'srun --mpi=pmix "$PALACE_EXECUTABLE" "$CONFIG_PATH"' in script
    assert "sbatch" not in script

    csv_text = result.points_csv_path.read_text(encoding="utf-8")
    assert (
        "array_index,point_slug,run_dir,config_path,mesh_path,log_dir,result_dir"
        in csv_text
    )
    assert "0,gap_6um,points/gap_6um,points/gap_6um/config.json" in csv_text
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert payload["status"] == "scripted"
    assert payload["launcher"] == {
        "array": True,
        "dry_run": True,
        "kind": "slurm",
        "submission": "manual",
    }
    assert payload["resources"]["array"] == {"point_count": 2, "max_parallel": 2}
    assert payload["script"]["path"] == "run_sweep_array.sbatch"
    assert payload["command"] == {
        "argv": ["sbatch", "run_sweep_array.sbatch"],
        "redacted": True,
    }

    summary = load_palace_sweep_summary(sweep_root)

    assert summary.handoff["present"] is True
    assert summary.handoff["status"] == "scripted"
    assert summary.handoff["profile"]["name"] == "public-slurm:sweep"
    assert summary.handoff["script_present"] is True
    assert summary.handoff["archive_present"] is False
    assert summary.handoff["resources"]["array"]["point_count"] == 2
    assert summary.handoff["resources"]["array"]["max_parallel"] == 2
    assert summary.point_slugs == ("gap_6um", "gap_8um")
    assert summary.to_dict()["handoff"]["present"] is True


def test_write_palace_sweep_handoff_archive_manifest_round_trips_summary(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_minimal_palace_run(sweep_root / "points" / "gap_6um")
    _write_minimal_palace_run(sweep_root / "points" / "gap_8um")
    write_palace_sweep_points(
        sweep_root,
        [
            PalaceSweepPointSpec(
                point_slug="gap_6um",
                parameters={"gap_um": 6.0},
                run_dir="points/gap_6um",
            ),
            PalaceSweepPointSpec(
                point_slug="gap_8um",
                parameters={"gap_um": 8.0},
                run_dir="points/gap_8um",
            ),
        ],
        sweep_id="gap_sweep",
    )
    write_palace_slurm_sweep_array_handoff(
        sweep_root,
        PalaceSlurmSweepArraySpec(
            job_name="palace_manifest_sweep",
            resources=PalaceSlurmResourceSpec(
                account="public_alloc",
                partition="cpu",
                wall_time="00:30:00",
            ),
            petsc_options=(),
        ),
    )

    result = write_palace_sweep_handoff_archive_manifest(
        sweep_root,
        metadata={"workflow": "public-test"},
    )

    assert result.manifest_path.name == "palace_sweep_handoff_archive_manifest.json"
    assert result.metadata_path == sweep_root / "palace_sweep_handoff_metadata.json"
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["source_kind"] == "sweep"
    assert payload["status"] == "manifested"
    assert payload["metadata"] == {
        "include_point_files": True,
        "workflow": "public-test",
    }
    assert payload["file_count"] == result.file_count == 8
    by_path = {row["path"]: row for row in payload["files"]}
    assert set(by_path) == {
        "points.json",
        "points.csv",
        "palace_sweep_handoff_metadata.json",
        "run_sweep_array.sbatch",
        "points/gap_6um/config.json",
        "points/gap_6um/palace.msh",
        "points/gap_8um/config.json",
        "points/gap_8um/palace.msh",
    }
    assert by_path["points.csv"]["role"] == "sweep_points_table"
    assert by_path["run_sweep_array.sbatch"]["role"] == "sweep_handoff_script"
    assert by_path["points/gap_6um/config.json"]["point_slug"] == "gap_6um"
    assert by_path["points/gap_8um/palace.msh"]["point_slug"] == "gap_8um"

    summary = load_palace_sweep_summary(sweep_root)
    assert summary.handoff["archive"] == {
        "manifest_path": "palace_sweep_handoff_archive_manifest.json"
    }
    assert summary.handoff["archive_present"] is False
    assert summary.handoff["archive_manifest_present"] is True
    assert summary.to_dict()["handoff"]["archive_manifest_present"] is True


def test_write_palace_slurm_sweep_array_handoff_validates_point_artifacts(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    write_palace_sweep_points(
        sweep_root,
        [PalaceSweepPointSpec(point_slug="missing", run_dir="points/missing")],
    )
    spec = PalaceSlurmSweepArraySpec(
        job_name="palace_missing",
        resources=PalaceSlurmResourceSpec(
            account="public_alloc",
            partition="cpu",
            wall_time="00:30:00",
        ),
    )

    with pytest.raises(FileNotFoundError, match="Palace config"):
        write_palace_slurm_sweep_array_handoff(sweep_root, spec)


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
