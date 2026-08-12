from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from gsim.palace import EigenmodeSim
from gsim.palace.handoff import (
    PalaceSlurmLauncherSpec,
    PalaceSlurmProfileSpec,
    PalaceSlurmResourceSpec,
    PalaceSlurmSbatchSpec,
    PalaceSlurmSweepArraySpec,
    load_palace_slurm_profile_catalog,
    package_palace_run_handoff_archive,
    resolve_palace_slurm_profile,
    write_palace_run_handoff_archive_manifest,
    write_palace_slurm_sbatch_handoff,
    write_palace_slurm_sweep_array_handoff,
    write_palace_sweep_handoff_archive_manifest,
)
from gsim.palace.resolve import (
    PalaceSweepPointSpec,
    load_palace_run_summary,
    load_palace_sweep_summary,
)
from gsim.palace.resolve.sources.sidecars import write_palace_sweep_points


def _write_minimal_palace_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"Problem": {"Type": "Electrostatic"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")


def test_package_palace_run_handoff_archive_defaults_beside_run_folder(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_001"
    _write_minimal_palace_run(run_dir)

    result = package_palace_run_handoff_archive(run_dir)

    assert result.archive_path == (tmp_path / "run_001-palace.tar.gz")


def test_resolve_palace_slurm_profile_accepts_mapping_and_overrides() -> None:
    resolution = resolve_palace_slurm_profile(
        {
            "public-slurm:cpu": {
                "source": "caller-supplied test fixture",
                "description": "Public CPU dry-run profile",
                "launcher": {
                    "setup_commands": ["module load palace"],
                    "srun_args": ["--mpi=pmix"],
                    "petsc_options": [],
                },
                "solver": {"device": "CPU"},
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
    assert resolution.launcher.to_sbatch_kwargs() == {
        "setup_commands": ("module load palace",),
        "srun_args": ("--mpi=pmix",),
        "petsc_options": (),
    }
    assert resolution.solver == {"device": "CPU"}
    assert resolution.to_palace_config_hints() == {"Solver": {"Device": "CPU"}}
    sbatch_spec = resolution.to_sbatch_spec(job_name="public-slurm-cpu")
    assert isinstance(sbatch_spec, PalaceSlurmSbatchSpec)
    assert sbatch_spec.resources is resolution.resources
    assert sbatch_spec.setup_commands == ("module load palace",)
    assert sbatch_spec.srun_args == ("--mpi=pmix",)
    assert resolution.profile == {
        "name": "public-slurm:cpu",
        "source": "caller-supplied test fixture",
        "description": "Public CPU dry-run profile",
        "launcher": {
            "setup_commands": ["module load palace"],
            "srun_args": ["--mpi=pmix"],
            "petsc_options": [],
        },
        "solver": {"device": "CPU"},
        "metadata": {"cluster": "public"},
        "resource_overrides": {
            "memory_mb": 64000,
            "wall_time": "01:00:00",
        },
    }


def test_sim_write_slurm_sbatch_handoff_uses_resolved_profile(tmp_path: Path) -> None:
    """Simulation method keeps notebooks on the sim Run Stage pipeline."""
    _write_minimal_palace_run(tmp_path)
    sim = EigenmodeSim()
    sim.set_output_dir(tmp_path)
    profile = resolve_palace_slurm_profile(
        {
            "public-slurm:cpu": {
                "source": "caller-supplied test fixture",
                "launcher": {"setup_commands": ["module load palace"]},
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
    )

    handoff = sim.write_slurm_sbatch_handoff(
        profile,
        job_name="public-slurm-cpu",
        metadata={"workflow": "unit"},
    )

    script = handoff.script_path.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=public-slurm-cpu" in script
    assert "#SBATCH --ntasks-per-node=2" in script
    assert "module load palace" in script
    summary = load_palace_run_summary(tmp_path)
    assert summary.handoff["script"] == {"path": "run_palace.sbatch"}
    assert summary.handoff["profile"]["name"] == "public-slurm:cpu"


def test_slurm_profile_resolution_maps_solver_metadata_to_config_hints() -> None:
    resolution = resolve_palace_slurm_profile(
        {
            "public-slurm:gpu": {
                "resources": {
                    "account": "public_alloc",
                    "partition": "gpu",
                    "wall_time": "00:30:00",
                    "nodes": 1,
                },
                "solver": {"device": "GPU", "backend": "/gpu/cuda"},
            }
        },
        "public-slurm:gpu",
    )

    assert resolution.to_palace_config_hints() == {
        "Solver": {"Device": "GPU", "Backend": "/gpu/cuda"}
    }

    resolution = resolve_palace_slurm_profile(
        {
            "public-slurm:cpu": {
                "resources": {
                    "account": "public_alloc",
                    "partition": "cpu",
                    "wall_time": "00:30:00",
                    "nodes": 1,
                },
                "solver": {"device": None, "backend": None},
            }
        },
        "public-slurm:cpu",
    )
    assert resolution.to_palace_config_hints() == {}

    with pytest.raises(ValueError, match="Unknown Slurm profile solver field"):
        resolve_palace_slurm_profile(
            {
                "public-slurm:bad": {
                    "resources": {
                        "account": "public_alloc",
                        "partition": "cpu",
                        "wall_time": "00:30:00",
                        "nodes": 1,
                    },
                    "solver": {"runtime": "cuda"},
                }
            },
            "public-slurm:bad",
        )


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
                launcher=PalaceSlurmLauncherSpec(srun_args=("--mpi=pmix",)),
            )
        },
        "public-slurm:cpu",
    )

    assert resolution.resources.account == "public_alloc"
    assert resolution.launcher.to_sbatch_kwargs() == {"srun_args": ("--mpi=pmix",)}
    assert resolution.profile == {
        "name": "public-slurm:cpu",
        "source": "caller-supplied",
        "launcher": {"srun_args": ["--mpi=pmix"]},
    }


def test_load_palace_slurm_profile_catalog_accepts_envelope(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "profiles.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {"owner": "public-test"},
                "profiles": {
                    "public-slurm:cpu": {
                        "source": "caller-supplied test catalog",
                        "launcher": {
                            "command_style": "binary",
                            "srun_args": ["--mpi=pmix"],
                        },
                        "solver": {"device": "GPU", "backend": "cuda"},
                        "resources": {
                            "account": "public_alloc",
                            "partition": "cpu",
                            "wall_time": "00:30:00",
                        },
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    catalog = load_palace_slurm_profile_catalog(catalog_path)
    resolution = resolve_palace_slurm_profile(
        catalog,
        "public-slurm:cpu",
        resource_overrides={"ntasks_per_node": 4},
    )

    assert set(catalog) == {"public-slurm:cpu"}
    assert catalog["public-slurm:cpu"].resources.wall_time == "00:30:00"
    assert resolution.resources.num_processes == 4
    assert resolution.launcher.to_sbatch_kwargs() == {
        "command_style": "binary",
        "srun_args": ("--mpi=pmix",),
    }
    assert resolution.solver == {"device": "GPU", "backend": "cuda"}
    assert resolution.profile == {
        "name": "public-slurm:cpu",
        "source": "caller-supplied test catalog",
        "launcher": {
            "command_style": "binary",
            "srun_args": ["--mpi=pmix"],
        },
        "solver": {"device": "GPU", "backend": "cuda"},
        "resource_overrides": {"ntasks_per_node": 4},
    }


def test_load_palace_slurm_profile_catalog_accepts_direct_mapping(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "profiles.json"
    catalog_path.write_text(
        json.dumps(
            {
                "public-slurm:cpu": {
                    "resources": {
                        "account": "public_alloc",
                        "partition": "cpu",
                        "wall_time": "00:30:00",
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    catalog = load_palace_slurm_profile_catalog(catalog_path)

    assert catalog["public-slurm:cpu"].source == "caller-supplied"
    assert catalog["public-slurm:cpu"].resources.partition == "cpu"


def test_load_palace_slurm_profile_catalog_validates_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="JSON"):
        load_palace_slurm_profile_catalog(tmp_path / "profiles.yml")

    catalog_path = tmp_path / "profiles.json"
    catalog_path.write_text(
        json.dumps({"schema_version": 2, "profiles": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_palace_slurm_profile_catalog(catalog_path)

    catalog_path.write_text(
        json.dumps({"schema_version": 1, "profile": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown Slurm profile catalog field"):
        load_palace_slurm_profile_catalog(catalog_path)

    catalog_path.write_text(
        json.dumps(
            {
                "public-slurm:cpu": {
                    "resources": {
                        "account": "public_alloc",
                        "partition": "cpu",
                        "wall_time": "00:30:00",
                    },
                    "launcher": {"mpi": "pmix"},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown Slurm launcher field"):
        load_palace_slurm_profile_catalog(catalog_path)

    catalog_path.write_text(
        json.dumps(
            {
                "public-slurm:cpu": {
                    "resources": {
                        "account": "public_alloc",
                        "partition": "cpu",
                        "wall_time": "00:30:00",
                    },
                    "solver": {"device": "GPU", "runtime": "cuda"},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown Slurm profile solver field"):
        load_palace_slurm_profile_catalog(catalog_path)

    catalog_path.write_text(
        json.dumps({"schema_version": 1, "profiles": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="profiles"):
        load_palace_slurm_profile_catalog(catalog_path)


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
                "launcher": {
                    "setup_commands": ["module load palace"],
                    "srun_args": ["--mpi=pmix"],
                    "petsc_options": [],
                },
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
        mail_user="user@example.org",
        **resolution.launcher.to_sbatch_kwargs(),
    )

    result = write_palace_slurm_sbatch_handoff(
        tmp_path,
        spec,
        profile=resolution.profile,
        metadata={"fixture": "public"},
    )

    script = result.script_path.read_text(encoding="utf-8")
    assert result.script_path.name == "run_palace.sbatch"
    if os.name != "nt":
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
    assert summary.handoff["profile"]["launcher"] == {
        "setup_commands": ["module load palace"],
        "srun_args": ["--mpi=pmix"],
        "petsc_options": [],
    }
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
    assert '2>&1 | tee "logs/palace-${SLURM_JOB_ID:-manual}.log"' in script


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
    assert result.manifest_path.parent == tmp_path / "metadata"
    assert (
        result.metadata_path == tmp_path / "metadata" / "palace_handoff_metadata.json"
    )
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
        "metadata/palace_handoff_metadata.json",
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
        "manifest_path": "metadata/palace_handoff_archive_manifest.json"
    }
    assert summary.handoff["archive_present"] is False
    assert summary.handoff["archive_manifest_present"] is True
    assert "palace_handoff_archive_manifest.json" not in summary.results


def test_package_palace_run_handoff_archive_writes_tar_with_empty_result_dirs(
    tmp_path: Path,
) -> None:
    _write_minimal_palace_run(tmp_path)
    (tmp_path / "results" / "palace").mkdir(parents=True)
    (tmp_path / "results" / "palace" / "stale.csv").write_text("x\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "stale.log").write_text("log\n")

    result = package_palace_run_handoff_archive(tmp_path)

    assert result.archive_path is not None
    assert result.archive_path == tmp_path.parent / f"{tmp_path.name}-palace.tar.gz"
    assert result.archive_path.exists()
    assert result.manifest_path == (
        tmp_path / "metadata" / "palace_handoff_archive_manifest.json"
    )

    import tarfile

    with tarfile.open(result.archive_path, "r:gz") as archive:
        names = set(archive.getnames())

    assert f"{tmp_path.name}/config.json" in names
    assert f"{tmp_path.name}/palace.msh" in names
    assert f"{tmp_path.name}/results/palace" in names
    assert f"{tmp_path.name}/logs" in names
    assert f"{tmp_path.name}/results/palace/stale.csv" not in names
    assert f"{tmp_path.name}/logs/stale.log" not in names


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
    if os.name != "nt":
        assert result.script_path.stat().st_mode & stat.S_IXUSR
    assert "#SBATCH --array=0-1%2" in script
    assert "#SBATCH --account=public_alloc" in script
    assert "POINTS_CSV=points.csv" in script
    assert 'srun --mpi=pmix "$PALACE_EXECUTABLE" "$CONFIG_PATH"' in script
    assert '2>&1 | tee "$LOG_DIR/palace-${SLURM_ARRAY_TASK_ID:-manual}.log"' in script
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
        PalaceSlurmResourceSpec(**cast(Any, values))


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
        PalaceSlurmSbatchSpec(**cast(Any, values))
