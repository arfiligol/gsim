"""Tests for Palace result loaders, report models, and runtime artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

import numpy as np
import pytest

from gsim.palace.handoff import write_palace_handoff_metadata
from gsim.palace.resolve import (
    PalaceResolvedResult,
    PalaceRunSummary,
    PalaceSweepPointSpec,
    PalaceSweepResourceIndexResult,
    PalaceSweepSummary,
    load_palace_run_summary,
    load_palace_sweep_summary,
)
from gsim.palace.resolve.assembly import (
    load_driven_report,
    load_eigenmode_report,
    load_electrostatic_report,
)
from gsim.palace.resolve.derived.loss import (
    summarize_domain_loss,
    summarize_loss_budget,
    summarize_loss_channel_budget,
    summarize_surface_loss,
)
from gsim.palace.resolve.derived.materials import (
    load_dielectric_interface_summary,
    load_domain_material_summary,
)
from gsim.palace.resolve.derived.participation import (
    load_domain_energy_summary,
    load_port_epr_summary,
    load_surface_q_summary,
    summarize_surface_q_by_interface,
)
from gsim.palace.resolve.derived.terminal_matrices import (
    load_terminal_matrix,
    load_terminal_matrix_history,
    summarize_terminal_matrix_history,
)
from gsim.palace.resolve.loaders.eigenmodes import (
    load_eigenmode_history,
    load_eigenmodes,
    summarize_eigenmode_history,
)
from gsim.palace.resolve.loaders.index_maps import load_postprocessing_index_map
from gsim.palace.resolve.loaders.indexed_csv import load_indexed_csv
from gsim.palace.resolve.sources.resource_log import parse_palace_resource_log
from gsim.palace.resolve.sources.resources import (
    write_palace_resource_record,
    write_palace_resource_record_from_log,
)
from gsim.palace.resolve.sources.sidecars import write_palace_sweep_points
from gsim.palace.resolve.sources.slurm import parse_slurm_scontrol_job
from gsim.palace.resolve.sweeps import write_palace_sweep_resource_index
from gsim.palace.results import (
    DrivenReport,
    EigenmodeReport,
    Eigenmodes,
    ElectrostaticReport,
    SimulationBenchmark,
    SParams,
)
from gsim.palace.results.driven import load_sparams
from gsim.palace.results.loss import DomainLoss, LossBudget, ReportLoss, SurfaceLoss


def test_palace_root_results_api_keeps_notebook_surface_narrow() -> None:
    import gsim.palace as palace
    import gsim.palace.resolve as resolve
    import gsim.palace.results as results

    assert palace.SParams is SParams
    assert palace.PalaceResolvedResult is PalaceResolvedResult
    assert palace.DrivenReport is DrivenReport
    assert palace.EigenmodeReport is EigenmodeReport
    assert palace.ElectrostaticReport is ElectrostaticReport
    assert palace.resolve_palace_result is resolve.resolve_palace_result
    assert not hasattr(palace, "load_dielectric_interface_summary")
    assert not hasattr(palace, "load_domain_material_summary")
    assert not hasattr(palace, "load_driven_report")
    assert not hasattr(palace, "load_eigenmode_report")
    assert not hasattr(palace, "load_electrostatic_report")
    assert not hasattr(palace, "load_fields")
    assert not hasattr(palace, "load_report_for_resolved_result")
    assert not hasattr(palace, "load_palace_run_summary")
    assert not hasattr(palace, "load_postprocessing_index_map")
    assert not hasattr(palace, "load_sparams")
    assert not hasattr(palace, "load_terminal_matrix")
    assert not hasattr(palace, "BasePalaceReport")
    assert not hasattr(resolve, "load_driven_report")
    assert not hasattr(resolve, "load_eigenmode_report")
    assert not hasattr(resolve, "load_electrostatic_report")
    assert resolve.load_palace_run_summary is load_palace_run_summary
    assert resolve.load_palace_sweep_summary is load_palace_sweep_summary
    assert not hasattr(resolve, "load_dielectric_interface_summary")
    assert not hasattr(resolve, "load_domain_material_summary")
    assert not hasattr(resolve, "load_fields")
    assert not hasattr(resolve, "load_indexed_csv")
    assert not hasattr(resolve, "load_postprocessing_index_map")
    assert not hasattr(resolve, "load_sparams")
    assert not hasattr(resolve, "load_terminal_matrix")
    assert not hasattr(resolve, "summarize_domain_loss")
    assert not hasattr(results, "resolve_palace_result")
    assert not hasattr(results, "load_fields")
    assert not hasattr(results, "PalaceRunSummary")
    assert not hasattr(results, "PalaceSweepPointSpec")
    assert not hasattr(results, "BasePalaceReport")

    detail_names = (
        "Eigenmodes",
        "IndexedCsv",
        "IndexedCsvColumn",
        "SParam",
        "TerminalMatrix",
        "load_domain_energy_summary",
        "load_eigenmode_history",
        "load_eigenmodes",
        "load_indexed_csv",
        "load_port_epr_summary",
        "load_surface_q_summary",
        "load_terminal_matrix_history",
        "summarize_domain_loss",
        "summarize_eigenmode_history",
        "summarize_loss_budget",
        "summarize_surface_loss",
        "summarize_surface_q_by_interface",
        "summarize_terminal_matrix_history",
    )
    result_owned_names = {
        "Eigenmodes",
        "IndexedCsv",
        "IndexedCsvColumn",
        "PostprocessingTable",
        "SParam",
        "SurfaceQ",
        "TerminalMatrix",
    }
    for name in detail_names:
        assert not hasattr(palace, name)
        if name in result_owned_names:
            assert hasattr(results, name)
            continue
        assert not hasattr(results, name)
        assert not hasattr(resolve, name)


SLURM_SCONTROL = """
JobId=12345 JobName=private_layout_run
   UserId=private-user(1000) GroupId=private-group(1000)
   Account=private_account JobState=COMPLETED
   SubmitTime=2026-05-21T18:16:44 StartTime=2026-05-21T18:24:47
   EndTime=2026-05-21T18:26:48
   Partition=public_cpu NodeList=private-node BatchHost=private-node
   NumNodes=1 NumCPUs=112 NumTasks=4 CPUs/Task=28 TimeLimit=00:10:00 RunTime=00:02:01
   TRES=cpu=112,mem=482496M,node=1,billing=112
   Command=/private/work/run_palace.sbatch
   WorkDir=/private/work
   StdOut=/private/work/slurm.out StdErr=/private/work/slurm.err
"""

PALACE_RESOURCE_LOG = (
    dedent(
        """
Git changeset ID: v0.16.1
Running with 4 MPI processes, 28 OpenMP threads
Device configuration: omp,cpu
Memory configuration: host-std
libCEED backend: /cpu/self/xsmm/blocked

Cumulative timing statistics:

Elapsed Time Report (s)           Min.        Max.        Avg.
==============================================================
Initialization                   1.000       1.100       1.050
Operator Construction            2.000       2.200       2.100
Disk IO                          0.400       0.500       0.450
--------------------------------------------------------------
Total                           58.573      58.580      58.578

Peak Memory                   Per-Node       Total   Total HWM
==============================================================
Initialization                   79.1M       79.1M       79.1M
Operator Construction             1.6G        1.6G        2.0G
Disk IO                         216.9M      216.9M        2.1G
--------------------------------------------------------------
Total                            10.8G       10.8G       10.8G
Estimated peak per-rank memory usage is: Min. 2.7G, Max. 2.7G, Avg. 2.7G, Total 10.9G
Estimated peak per-node memory usage is: Min. 10.9G, Max. 10.9G, Avg. 10.9G, Total 10.9G

Adaptive mesh refinement (AMR) iteration 1:
 Indicator norm = 3.158e-01, global unknowns = 887970
 Max. iterations = 15, tol. = 1.000e-02, max. size = 5000000
 Marked 12568/664696 elements for refinement (70.00% of the error, theta = 0.70)
 Conforming mesh refinement added 659265 elements (initial = 664696, final = 1323961)

Proceeding with solve/estimate iteration 2...

Elapsed Time Report (s)           Min.        Max.        Avg.
==============================================================
Initialization                   1.000       1.100       1.050
Operator Construction            3.000       3.200       3.100
Disk IO                          0.400       0.500       0.450
--------------------------------------------------------------
Total                          120.000     121.000     120.500

Peak Memory                   Per-Node       Total   Total HWM
==============================================================
Initialization                   79.1M       79.1M       79.1M
Operator Construction             2.6G        2.6G        3.0G
Disk IO                         216.9M      216.9M        3.1G
--------------------------------------------------------------
Total                            20.8G       20.8G       20.8G
Estimated peak per-rank memory usage is: Min. 5.2G, Max. 5.2G, Avg. 5.2G, Total 20.9G
Estimated peak per-node memory usage is: Min. 20.9G, Max. 20.9G, Avg. 20.9G, Total 20.9G

Completed 1 iterations of adaptive mesh refinement (AMR):
 Indicator norm = 1.522e-01, global unknowns = 10718029
 Max. iterations = 15, tol. = 1.000e-02, max. size = 5000000

"""
    )
    + "-" * 66
    + " PETSc Performance Summary: "
    + "-" * 66
    + "\n\n"
    + (
        "/opt/private-palace/bin/palace-x86_64.bin on a  named private-node "
        "with 4 processes, by private-user on Thu May 21 18:41:59 2026\n"
    )
    + """
Using 28 OpenMP threads
Using PETSc Release Version 3.24.3, unknown

                         Max       Max/Min     Avg       Total
Time (sec):           1.029e+03     1.000   1.029e+03
"""
)


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace output directory."""
    palace_dir = tmp_path / "results" / "palace"
    palace_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    port_info = {
        "ports": [
            {"portnumber": 1, "name": "o1", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "name": "o2", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 3, "name": "o3", "Z0": 50.0, "type": "lumped"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (metadata_dir / "port_information.json").write_text(json.dumps(port_info))

    csv_content = (
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.),"
        " |S[3][1]| (dB), arg(S[3][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0, -30.0, -120.0\n"
        "2.0, -18.0, -50.0, -2.5, -85.0, -28.0, -115.0\n"
    )
    (palace_dir / "port-S.csv").write_text(csv_content)

    return tmp_path


@pytest.fixture
def indexed_report_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace indexed-report output with an index map."""
    palace_dir = tmp_path / "results" / "palace"
    palace_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    index_map = {
        "schema_version": 1,
        "entries": [
            {
                "section": "Domains.Postprocessing.Energy",
                "index": 1,
                "entry_name": "substrate",
                "role": "dielectric_volume",
                "attributes": [10],
                "physical_names": ["D1_SUBSTRATE"],
                "dimension": 3,
                "metadata": {"material": "silicon"},
            },
            {
                "section": "Boundaries.Postprocessing.Dielectric",
                "index": 2,
                "entry_name": "ma_interface",
                "role": "boundary_surface",
                "attributes": [20],
                "physical_names": ["MA:D1_TOP_M1___D1_SUBSTRATE"],
                "dimension": 2,
                "Type": "MA",
                "preset_name": "public_ma",
                "preset_source": "test source",
            },
            {
                "section": "Boundaries.Postprocessing.SurfaceFlux",
                "index": 3,
                "entry_name": "readout_port_surface",
                "role": "port_surface",
                "attributes": [30],
                "physical_names": ["P1"],
                "dimension": 2,
                "Type": "Power",
            },
        ],
    }
    (metadata_dir / "palace_index_map.json").write_text(json.dumps(index_map))
    config = {
        "Domains": {
            "Materials": [
                {
                    "Attributes": [10],
                    "Name": "silicon",
                    "Permittivity": 11.45,
                    "LossTan": 1.0e-6,
                    "Conductivity": 2.0,
                },
                {
                    "Attributes": [99],
                    "Permittivity": 4.2,
                    "LossTan": 0.0,
                },
            ]
        },
        "Boundaries": {
            "Postprocessing": {
                "Dielectric": [
                    {
                        "Index": 2,
                        "Attributes": [20],
                        "Type": "MA",
                        "Thickness": 0.002,
                        "Permittivity": 10.0,
                        "LossTan": 0.0033,
                    },
                    {
                        "Index": 99,
                        "Attributes": [199],
                        "Type": "SA",
                        "Thickness": 0.003,
                        "Permittivity": 4.0,
                        "LossTan": 0.0017,
                    },
                ]
            }
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    material_resolution = {
        "schema_version": 1,
        "materials": [
            {
                "material_row_index": 1,
                "material_attribute": 10,
                "material_attributes": [10],
                "volume_name": "substrate",
                "stack_material_name": "Si",
                "matched_material_name": "Si",
                "evaluation_frequency_hz": 5.0e9,
                "evaluation_frequency_ghz": 5.0,
                "model_type": "constant",
                "model_source": "test PDK material overlay",
                "within_validity": True,
                "validity_note": None,
                "effective_material": {
                    "permittivity": 11.45,
                    "loss_tangent": 1.0e-6,
                    "conductivity": 2.0,
                },
                "palace_material": {
                    "Attributes": [10],
                    "Name": "silicon",
                    "Permittivity": 11.45,
                    "LossTan": 1.0e-6,
                    "Conductivity": 2.0,
                },
            },
            {
                "material_row_index": 2,
                "material_attribute": 99,
                "material_attributes": [99],
                "volume_name": "unmapped",
                "stack_material_name": "custom",
                "matched_material_name": None,
                "evaluation_frequency_hz": 5.0e9,
                "evaluation_frequency_ghz": 5.0,
                "model_type": None,
                "model_source": None,
                "within_validity": None,
                "validity_note": "material not found in gsim material database",
                "effective_material": {"permittivity": 4.2, "loss_tangent": 0.0},
                "palace_material": {
                    "Attributes": [99],
                    "Permittivity": 4.2,
                    "LossTan": 0.0,
                },
            },
        ],
        "interfaces": [
            {
                "interface_row_index": 1,
                "surface_index": 2,
                "surface_attributes": [20],
                "interface_type": "MA",
                "interface_material_name": "AlOx_native_generic",
                "matched_material_name": "AlOx_native_generic",
                "evaluation_frequency_hz": 5.0e9,
                "evaluation_frequency_ghz": 5.0,
                "model_type": "constant",
                "model_source": "test PDK interface material",
                "within_validity": True,
                "validity_note": None,
                "effective_material": {
                    "permittivity": 10.0,
                    "loss_tangent": 0.0033,
                },
                "palace_interface": {
                    "Index": 2,
                    "Attributes": [20],
                    "Type": "MA",
                    "Thickness": 0.002,
                    "Permittivity": 10.0,
                    "LossTan": 0.0033,
                },
            }
        ],
    }
    (tmp_path / "metadata" / "palace_material_resolution.json").write_text(
        json.dumps(material_resolution)
    )
    (palace_dir / "domain-E.csv").write_text(
        "m, E_elec[1] (J), p_elec[1], E_elec[99] (J)\n1, 2.0, 0.5, 0.0\n"
    )
    (palace_dir / "surface-Q.csv").write_text(
        "m, p_surf[2], Q_surf[2]\n1, 1.0e-7, 2.0e6\n"
    )
    (palace_dir / "port-EPR.csv").write_text("m, p[3]\n1, -2.5e-4\n")
    return tmp_path


def _write_sweep_point_artifacts(
    source: Path,
    run_dir: Path,
    *,
    result_dir: Path | None = None,
    mesh_name: str = "palace.msh",
) -> None:
    run_dir.mkdir(parents=True)
    result_dir = result_dir or run_dir / "results" / "palace"
    result_dir.mkdir(parents=True)
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    config = json.loads((source / "config.json").read_text())
    config["Problem"] = {"Type": "Eigenmode"}
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / mesh_name).write_text("$MeshFormat\n")
    (metadata_dir / "mesh_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "name": "substrate",
                        "role": "dielectric_volume",
                        "dimension": 3,
                        "physical_names": ["D1_SUBSTRATE"],
                    }
                ],
            }
        )
    )
    for name in ("palace_index_map.json", "palace_material_resolution.json"):
        shutil.copy(source / "metadata" / name, metadata_dir / name)
    (metadata_dir / "palace_run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "return_code": 0,
                "elapsed_seconds": 2.5,
                "launcher": {"kind": "executable"},
                "resources": {"num_processes": 1, "num_threads": 1},
                "outputs": {"domain-E.csv": {"bytes": 42}},
            }
        )
    )
    for csv_path in (source / "results" / "palace").glob("*.csv"):
        shutil.copy(csv_path, result_dir / csv_path.name)


def _single_point_report_sweep(
    sweep_root: Path,
    run_dir: Path,
    *,
    point_slug: str,
    problem_type: str,
) -> PalaceSweepSummary:
    sweep_root.mkdir(parents=True)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config["Problem"] = {"Type": problem_type}
    config_path.write_text(json.dumps(config))
    (sweep_root / "points.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sweep_id": f"{point_slug}_sweep",
                "points": [
                    {
                        "point_slug": point_slug,
                        "parameters": {"problem_type": problem_type},
                        "run_dir": str(run_dir),
                    }
                ],
            }
        )
    )
    return load_palace_sweep_summary(sweep_root, include_report_metrics=True)


@pytest.fixture
def driven_report_dir(indexed_report_dir: Path) -> Path:
    """Create a Palace driven report output with S-parameters and port EPR."""
    palace_dir = indexed_report_dir / "results" / "palace"
    port_info = {
        "ports": [
            {"portnumber": 1, "name": "readout", "Z0": 50.0, "type": "cpw"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (indexed_report_dir / "metadata" / "port_information.json").write_text(
        json.dumps(port_info)
    )
    (palace_dir / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.)\n"
        "5.0, -12.0, -33.0\n"
        "6.0, -6.0, -45.0\n"
    )
    return indexed_report_dir


@pytest.fixture
def eigenmode_report_dir(indexed_report_dir: Path) -> Path:
    """Create a Palace eigenmode report output with AMR and EPR tables."""
    palace_dir = indexed_report_dir / "results" / "palace"
    iteration01 = palace_dir / "iteration01"
    iteration01.mkdir()
    _write_eig_csv(
        iteration01 / "eig.csv",
        [
            [1, 6.0, 0.01, 100.0, 1.0e-7, 1.0e-4],
            [2, 8.0, 0.02, 200.0, 2.0e-7, 2.0e-4],
        ],
    )
    _write_eig_csv(
        palace_dir / "eig.csv",
        [
            [1, 6.3, 0.02, 110.0, 1.0e-8, 1.0e-5],
            [2, 8.4, 0.04, 210.0, 2.0e-8, 2.0e-5],
        ],
    )
    return indexed_report_dir


@pytest.fixture
def terminal_matrix_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace electrostatic matrix output with an index map."""
    palace_dir = tmp_path / "results" / "palace"
    palace_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    index_map = {
        "schema_version": 1,
        "entries": [
            {
                "section": "Boundaries.Terminal",
                "index": 1,
                "entry_name": "left_metal",
                "role": "pec_surface",
                "attributes": [11],
                "physical_names": ["D1_TOP_M1@left"],
                "dimension": 2,
                "terminal_name": "left",
            },
            {
                "section": "Boundaries.Terminal",
                "index": 2,
                "entry_name": "right_metal",
                "role": "pec_surface",
                "attributes": [12],
                "physical_names": ["D1_TOP_M1@right"],
                "dimension": 2,
                "terminal_name": "right",
            },
        ],
    }
    (metadata_dir / "palace_index_map.json").write_text(json.dumps(index_map))
    _write_terminal_matrix_csv(
        palace_dir / "terminal-C.csv",
        "C",
        [
            [1.0e-15, -2.0e-15],
            [-2.0e-15, 4.0e-15],
        ],
    )
    _write_terminal_matrix_csv(
        palace_dir / "terminal-Cm.csv",
        "Cm",
        [
            [0.0, 2.0e-15],
            [2.0e-15, 0.0],
        ],
    )
    _write_terminal_matrix_csv(
        palace_dir / "terminal-Cinv.csv",
        "Cinv",
        [
            [1.0e15, 2.0e15],
            [2.0e15, 4.0e15],
        ],
    )
    return tmp_path


@pytest.fixture
def electrostatic_report_dir(terminal_matrix_dir: Path) -> Path:
    """Create a Palace electrostatic matrix output with EPR report tables."""
    palace_dir = terminal_matrix_dir / "results" / "palace"
    index_map_path = terminal_matrix_dir / "metadata" / "palace_index_map.json"
    index_map = json.loads(index_map_path.read_text())
    index_map["entries"].extend(
        [
            {
                "section": "Domains.Postprocessing.Energy",
                "index": 1,
                "entry_name": "substrate",
                "role": "dielectric_volume",
                "attributes": [10],
                "physical_names": ["substrate"],
                "dimension": 3,
                "metadata": {"material": "silicon"},
            },
            {
                "section": "Boundaries.Postprocessing.Dielectric",
                "index": 2,
                "entry_name": "ma_interface",
                "role": "boundary_surface",
                "attributes": [20],
                "physical_names": ["MA:top_metal__substrate"],
                "dimension": 2,
                "Type": "MA",
            },
        ]
    )
    index_map_path.write_text(json.dumps(index_map))
    config = {
        "Domains": {
            "Materials": [
                {
                    "Attributes": [10],
                    "Name": "silicon",
                    "Permittivity": 11.45,
                    "LossTan": 1.0e-6,
                },
            ]
        },
        "Boundaries": {
            "Postprocessing": {
                "Dielectric": [
                    {
                        "Index": 2,
                        "Attributes": [20],
                        "Type": "MA",
                        "Thickness": 0.002,
                        "Permittivity": 10.0,
                        "LossTan": 0.0033,
                    },
                ]
            }
        },
    }
    (terminal_matrix_dir / "config.json").write_text(json.dumps(config))
    (palace_dir / "domain-E.csv").write_text(
        "i, E_elec[1] (J), p_elec[1]\n1, 2.0, 0.5\n2, 3.0, 0.25\n"
    )
    (palace_dir / "surface-Q.csv").write_text(
        "i, p_surf[2], Q_surf[2]\n1, 1.0e-7, 2.0e6\n2, 2.0e-7, 4.0e6\n"
    )
    return terminal_matrix_dir


@pytest.fixture
def eigenmode_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace eigenmode output directory."""
    palace_dir = tmp_path / "results" / "palace"
    palace_dir.mkdir(parents=True)
    (palace_dir / "eig.csv").write_text(
        "m, Re{f} (GHz), Im{f} (GHz), Q, Error (Bkwd.), Error (Abs.)\n"
        "1.00e+00, 6.1, 0.01, 300.0, 1.0e-7, 2.0e-4\n"
        "2.00e+00, 7.2, 0.02, 400.0, 2.0e-7, 3.0e-4\n"
    )
    return tmp_path


@pytest.fixture
def sim_dir_no_names(tmp_path: Path) -> Path:
    """Sim dir with port_information.json without name fields."""
    palace_dir = tmp_path / "results" / "palace"
    palace_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    port_info = {
        "ports": [
            {"portnumber": 1, "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "Z0": 50.0, "type": "cpw"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (metadata_dir / "port_information.json").write_text(json.dumps(port_info))

    csv_content = (
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0\n"
    )
    (palace_dir / "port-S.csv").write_text(csv_content)

    return tmp_path


class TestSParams:
    """Tests for the SParams result object."""

    def test_returns_sparams_object(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert isinstance(sp, SParams)

    def test_freq(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.freq[0] == pytest.approx(1.0)
        assert sp.freq[1] == pytest.approx(2.0)
        assert len(sp.freq) == 2

    def test_port_names(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.port_names == ["o1", "o2", "o3"]

    def test_bracket_access(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        assert s11.db[0] == pytest.approx(-20.0)
        assert s11.deg[0] == pytest.approx(-45.0)

    def test_bracket_access_cross(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s21 = sp["o2", "o1"]
        assert s21.db[0] == pytest.approx(-3.0)
        assert s21.deg[0] == pytest.approx(-90.0)

    def test_mag_property(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        expected_mag = 10 ** (-20.0 / 20)
        assert s11.mag[0] == pytest.approx(expected_mag)

    def test_complex_property(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        c = s11.complex[0]
        assert abs(c) == pytest.approx(10 ** (-20.0 / 20))
        assert np.rad2deg(np.angle(c)) == pytest.approx(-45.0)

    def test_rf_shorthand_s11(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s11.db[0] == pytest.approx(-20.0)

    def test_rf_shorthand_s21(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s21.db[0] == pytest.approx(-3.0)

    def test_rf_shorthand_s31(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s31.db[0] == pytest.approx(-30.0)

    def test_invalid_shorthand_raises(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        with pytest.raises(AttributeError):
            _ = sp.s99

    def test_invalid_bracket_raises(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        with pytest.raises(KeyError, match="not found"):
            _ = sp["o1", "o99"]

    def test_keys(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        keys = sp.keys()
        assert ("o1", "o1") in keys
        assert ("o2", "o1") in keys
        assert ("o3", "o1") in keys

    def test_repr(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        r = repr(sp)
        assert "3 ports" in r
        assert "o1" in r

    def test_to_dataframe(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        df = sp.to_dataframe()
        assert "freq_ghz" in df.columns
        assert "S_o1_o1_db" in df.columns

    def test_plot_runs(self, sim_dir: Path) -> None:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        sp = load_sparams(sim_dir)
        sp.plot()
        plt.close("all")


class TestLoadSparamsSource:
    """Tests for source resolution (dir, subdir, dict)."""

    def test_accepts_dir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert len(sp.freq) == 2

    def test_accepts_palace_subdir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir / "results" / "palace")
        assert len(sp.freq) == 2

    def test_accepts_results_dict(self, sim_dir: Path) -> None:
        results = {
            "port-S.csv": sim_dir / "results" / "palace" / "port-S.csv",
        }
        sp = load_sparams(results)
        assert sp["o1", "o1"].db[0] == pytest.approx(-20.0)

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"port-S\.csv"):
            load_sparams(tmp_path)

    def test_results_dict_missing_csv_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="port-S"):
            load_sparams({"other.csv": Path("/nonexistent")})

    def test_fallback_numeric_names(self, sim_dir_no_names: Path) -> None:
        sp = load_sparams(sim_dir_no_names)
        assert sp.port_names == ["p1", "p2"]
        assert sp["p1", "p1"].db[0] == pytest.approx(-20.0)


class TestPalaceRunSummary:
    """Tests for reusable Palace run artifact summaries."""

    def test_parse_palace_resource_log_extracts_sanitized_tables(
        self,
        tmp_path: Path,
    ) -> None:
        log_path = tmp_path / "palace-public.log"
        log_path.write_text(PALACE_RESOURCE_LOG)

        record = parse_palace_resource_log(log_path)

        assert record["solver"]["palace_git_changeset"] == "v0.16.1"
        assert record["solver"]["petsc_version"] == "3.24.3"
        assert record["allocation"]["num_processes"] == 4
        assert record["allocation"]["num_threads"] == 28
        assert record["runtime"]["wall_time_seconds"] == pytest.approx(121.0)
        assert record["model_size"]["completed_amr_iterations"] == 1
        assert record["model_size"]["global_unknowns"] == 10718029
        assert record["memory"]["peak_total_hwm_bytes"] == pytest.approx(20.8 * 1024**3)
        assert record["estimated_peak_memory"]["node"]["total"] == "20.9G"
        assert record["petsc_summary"]["processes"] == 4
        assert len(record["amr_passes"]) == 1
        assert record["amr_passes"][0]["global_unknowns"] == 887970
        assert len(record["stage_timing"]) == 8
        assert len(record["stage_memory"]) == 8
        serialized = json.dumps(record)
        assert "private-node" not in serialized
        assert "private-user" not in serialized
        assert "/opt/private-palace" not in serialized

    def test_estimated_peak_memory_populates_benchmark_pass_table(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        log_path = run_dir / "logs" / "palace-public.log"
        log_path.parent.mkdir(parents=True)
        memory_line_1 = (
            "Estimated peak per-node memory usage is: Min. 100.0M, "
            "Max. 150.0M, Avg. 125.0M, Total 300.0M"
        )
        memory_line_2 = (
            "Estimated peak per-node memory usage is: Min. 200.0M, "
            "Max. 250.0M, Avg. 225.0M, Total 500.0M"
        )
        log_path.write_text(
            dedent(
                f"""
                Running with 2 MPI processes, 3 OpenMP threads

                Elapsed Time Report (s)           Min.        Max.        Avg.
                ==============================================================
                Total                            1.000       1.200       1.100
                {memory_line_1}

                Proceeding with solve/estimate iteration 2...

                Elapsed Time Report (s)           Min.        Max.        Avg.
                ==============================================================
                Total                            2.000       2.400       2.200
                {memory_line_2}
                """
            )
        )

        write_palace_resource_record_from_log(
            run_dir,
            log_path,
            allocation={"cores": 6},
        )
        summary = load_palace_run_summary(run_dir)

        adaptive_passes = SimulationBenchmark.from_run_summary(
            summary
        ).adaptive_pass_dataframe()
        memory_gib = adaptive_passes.set_index("adaptive_pass")["peak_total_hwm_gib"]

        assert memory_gib.loc[1] == pytest.approx(300 * 1024**2 / 1024**3)
        assert memory_gib.loc[2] == pytest.approx(500 * 1024**2 / 1024**3)

    def test_parse_slurm_scontrol_job_extracts_sanitized_allocation(
        self,
        tmp_path: Path,
    ) -> None:
        scontrol_path = tmp_path / "scontrol-job-12345.txt"
        scontrol_path.write_text(SLURM_SCONTROL)

        record = parse_slurm_scontrol_job(scontrol_path)

        assert record["scheduler"] == {
            "kind": "slurm",
            "job_id": 12345,
            "job_state": "COMPLETED",
            "partition": "public_cpu",
            "submit_time": "2026-05-21T18:16:44",
            "start_time": "2026-05-21T18:24:47",
            "end_time": "2026-05-21T18:26:48",
            "time_limit": "00:10:00",
            "time_limit_seconds": 600,
            "run_time": "00:02:01",
            "run_time_seconds": 121,
        }
        assert record["allocation"]["nodes"] == 1
        assert record["allocation"]["num_cpus"] == 112
        assert record["allocation"]["num_tasks"] == 4
        assert record["allocation"]["cpus_per_task"] == 28
        assert record["allocation"]["num_processes"] == 4
        assert record["allocation"]["num_threads"] == 28
        assert record["allocation"]["cores"] == 112
        assert record["allocation"]["requested_memory"] == "482496M"
        assert record["allocation"]["requested_memory_bytes"] == pytest.approx(
            482496 * 1024**2
        )
        serialized = json.dumps(record)
        for forbidden in (
            "private_layout_run",
            "private-user",
            "private_account",
            "private-node",
            "/private/work",
            "Command",
            "WorkDir",
            "StdOut",
            "StdErr",
        ):
            assert forbidden not in serialized

    def test_load_palace_run_summary_records_handoff_and_results(
        self,
        indexed_report_dir: Path,
    ) -> None:
        config_path = indexed_report_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["Problem"] = {"Type": "Eigenmode"}
        config_path.write_text(json.dumps(config))

        (indexed_report_dir / "palace.msh").write_text("$MeshFormat\n")
        (indexed_report_dir / "metadata" / "mesh_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "name": "substrate",
                            "role": "dielectric_volume",
                            "dimension": 3,
                            "attributes": [10],
                            "physical_names": ["D1_SUBSTRATE"],
                        },
                        {
                            "name": "air___silicon",
                            "role": "boundary_surface",
                            "dimension": 2,
                            "attributes": [20],
                            "physical_names": ["air___silicon"],
                            "interface_of": ["air", "silicon"],
                        },
                    ],
                }
            )
        )
        (indexed_report_dir / "metadata" / "palace_run_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "return_code": 0,
                    "elapsed_seconds": 1.25,
                    "launcher": {"kind": "executable", "executable_mode": "binary"},
                    "resources": {"num_processes": 1, "num_threads": 2},
                    "command": {"argv": ["palace", "config.json"]},
                    "outputs": {"domain-E.csv": {"bytes": 42}},
                }
            )
        )

        summary = load_palace_run_summary(indexed_report_dir, include_hashes=True)

        assert isinstance(summary, PalaceRunSummary)
        assert summary.problem_type == "Eigenmode"
        assert summary.missing_artifacts == ()
        assert summary.artifacts["palace.msh"].present
        assert summary.artifacts["palace.msh"].bytes > 0
        assert summary.artifacts["palace.msh"].sha256
        assert summary.config["material_count"] == 2
        assert summary.config["problem_type"] == "Eigenmode"
        assert summary.mesh_manifest["roles"] == {
            "boundary_surface": 1,
            "dielectric_volume": 1,
        }
        assert summary.mesh_manifest["interface_entry_count"] == 1
        assert summary.index_map["sections"]["Domains.Postprocessing.Energy"] == 1
        assert summary.material_resolution["material_count"] == 2
        assert summary.material_resolution["interface_count"] == 1
        assert summary.runtime["present"] is True
        assert summary.runtime["status"] == "completed"
        assert summary.runtime["elapsed_seconds"] == pytest.approx(1.25)
        assert summary.runtime["launcher"] == {
            "kind": "executable",
            "executable_mode": "binary",
        }
        assert summary.runtime["output_count"] == 1
        assert summary.runtime["output_bytes"] == 42
        assert summary.resource["present"] is False
        assert summary.results["domain-E.csv"].present
        assert summary.results["surface-Q.csv"].present
        assert summary.results["port-EPR.csv"].present

        as_dict = summary.to_dict()
        assert as_dict["problem_type"] == "Eigenmode"
        assert as_dict["missing_artifacts"] == []
        assert as_dict["artifacts"]["config.json"]["present"] is True
        assert as_dict["results"]["domain-E.csv"]["bytes"] > 0
        assert as_dict["runtime"]["present"] is True

    def test_load_palace_run_summary_accepts_results_dict(
        self,
        indexed_report_dir: Path,
    ) -> None:
        config_path = indexed_report_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["Problem"] = {"Type": "Eigenmode"}
        config_path.write_text(json.dumps(config))
        results = {
            "domain-E.csv": indexed_report_dir / "results" / "palace" / "domain-E.csv",
            "config.json": config_path,
            "palace_index_map.json": indexed_report_dir
            / "metadata"
            / "palace_index_map.json",
            "palace_material_resolution.json": indexed_report_dir
            / "metadata"
            / "palace_material_resolution.json",
        }

        summary = load_palace_run_summary(results)

        assert summary.problem_type == "Eigenmode"
        assert summary.artifacts["config.json"].present
        assert summary.artifacts["palace_index_map.json"].present
        assert summary.artifacts["palace.msh"].present is False
        assert summary.results["domain-E.csv"].present
        assert summary.runtime["present"] is False
        assert "palace.msh" in summary.missing_artifacts

    def test_write_palace_handoff_metadata_round_trips_through_run_summary(
        self,
        indexed_report_dir: Path,
    ) -> None:
        script_path = indexed_report_dir / "run_palace.sbatch"
        archive_path = indexed_report_dir.parent / "public-fixture-palace.tar.gz"
        script_path.write_text("#!/bin/bash\n")
        archive_path.write_text("archive placeholder\n")

        sidecar_path = write_palace_handoff_metadata(
            indexed_report_dir,
            status="planned",
            launcher={"kind": "slurm"},
            profile={"name": "public-test:cpu", "partition": "cpu"},
            resources={"nodes": 1, "tasks_per_node": 2, "wall_time": "00:10:00"},
            script_path=script_path.name,
            archive_path=archive_path,
            command={"argv": ["sbatch", script_path.name]},
            metadata={"campaign": "public_fixture"},
        )

        assert sidecar_path == (
            indexed_report_dir / "metadata" / "palace_handoff_metadata.json"
        )
        payload = json.loads(sidecar_path.read_text())
        assert payload["schema_version"] == 1
        assert payload["script"]["path"] == "run_palace.sbatch"

        summary = load_palace_run_summary(indexed_report_dir)

        assert summary.handoff["present"] is True
        assert summary.handoff["status"] == "planned"
        assert summary.handoff["launcher"] == {"kind": "slurm"}
        assert summary.handoff["profile"]["name"] == "public-test:cpu"
        assert summary.handoff["resources"]["nodes"] == 1
        assert summary.handoff["script_present"] is True
        assert summary.handoff["archive_present"] is True
        assert summary.handoff["metadata"] == {"campaign": "public_fixture"}
        assert "palace_handoff_metadata.json" not in summary.results
        assert summary.to_dict()["handoff"]["present"] is True

    def test_write_palace_resource_record_round_trips_through_run_summary(
        self,
        indexed_report_dir: Path,
    ) -> None:
        record_path = write_palace_resource_record(
            indexed_report_dir,
            status="completed",
            sources={"palace_log": {"path": "logs/palace-public.log"}},
            launcher={"kind": "slurm"},
            solver={
                "palace_git_changeset": "v0.16.1",
                "petsc_version": "3.24.3",
            },
            allocation={"nodes": 1, "num_processes": 4, "num_threads": 28},
            runtime={"wall_time_seconds": 120.0},
            model_size={"global_unknowns": 123456},
            memory={"peak_total_hwm_bytes": 2 * 1024**3},
            tables={"stage_timing": "metadata/palace_stage_timing.csv"},
            missing_sources=("metadata/scontrol-job-123.txt",),
            parse_warnings=("partial Palace log",),
            metadata={"workflow": "public-test"},
        )

        assert record_path == (
            indexed_report_dir / "metadata" / "palace_resource_record.json"
        )
        summary = load_palace_run_summary(indexed_report_dir)

        assert summary.resource["present"] is True
        assert summary.resource["status"] == "completed"
        assert summary.resource["launcher"] == {"kind": "slurm"}
        assert summary.resource["solver"]["palace_git_changeset"] == "v0.16.1"
        assert summary.resource["allocation"]["num_processes"] == 4
        assert summary.resource["allocation"]["num_threads"] == 28
        assert summary.resource["runtime"]["wall_time_seconds"] == pytest.approx(120.0)
        assert summary.resource["runtime"]["core_hours"] == pytest.approx(
            120.0 * 112 / 3600
        )
        assert summary.resource["model_size"]["global_unknowns"] == 123456
        assert summary.resource["memory"]["peak_total_hwm_gib"] == pytest.approx(2.0)
        assert summary.resource["source_count"] == 1
        assert summary.resource["table_count"] == 1
        assert summary.resource["missing_source_count"] == 1
        assert summary.resource["parse_warning_count"] == 1
        assert summary.resource["metadata"] == {"workflow": "public-test"}
        assert summary.resource["path"] == str(record_path)
        assert "palace_resource_record.json" not in summary.results
        assert summary.to_dict()["resource"]["present"] is True

    def test_write_palace_resource_record_from_log_writes_table_sidecars(
        self,
        indexed_report_dir: Path,
    ) -> None:
        log_path = indexed_report_dir / "logs" / "palace-public.log"
        log_path.parent.mkdir()
        log_path.write_text(PALACE_RESOURCE_LOG)
        scontrol_path = indexed_report_dir / "metadata" / "scontrol-job-12345.txt"
        scontrol_path.parent.mkdir(parents=True, exist_ok=True)
        scontrol_path.write_text(SLURM_SCONTROL)

        record_path = write_palace_resource_record_from_log(
            indexed_report_dir,
            log_path,
            scontrol_path=scontrol_path,
            allocation={"nodes": 1},
            metadata={"workflow": "public-test"},
        )

        assert record_path == (
            indexed_report_dir / "metadata" / "palace_resource_record.json"
        )
        records_dir = indexed_report_dir / "metadata"
        assert (records_dir / "palace_amr_passes.csv").is_file()
        assert (records_dir / "palace_stage_timing.csv").is_file()
        assert (records_dir / "palace_stage_memory.csv").is_file()
        assert (
            "Operator Construction"
            in (records_dir / "palace_stage_timing.csv").read_text()
        )

        summary = load_palace_run_summary(indexed_report_dir)

        assert summary.resource["present"] is True
        assert summary.resource["status"] == "completed"
        assert summary.resource["launcher"] == {"kind": "slurm"}
        assert summary.resource["scheduler"]["kind"] == "slurm"
        assert summary.resource["scheduler"]["job_id"] == 12345
        assert summary.resource["scheduler"]["job_state"] == "COMPLETED"
        assert summary.resource["scheduler"]["partition"] == "public_cpu"
        assert summary.resource["solver"]["palace_git_changeset"] == "v0.16.1"
        assert summary.resource["solver"]["petsc_version"] == "3.24.3"
        assert summary.resource["allocation"]["nodes"] == 1
        assert summary.resource["allocation"]["num_processes"] == 4
        assert summary.resource["allocation"]["num_threads"] == 28
        assert summary.resource["allocation"]["cores"] == 112
        assert summary.resource["runtime"]["wall_time_seconds"] == pytest.approx(121.0)
        assert summary.resource["runtime"]["core_hours"] == pytest.approx(
            121.0 * 112 / 3600
        )
        assert summary.resource["model_size"]["global_unknowns"] == 10718029
        assert summary.resource["memory"]["peak_total_hwm_gib"] == pytest.approx(20.8)
        assert summary.resource["source_count"] == 2
        assert summary.resource["sources"]["palace_log"]["path"] == (
            "logs/palace-public.log"
        )
        assert summary.resource["sources"]["slurm_scontrol"]["path"] == (
            "metadata/scontrol-job-12345.txt"
        )
        assert summary.resource["table_count"] == 3
        assert summary.resource["tables"]["stage_timing"]["path"] == (
            "metadata/palace_stage_timing.csv"
        )
        assert summary.resource["tables"]["stage_timing"]["row_count"] == 8

        benchmark = SimulationBenchmark.from_run_summary(summary)
        adaptive_passes = benchmark.adaptive_pass_dataframe().set_index("adaptive_pass")
        assert adaptive_passes.index.tolist() == [1, 2]
        assert adaptive_passes.loc[1, "cumulative_wall_time_seconds"] == (
            pytest.approx(58.58)
        )
        assert adaptive_passes.loc[2, "cumulative_wall_time_seconds"] == (
            pytest.approx(121.0)
        )
        assert adaptive_passes.loc[2, "cumulative_core_hours"] == pytest.approx(
            121.0 * 112 / 3600
        )
        assert adaptive_passes.loc[1, "peak_total_hwm_gib"] == pytest.approx(10.8)
        assert adaptive_passes.loc[2, "peak_total_hwm_gib"] == pytest.approx(20.8)
        assert adaptive_passes.loc[1, "global_unknowns"] == 887970
        assert adaptive_passes.loc[2, "global_unknowns"] == 10718029
        benchmark_items = benchmark.visualize()
        assert "simulation_benchmark_summary_table" in benchmark_items
        assert "simulation_benchmark_table" not in benchmark_items
        assert "simulation_benchmark_adaptive_pass_table" in benchmark_items
        assert "simulation_benchmark_adaptive_pass_trace_plot" in benchmark_items
        assert "simulation_benchmark_metrics_bar_plot" not in benchmark_items
        figure = cast(
            Any, benchmark_items["simulation_benchmark_adaptive_pass_trace_plot"]
        )
        assert figure.data[0].y[-1] == pytest.approx(121.0 / 60.0)
        assert figure.layout.yaxis.title.text == "Wall time (minutes)"
        assert figure.layout.yaxis2.title.text == (
            "Core-hours (allocated cores x elapsed hours)"
        )
        assert figure.layout.xaxis.title.text is None
        assert figure.layout.xaxis4.title.text == "Adaptive pass"
        assert figure.layout.height >= 1200

        serialized = json.dumps(summary.resource)
        assert "private-node" not in serialized
        assert "private-user" not in serialized
        assert "private_account" not in serialized
        assert "private_layout_run" not in serialized
        assert "/private/work" not in serialized
        assert "/opt/private-palace" not in serialized

    def test_simulation_benchmark_omits_empty_runtime_resource_summary(self) -> None:
        benchmark = SimulationBenchmark(
            problem_type="Electrostatic",
            runtime={"present": False},
            resource={"present": False},
        )

        assert benchmark.visualize() == {}


class TestPalaceSweepSummary:
    """Tests for reusable point-local Palace sweep summaries."""

    def test_write_palace_sweep_points_round_trips_through_loader(
        self,
        indexed_report_dir: Path,
    ) -> None:
        sweep_root = indexed_report_dir / "written_sweep"
        point_root = sweep_root / "points" / "gap_6um"
        _write_sweep_point_artifacts(indexed_report_dir, point_root)

        points_path = write_palace_sweep_points(
            sweep_root,
            [
                PalaceSweepPointSpec(
                    point_slug="gap_6um",
                    parameters={
                        "gap_um": 6.0,
                        "solver": {"order": 2},
                    },
                    run_dir="points/gap_6um",
                )
            ],
            sweep_id="gap_sweep",
            metadata={"campaign": "public_fixture"},
        )

        payload = json.loads(points_path.read_text())
        assert payload["schema_version"] == 1
        assert payload["sweep_id"] == "gap_sweep"
        assert payload["campaign"] == "public_fixture"
        assert payload["points"][0]["point_slug"] == "gap_6um"
        assert payload["points"][0]["parameters"]["solver"] == {"order": 2}

        summary = load_palace_sweep_summary(sweep_root)
        assert summary.sweep_id == "gap_sweep"
        assert summary.metadata["campaign"] == "public_fixture"
        assert summary.point_slugs == ("gap_6um",)
        assert summary.duplicate_point_slugs == ()
        assert summary.points[0].point_slug == "gap_6um"
        assert summary.points[0].parameters["gap_um"] == 6.0
        assert summary.points[0].run_summary.missing_artifacts == ()

    def test_write_palace_sweep_points_requires_explicit_point_slug(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="point_slug"):
            write_palace_sweep_points(tmp_path, [{"parameters": {"gap_um": 6.0}}])

    def test_write_palace_sweep_points_rejects_duplicate_point_slugs(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="duplicates: 'gap_6um'"):
            write_palace_sweep_points(
                tmp_path,
                [
                    PalaceSweepPointSpec(point_slug="gap_6um"),
                    PalaceSweepPointSpec(point_slug="gap_6um"),
                ],
            )

    def test_load_palace_sweep_summary_uses_points_json_run_dirs(
        self,
        indexed_report_dir: Path,
    ) -> None:
        sweep_root = indexed_report_dir / "sweep_run_dirs"
        point_root = sweep_root / "points" / "gap_6um"
        _write_sweep_point_artifacts(indexed_report_dir, point_root)
        (sweep_root / "points.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sweep_id": "gap_sweep",
                    "points": [
                        {
                            "point_slug": "gap_6um",
                            "parameters": {
                                "gap_um": 6.0,
                                "solver": {"order": 2},
                            },
                            "run_dir": "points/gap_6um",
                        }
                    ],
                }
            )
        )

        summary = load_palace_sweep_summary(sweep_root)

        assert isinstance(summary, PalaceSweepSummary)
        assert summary.sweep_id == "gap_sweep"
        assert summary.point_count == 1
        assert summary.complete_point_count == 1
        assert summary.runtime_present_count == 1
        assert summary.problem_types == ("Eigenmode",)
        assert summary.total_runtime_elapsed_seconds == pytest.approx(2.5)
        point = summary.points[0]
        assert point.point_slug == "gap_6um"
        assert point.parameters == {"gap_um": 6.0, "solver": {"order": 2}}
        assert point.run_summary.results["domain-E.csv"].present
        assert point.run_summary.runtime["status"] == "completed"

        records = summary.to_point_records()
        assert records[0]["sweep_id"] == "gap_sweep"
        assert records[0]["point_slug"] == "gap_6um"
        assert records[0]["parameter_gap_um"] == 6.0
        assert records[0]["parameter_solver"] == '{"order": 2}'
        assert records[0]["problem_type"] == "Eigenmode"
        assert records[0]["complete"] is True
        assert records[0]["runtime_elapsed_seconds"] == pytest.approx(2.5)
        assert records[0]["result_count"] == 3
        assert records[0]["result_bytes"] > 0
        assert records[0]["core_artifact_count"] == 5

        frame = summary.to_dataframe()
        assert frame.loc[0, "parameter_gap_um"] == 6.0
        assert bool(frame.loc[0, "runtime_present"]) is True

        as_dict = summary.to_dict()
        assert as_dict["point_count"] == 1
        assert as_dict["point_slugs"] == ["gap_6um"]
        assert as_dict["duplicate_point_slugs"] == []
        assert as_dict["point_records"][0]["parameter_gap_um"] == 6.0
        assert as_dict["points"][0]["missing_artifacts"] == []
        assert as_dict["points"][0]["record"]["parameter_gap_um"] == 6.0

    def test_load_palace_sweep_summary_includes_handoff_records(
        self,
        indexed_report_dir: Path,
    ) -> None:
        sweep_root = indexed_report_dir / "sweep_handoff"
        point_root = sweep_root / "points" / "gap_6um"
        _write_sweep_point_artifacts(indexed_report_dir, point_root)
        (point_root / "run_palace.sbatch").write_text("#!/bin/bash\n")
        write_palace_handoff_metadata(
            point_root,
            status="planned",
            launcher={"kind": "slurm"},
            profile={"name": "public-test:cpu"},
            resources={"nodes": 1, "tasks_per_node": 1},
            script_path="run_palace.sbatch",
        )
        points_path = write_palace_sweep_points(
            sweep_root,
            [
                PalaceSweepPointSpec(
                    point_slug="gap_6um",
                    parameters={"gap_um": 6.0},
                    run_dir="points/gap_6um",
                    handoff_metadata_path=(
                        "points/gap_6um/palace_handoff_metadata.json"
                    ),
                )
            ],
            sweep_id="handoff_sweep",
        )
        payload = json.loads(points_path.read_text())
        assert payload["points"][0]["handoff_metadata_path"] == (
            "points/gap_6um/palace_handoff_metadata.json"
        )

        summary = load_palace_sweep_summary(sweep_root)

        point = summary.points[0]
        assert point.run_summary.handoff["present"] is True
        assert point.run_summary.handoff["profile"]["name"] == "public-test:cpu"
        record = summary.to_point_records()[0]
        assert record["handoff_present"] is True
        assert record["handoff_status"] == "planned"
        assert record["handoff_profile_name"] == "public-test:cpu"
        assert record["handoff_script_present"] is True
        assert record["handoff_archive_present"] is False
        assert (
            summary.to_dict()["points"][0]["run_summary"]["handoff"]["present"] is True
        )

    def test_load_palace_sweep_summary_includes_resource_records(
        self,
        indexed_report_dir: Path,
    ) -> None:
        sweep_root = indexed_report_dir / "sweep_resource"
        point_root = sweep_root / "points" / "gap_6um"
        _write_sweep_point_artifacts(indexed_report_dir, point_root)
        write_palace_resource_record(
            point_root,
            status="completed",
            scheduler={
                "kind": "slurm",
                "job_id": 12345,
                "job_state": "COMPLETED",
                "partition": "public_cpu",
            },
            allocation={"nodes": 1, "num_processes": 2, "num_threads": 8},
            runtime={"wall_time_seconds": 90.0},
            model_size={"global_unknowns": 654321},
            memory={"peak_total_hwm_bytes": 1024**3},
        )
        points_path = write_palace_sweep_points(
            sweep_root,
            [
                PalaceSweepPointSpec(
                    point_slug="gap_6um",
                    parameters={"gap_um": 6.0},
                    run_dir="points/gap_6um",
                    resource_record_path=(
                        "points/gap_6um/metadata/palace_resource_record.json"
                    ),
                )
            ],
            sweep_id="resource_sweep",
        )
        payload = json.loads(points_path.read_text())
        assert payload["points"][0]["resource_record_path"] == (
            "points/gap_6um/metadata/palace_resource_record.json"
        )

        summary = load_palace_sweep_summary(sweep_root)

        assert summary.resource_present_count == 1
        point = summary.points[0]
        assert point.run_summary.resource["present"] is True
        assert point.run_summary.resource["status"] == "completed"
        record = summary.to_point_records()[0]
        assert record["resource_present"] is True
        assert record["resource_status"] == "completed"
        assert record["resource_wall_time_seconds"] == pytest.approx(90.0)
        assert record["resource_core_hours"] == pytest.approx(90.0 * 16 / 3600)
        assert record["resource_nodes"] == 1
        assert record["resource_num_processes"] == 2
        assert record["resource_num_threads"] == 8
        assert record["resource_global_unknowns"] == 654321
        assert record["resource_peak_total_hwm_gib"] == pytest.approx(1.0)
        assert record["resource_scheduler_kind"] == "slurm"
        assert record["resource_scheduler_job_id"] == 12345
        assert record["resource_scheduler_job_state"] == "COMPLETED"
        assert record["resource_scheduler_partition"] == "public_cpu"
        assert summary.to_dict()["resource_present_count"] == 1

        index_result = write_palace_sweep_resource_index(sweep_root)
        assert isinstance(index_result, PalaceSweepResourceIndexResult)
        assert index_result.point_count == 1
        assert index_result.resource_present_count == 1
        assert index_result.summary_path == (
            sweep_root / "metadata" / "records" / "sweep_resource_index.json"
        )
        assert index_result.point_records_csv_path.is_file()
        assert index_result.resource_records_csv_path.is_file()
        assert index_result.benchmark_jsonl_path.is_file()

        index_payload = json.loads(index_result.summary_path.read_text())
        assert index_payload["schema_version"] == 1
        assert index_payload["sweep_id"] == "resource_sweep"
        assert index_payload["point_count"] == 1
        assert index_payload["resource_present_count"] == 1
        assert index_payload["records"] == {
            "benchmark_jsonl": "metadata/records/sweep_benchmark_index.jsonl",
            "point_records_csv": "metadata/records/sweep_point_records.csv",
            "resource_records_csv": "metadata/records/sweep_resource_records.csv",
        }

        point_csv = index_result.point_records_csv_path.read_text()
        resource_csv = index_result.resource_records_csv_path.read_text()
        assert "resource_core_hours" in point_csv
        assert "resource_scheduler_job_id" in point_csv
        assert "gap_6um" in resource_csv
        jsonl_rows = index_result.benchmark_jsonl_path.read_text().splitlines()
        assert len(jsonl_rows) == 1
        jsonl_record = json.loads(jsonl_rows[0])
        assert jsonl_record["point_slug"] == "gap_6um"
        assert jsonl_record["resource_scheduler_partition"] == "public_cpu"
        assert index_result.to_dict()["resource_present_count"] == 1

    def test_load_palace_sweep_summary_reports_duplicate_point_slugs(
        self,
        indexed_report_dir: Path,
    ) -> None:
        sweep_root = indexed_report_dir / "duplicate_sweep"
        point_root = sweep_root / "points" / "gap_6um"
        _write_sweep_point_artifacts(indexed_report_dir, point_root)
        (sweep_root / "points.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sweep_id": "duplicate_sweep",
                    "points": [
                        {"point_slug": "gap_6um", "run_dir": "points/gap_6um"},
                        {"point_slug": "gap_6um", "run_dir": "points/gap_6um"},
                    ],
                }
            )
        )

        summary = load_palace_sweep_summary(sweep_root)

        assert summary.point_slugs == ("gap_6um", "gap_6um")
        assert summary.duplicate_point_slugs == ("gap_6um",)
        assert summary.parse_warnings == (
            "Duplicate sweep point_slug 'gap_6um' at index 1",
        )
        assert summary.to_dict()["duplicate_point_slugs"] == ["gap_6um"]

    def test_load_palace_sweep_summary_accepts_split_point_and_result_dirs(
        self,
        indexed_report_dir: Path,
    ) -> None:
        sweep_root = indexed_report_dir / "sweep_split_dirs"
        run_dir = sweep_root / "points" / "gap_8um"
        result_dir = sweep_root / "results" / "gap_8um" / "palace"
        _write_sweep_point_artifacts(
            indexed_report_dir,
            run_dir,
            result_dir=result_dir,
            mesh_name="mesh.msh",
        )
        (sweep_root / "points.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sweep_id": "split_sweep",
                    "points": [
                        {
                            "point_slug": "gap_8um",
                            "parameters": {"gap_um": 8.0},
                            "config_path": "points/gap_8um/config.json",
                            "mesh_path": "points/gap_8um/mesh.msh",
                            "result_dir": "results/gap_8um/palace",
                        }
                    ],
                }
            )
        )

        summary = load_palace_sweep_summary(sweep_root, include_hashes=True)

        point = summary.points[0]
        assert summary.sweep_id == "split_sweep"
        assert point.run_summary.artifacts["palace.msh"].present
        assert point.run_summary.artifacts["palace.msh"].path == run_dir / "mesh.msh"
        assert point.run_summary.artifacts["palace.msh"].sha256
        assert point.run_summary.results["domain-E.csv"].path == (
            result_dir / "domain-E.csv"
        )
        assert point.run_summary.runtime["present"] is True
        assert point.run_summary.missing_artifacts == ()

    def test_load_palace_sweep_summary_report_metrics_for_driven(
        self,
        tmp_path: Path,
        driven_report_dir: Path,
    ) -> None:
        summary = _single_point_report_sweep(
            tmp_path / "driven_sweep",
            driven_report_dir,
            point_slug="driven_point",
            problem_type="Driven",
        )

        point = summary.points[0]
        assert point.report_metrics["status"] == "loaded"
        assert point.report_metrics["frequency_point_count"] == 2
        assert point.report_metrics["port_count"] == 1
        assert point.report_metrics["s_parameter_count"] == 1
        assert point.report_metrics["domain_energy_rows"] == 2
        assert point.report_metrics["surface_q_rows"] == 1
        assert point.report_metrics["loss_budget_rows"] == 1

        record = summary.to_point_records()[0]
        assert record["report_status"] == "loaded"
        assert record["report_frequency_point_count"] == 2
        assert record["report_s_parameter_count"] == 1

    def test_load_palace_sweep_summary_report_metrics_for_eigenmode(
        self,
        tmp_path: Path,
        eigenmode_report_dir: Path,
    ) -> None:
        summary = _single_point_report_sweep(
            tmp_path / "eigenmode_sweep",
            eigenmode_report_dir,
            point_slug="eigenmode_point",
            problem_type="Eigenmode",
        )

        point = summary.points[0]
        assert point.report_metrics["status"] == "loaded"
        assert point.report_metrics["mode_count"] == 2
        assert point.report_metrics["pass_count"] == 2
        assert point.report_metrics["min_frequency_ghz"] == pytest.approx(6.3)
        assert point.report_metrics["min_q"] == pytest.approx(110.0)
        assert point.report_metrics["loss_budget_rows"] == 2

        record = summary.to_point_records()[0]
        assert record["report_status"] == "loaded"
        assert record["report_mode_count"] == 2
        assert record["report_loss_budget_rows"] == 2

    def test_load_palace_sweep_summary_report_metrics_for_electrostatic(
        self,
        tmp_path: Path,
        electrostatic_report_dir: Path,
    ) -> None:
        summary = _single_point_report_sweep(
            tmp_path / "electrostatic_sweep",
            electrostatic_report_dir,
            point_slug="electrostatic_point",
            problem_type="Electrostatic",
        )

        point = summary.points[0]
        assert point.report_metrics["status"] == "loaded"
        assert point.report_metrics["terminal_count"] == 2
        assert point.report_metrics["capacitance_row_count"] == 2
        assert point.report_metrics["capacitance_column_count"] == 2
        assert point.report_metrics["has_mutual_capacitance"] is True
        assert point.report_metrics["has_inverse_capacitance"] is True
        assert point.report_metrics["loss_budget_rows"] == 2

        record = summary.to_point_records()[0]
        assert record["report_status"] == "loaded"
        assert record["report_terminal_count"] == 2
        assert record["report_has_mutual_capacitance"] is True

    def test_load_palace_sweep_summary_report_metrics_can_record_missing_reports(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "missing_driven_report"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps({"Problem": {"Type": "Driven"}})
        )
        summary = _single_point_report_sweep(
            tmp_path / "missing_report_sweep",
            run_dir,
            point_slug="missing_report_point",
            problem_type="Driven",
        )

        point = summary.points[0]
        assert point.report_metrics["status"] == "missing"
        assert point.report_metrics["problem_type"] == "Driven"
        assert "port-S.csv" in point.report_metrics["message"]


class TestDrivenReport:
    """Tests for composed Palace driven report bundles."""

    def test_load_driven_report_composes_existing_summaries(
        self,
        driven_report_dir: Path,
    ) -> None:
        report = load_driven_report(driven_report_dir)

        assert isinstance(report, DrivenReport)
        assert report.sparams.port_names == ["readout"]
        assert report.network is report.sparams
        assert report.sparams["readout", "readout"].db.tolist() == pytest.approx(
            [-12.0, -6.0]
        )
        material_rows = report.domain_materials.set_index("material_attribute")
        assert material_rows.loc[10, "source_name"] == "D1_SUBSTRATE"
        assert material_rows.loc[10, "material_name"] == "silicon"
        interface_rows = report.dielectric_interfaces.set_index("surface_index")
        assert interface_rows.loc[2, "source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert interface_rows.loc[2, "preset_name"] == "public_ma"
        assert report.domain_energy.iloc[0]["source_name"] == "D1_SUBSTRATE"
        assert report.domain_energy.iloc[0]["p_elec"] == pytest.approx(0.5)
        assert report.surface_q.iloc[0]["source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert report.surface_q.iloc[0]["p_surf"] == pytest.approx(1.0e-7)
        surface_by_interface = report.surface_interface_summary.set_index(
            "interface_type"
        )
        assert surface_by_interface.loc["MA", "surface_count"] == 1
        assert report.domain_loss.iloc[0]["p_elec"] == pytest.approx(0.5)
        assert report.surface_loss.iloc[0]["p_surf"] == pytest.approx(1.0e-7)
        budget_by_sample = report.loss_budget.set_index(
            ["sample_column", "sample_value"]
        )
        assert budget_by_sample.loc[("m", 1.0), "total_inverse_q_sum"] == pytest.approx(
            1.0e-6
        )
        assert report.index_map["entry_name"].tolist() == [
            "substrate",
            "ma_interface",
            "readout_port_surface",
        ]
        assert report.missing_reports == ()
        sources = report.sources.set_index("name")
        assert bool(sources.loc["port-S.csv", "loaded"])
        assert bool(sources.loc["palace_index_map.json", "loaded"])
        assert bool(sources.loc["config.json", "loaded"])
        assert bool(sources.loc["domain-E.csv", "loaded"])
        assert bool(sources.loc["surface-Q.csv", "loaded"])
        assert "port-EPR.csv" not in sources.index

    def test_load_driven_report_allows_missing_optional_reports(
        self,
        sim_dir: Path,
    ) -> None:
        report = load_driven_report(sim_dir)

        assert report.sparams.port_names == ["o1", "o2", "o3"]
        assert report.domain_energy.empty
        assert report.surface_q.empty
        assert report.domain_materials.empty
        assert report.dielectric_interfaces.empty
        assert report.index_map.empty
        assert report.missing_reports == (
            "palace_index_map.json",
            "config.json",
            "domain-E.csv",
            "surface-Q.csv",
        )
        sources = report.sources.set_index("name")
        assert bool(sources.loc["port-S.csv", "loaded"])
        assert not bool(sources.loc["domain-E.csv", "loaded"])
        assert not bool(sources.loc["surface-Q.csv", "loaded"])
        assert "port-EPR.csv" not in sources.index

    def test_load_driven_report_does_not_expose_port_epr(
        self,
        driven_report_dir: Path,
    ) -> None:
        report = load_driven_report(driven_report_dir)

        assert not hasattr(report, "port_epr")
        assert "port-EPR.csv" not in report.sources.set_index("name").index


class TestEigenmodes:
    """Tests for Palace eigenmode CSV loading."""

    def test_load_eigenmodes_normalizes_palace_columns(
        self,
        eigenmode_dir: Path,
    ) -> None:
        eigenmodes = load_eigenmodes(eigenmode_dir)

        assert isinstance(eigenmodes, Eigenmodes)
        assert eigenmodes.n_modes == 2
        assert eigenmodes.mode_indices.tolist() == [1, 2]
        np.testing.assert_allclose(eigenmodes.freq_real_ghz, [6.1, 7.2])
        np.testing.assert_allclose(eigenmodes.freq_imag_ghz, [0.01, 0.02])
        np.testing.assert_allclose(eigenmodes.q, [300.0, 400.0])

        frame = eigenmodes.to_dataframe()
        assert list(frame.columns) == [
            "mode_index",
            "freq_real_ghz",
            "freq_imag_ghz",
            "q",
            "error_backward",
            "error_absolute",
        ]
        assert frame.attrs["csv_path"].endswith("eig.csv")
        assert tuple(frame.attrs["source_columns"]) == (
            "m",
            "Re{f} (GHz)",
            "Im{f} (GHz)",
            "Q",
            "Error (Bkwd.)",
            "Error (Abs.)",
        )

        report = eigenmodes.to_report_dataframe()
        assert list(report.columns) == [
            "mode_index",
            "frequency_ghz",
            "imaginary_frequency_ghz",
            "q_factor",
            "backward_error",
            "absolute_error",
        ]

    def test_load_eigenmodes_accepts_results_dict(
        self,
        eigenmode_dir: Path,
    ) -> None:
        results = {
            "eig.csv": eigenmode_dir / "results" / "palace" / "eig.csv",
        }

        eigenmodes = load_eigenmodes(results)

        assert eigenmodes.source_path == results["eig.csv"]
        np.testing.assert_allclose(eigenmodes.freq_real_ghz, [6.1, 7.2])

    def test_load_eigenmodes_accepts_csv_path(self, eigenmode_dir: Path) -> None:
        csv_path = eigenmode_dir / "results" / "palace" / "eig.csv"

        eigenmodes = load_eigenmodes(csv_path)

        assert eigenmodes.source_path == csv_path
        assert eigenmodes.n_modes == 2

    def test_load_eigenmodes_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"eig\.csv"):
            load_eigenmodes(tmp_path)

    def test_load_eigenmodes_fills_optional_columns(self, tmp_path: Path) -> None:
        palace_dir = tmp_path / "results" / "palace"
        palace_dir.mkdir(parents=True)
        (palace_dir / "eig.csv").write_text("m, Re{f} (GHz)\n1, 5.5\n")

        eigenmodes = load_eigenmodes(tmp_path)

        assert eigenmodes.freq_imag_ghz.tolist() == [0.0]
        assert np.isnan(eigenmodes.q[0])

    def test_load_eigenmode_history_deduplicates_final_and_summarizes(
        self,
        tmp_path: Path,
    ) -> None:
        palace_dir = tmp_path / "results" / "palace"
        iteration01 = palace_dir / "iteration01"
        iteration02 = palace_dir / "iteration02"
        iteration01.mkdir(parents=True)
        iteration02.mkdir()
        pass1 = [
            [1, 6.0, 0.01, 100.0, 1.0e-7, 1.0e-4],
            [2, 8.0, 0.02, 200.0, 2.0e-7, 2.0e-4],
        ]
        pass2 = [
            [1, 6.3, 0.02, 110.0, 1.0e-8, 1.0e-5],
            [2, 8.4, 0.04, 210.0, 2.0e-8, 2.0e-5],
        ]
        _write_eig_csv(iteration01 / "eig.csv", pass1)
        _write_eig_csv(iteration02 / "eig.csv", pass2)
        _write_eig_csv(palace_dir / "eig.csv", pass2)

        history = load_eigenmode_history(tmp_path)

        assert history["iteration_index"].drop_duplicates().tolist() == [1, 2]
        assert len(history) == 4
        mode1_pass2 = history.loc[
            (history["iteration_index"] == 2) & (history["mode_index"] == 1)
        ].iloc[0]
        assert mode1_pass2["source_kind"] == "iteration"
        assert mode1_pass2["source_iteration"] == 2
        assert mode1_pass2["delta_to_previous_mhz"] == pytest.approx(300.0)
        assert mode1_pass2["abs_relative_delta_to_previous_percent"] == pytest.approx(
            5.0
        )

        summary = summarize_eigenmode_history(history)
        pass2_summary = summary.loc[summary["iteration_index"] == 2].iloc[0]
        assert pass2_summary["n_modes"] == 2
        assert pass2_summary["max_abs_delta_to_previous_mhz"] == pytest.approx(400.0)
        assert pass2_summary["hfss_max_delta_freq_percent"] == pytest.approx(5.0)

    def test_load_eigenmode_history_appends_nonmatching_final(
        self,
        tmp_path: Path,
    ) -> None:
        palace_dir = tmp_path / "results" / "palace"
        iteration01 = palace_dir / "iteration01"
        iteration01.mkdir(parents=True)
        _write_eig_csv(iteration01 / "eig.csv", [[1, 6.0, 0.01, 100.0, 1e-7, 1e-4]])
        _write_eig_csv(palace_dir / "eig.csv", [[1, 6.2, 0.02, 120.0, 1e-8, 1e-5]])

        history = load_eigenmode_history(tmp_path)

        assert history["iteration_index"].drop_duplicates().tolist() == [1, 2]
        final_rows = history.loc[history["is_final"]]
        assert len(final_rows) == 1
        assert final_rows.iloc[0]["source_kind"] == "final"
        assert final_rows.iloc[0]["source_iteration"] is None


class TestEigenmodeReport:
    """Tests for composed Palace eigenmode report bundles."""

    def test_load_eigenmode_report_composes_existing_summaries(
        self,
        eigenmode_report_dir: Path,
    ) -> None:
        report = load_eigenmode_report(eigenmode_report_dir)

        assert isinstance(report, EigenmodeReport)
        assert report.eigenmodes.n_modes == 2
        assert report.modes["frequency_ghz"].tolist() == pytest.approx([6.3, 8.4])
        assert report.mode_history["iteration_index"].drop_duplicates().tolist() == [
            1,
            2,
        ]
        assert report.pass_summary["n_modes"].tolist() == [2, 2]
        material_rows = report.domain_materials.set_index("material_attribute")
        assert material_rows.loc[10, "source_name"] == "D1_SUBSTRATE"
        assert material_rows.loc[10, "material_name"] == "silicon"
        assert material_rows.loc[10, "permittivity"] == pytest.approx(11.45)
        assert material_rows.loc[10, "stack_material_name"] == "Si"
        assert material_rows.loc[10, "material_model_source"] == (
            "test PDK material overlay"
        )
        assert material_rows.loc[99, "source_name"] == "Attribute 99"
        interface_rows = report.dielectric_interfaces.set_index("surface_index")
        assert interface_rows.loc[2, "source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert interface_rows.loc[2, "interface_type"] == "MA"
        assert interface_rows.loc[2, "loss_tangent"] == pytest.approx(0.0033)
        assert report.domain_energy.iloc[0]["source_name"] == "D1_SUBSTRATE"
        assert report.surface_q.iloc[0]["interface_type"] == "MA"
        domain_loss = report.domain_loss.set_index("domain_index")
        assert domain_loss.loc[1, "loss_tangent"] == pytest.approx(1.0e-6)
        assert domain_loss.loc[1, "inverse_q"] == pytest.approx(5.0e-7)
        assert domain_loss.loc[1, "q_equivalent"] == pytest.approx(2.0e6)
        assert domain_loss.loc[1, "frequency_ghz"] == pytest.approx(6.3)
        surface_loss = report.surface_loss.set_index("surface_index")
        assert surface_loss.loc[2, "interface_type"] == "MA"
        assert surface_loss.loc[2, "thickness"] == pytest.approx(0.002)
        assert surface_loss.loc[2, "permittivity"] == pytest.approx(10.0)
        assert surface_loss.loc[2, "loss_tangent"] == pytest.approx(0.0033)
        assert surface_loss.loc[2, "inverse_q"] == pytest.approx(5.0e-7)
        loss_budget = report.loss_budget.set_index("mode_index")
        assert loss_budget.loc[1, "domain_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert loss_budget.loc[1, "surface_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert loss_budget.loc[1, "total_inverse_q_sum"] == pytest.approx(1.0e-6)
        assert loss_budget.loc[1, "q_total"] == pytest.approx(1.0e6)
        assert (
            report.surface_interface_summary.set_index("interface_type").loc[
                "MA",
                "surface_count",
            ]
            == 1
        )
        assert report.port_epr.iloc[0]["source_name"] == "P1"
        assert report.index_map["entry_name"].tolist() == [
            "substrate",
            "ma_interface",
            "readout_port_surface",
        ]
        assert report.missing_reports == ()
        assert bool(report.sources.set_index("name").loc["surface-Q.csv", "loaded"])
        assert bool(report.sources.set_index("name").loc["config.json", "loaded"])

        assert isinstance(report.domain_epr_loss, DomainLoss)
        domain_metrics = report.domain_epr_loss.to_epr_dataframe().set_index(
            "source_index"
        )
        assert domain_metrics.loc[1, "channel"] == "domain"
        assert domain_metrics.loc[1, "participation"] == pytest.approx(0.5)
        assert domain_metrics.loc[1, "loss_tangent"] == pytest.approx(1.0e-6)
        assert domain_metrics.loc[1, "inverse_q"] == pytest.approx(5.0e-7)
        assert domain_metrics.loc[1, "gamma_hz"] == pytest.approx(6.3e9 * 5.0e-7)
        assert report.domain_epr_loss.to_dataframe().equals(report.domain_loss)
        assert domain_metrics.loc[1, "q_equivalent"] == pytest.approx(2.0e6)
        assert isinstance(report.loss, ReportLoss)
        assert isinstance(report.loss.budget, LossBudget)
        assert report.loss.domain.to_dataframe().equals(report.domain_loss)
        assert report.loss.surface.to_dataframe().equals(report.surface_loss)
        assert report.loss.budget.to_dataframe().equals(report.loss_budget)

        assert isinstance(report.surface_epr_loss, SurfaceLoss)
        surface_metrics = report.surface_epr_loss.to_epr_dataframe().set_index(
            "source_index"
        )
        assert surface_metrics.loc[2, "channel"] == "surface"
        assert surface_metrics.loc[2, "participation"] == pytest.approx(1.0e-7)
        assert surface_metrics.loc[2, "loss_tangent"] == pytest.approx(0.0033)
        assert surface_metrics.loc[2, "inverse_q"] == pytest.approx(5.0e-7)

    def test_load_eigenmode_report_allows_missing_optional_epr(
        self,
        eigenmode_dir: Path,
    ) -> None:
        report = load_eigenmode_report(eigenmode_dir)

        assert report.eigenmodes.n_modes == 2
        assert report.domain_materials.empty
        assert report.dielectric_interfaces.empty
        assert report.domain_energy.empty
        assert report.domain_loss.empty
        assert report.surface_q.empty
        assert report.surface_loss.empty
        assert report.loss_budget.empty
        assert report.port_epr.empty
        assert report.index_map.empty
        assert report.missing_reports == (
            "palace_index_map.json",
            "config.json",
            "domain-E.csv",
            "surface-Q.csv",
            "port-EPR.csv",
        )
        sources = report.sources.set_index("name")
        assert bool(sources.loc["eig.csv", "loaded"])
        assert not bool(sources.loc["config.json", "loaded"])
        assert not bool(sources.loc["domain-E.csv", "loaded"])

    def test_load_eigenmode_report_requires_bulk_and_surface_epr(
        self,
        eigenmode_dir: Path,
    ) -> None:
        with pytest.raises(
            FileNotFoundError,
            match=r"domain-E\.csv, surface-Q\.csv",
        ):
            load_eigenmode_report(eigenmode_dir, require_epr=True)


class TestIndexedCsv:
    """Tests for indexed Palace CSV loading through palace_index_map.json."""

    def test_loads_postprocessing_index_map(self, indexed_report_dir: Path) -> None:
        index_map = load_postprocessing_index_map(indexed_report_dir)

        entry = index_map.entry_for_index("Domains.Postprocessing.Energy", 1)
        assert entry is not None
        assert entry.primary_physical_name == "D1_SUBSTRATE"
        assert entry.metadata == {"material": "silicon"}

        interface_entry = index_map.entry_for_index(
            "Boundaries.Postprocessing.Dielectric",
            2,
        )
        assert interface_entry is not None
        assert interface_entry.extra["preset_name"] == "public_ma"
        assert interface_entry.extra["preset_source"] == "test source"

    def test_load_indexed_csv_renames_physical_columns(
        self, indexed_report_dir: Path
    ) -> None:
        result = load_indexed_csv(indexed_report_dir, "domain-E.csv")

        assert result.section == "Domains.Postprocessing.Energy"
        assert "E_elec[D1_SUBSTRATE] (J)" in result.dataframe.columns
        assert "p_elec[D1_SUBSTRATE]" in result.dataframe.columns
        assert "E_elec[99] (J)" in result.dataframe.columns
        assert result.dataframe.loc[0, "E_elec[D1_SUBSTRATE] (J)"] == pytest.approx(2.0)
        assert result.column_map[0]["physical_name"] == "D1_SUBSTRATE"
        assert result.column_map[0]["role"] == "dielectric_volume"
        assert result.column_map[2]["original_name"] == "E_elec[99] (J)"
        assert "physical_name" not in result.column_map[2]

    def test_load_indexed_csv_infers_surface_q_section(
        self, indexed_report_dir: Path
    ) -> None:
        result = load_indexed_csv(indexed_report_dir, "surface-Q.csv")

        assert result.section == "Boundaries.Postprocessing.Dielectric"
        assert "p_surf[MA:D1_TOP_M1___D1_SUBSTRATE]" in result.dataframe.columns
        assert result.columns[0].attributes == (20,)
        assert result.columns[0].extra["Type"] == "MA"

    def test_load_indexed_csv_accepts_results_dict(
        self, indexed_report_dir: Path
    ) -> None:
        results = {
            "domain-E.csv": indexed_report_dir / "results" / "palace" / "domain-E.csv",
            "palace_index_map.json": indexed_report_dir
            / "metadata"
            / "palace_index_map.json",
        }

        result = load_indexed_csv(results, "domain-E.csv")

        assert "E_elec[D1_SUBSTRATE] (J)" in result.dataframe.columns

    def test_load_indexed_csv_requires_unknown_section(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "custom.csv"
        csv_path.write_text("x[1]\n1\n")
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "palace_index_map.json").write_text(
            json.dumps({"schema_version": 1, "entries": []})
        )

        with pytest.raises(ValueError, match="section"):
            load_indexed_csv(csv_path)


class TestIndexedReportSummaries:
    """Tests for high-level indexed Palace report summary frames."""

    def test_load_dielectric_interface_summary_joins_config_to_index_map(
        self, indexed_report_dir: Path
    ) -> None:
        summary = load_dielectric_interface_summary(indexed_report_dir)

        by_index = summary.set_index("surface_index")
        assert by_index.loc[2, "section"] == "Boundaries.Postprocessing.Dielectric"
        assert by_index.loc[2, "source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert by_index.loc[2, "physical_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert by_index.loc[2, "entry_name"] == "ma_interface"
        assert by_index.loc[2, "role"] == "boundary_surface"
        assert by_index.loc[2, "interface_type"] == "MA"
        assert by_index.loc[2, "preset_name"] == "public_ma"
        assert by_index.loc[2, "preset_source"] == "test source"
        assert by_index.loc[2, "thickness"] == pytest.approx(0.002)
        assert by_index.loc[2, "permittivity"] == pytest.approx(10.0)
        assert by_index.loc[2, "loss_tangent"] == pytest.approx(0.0033)
        assert by_index.loc[2, "interface_material_name"] == "AlOx_native_generic"
        assert by_index.loc[2, "matched_material_name"] == "AlOx_native_generic"
        assert by_index.loc[2, "material_model_type"] == "constant"
        assert by_index.loc[2, "material_model_source"] == (
            "test PDK interface material"
        )
        assert bool(by_index.loc[2, "material_within_validity"])
        assert by_index.loc[2, "material_frequency_ghz"] == pytest.approx(5.0)

    def test_load_dielectric_interface_summary_keeps_unmapped_interfaces(
        self, indexed_report_dir: Path
    ) -> None:
        import pandas as pd

        summary = load_dielectric_interface_summary(indexed_report_dir)

        by_index = summary.set_index("surface_index")
        assert by_index.loc[99, "source_name"] == "Surface 99"
        assert pd.isna(by_index.loc[99, "physical_name"])
        assert by_index.loc[99, "surface_attributes"] == (199,)
        assert by_index.loc[99, "interface_type"] == "SA"
        assert by_index.loc[99, "permittivity"] == pytest.approx(4.0)
        assert by_index.loc[99, "loss_tangent"] == pytest.approx(0.0017)

    def test_load_domain_material_summary_joins_config_to_index_map(
        self, indexed_report_dir: Path
    ) -> None:
        summary = load_domain_material_summary(indexed_report_dir)

        by_attribute = summary.set_index("material_attribute")
        assert by_attribute.loc[10, "domain_index"] == 1
        assert by_attribute.loc[10, "section"] == "Domains.Postprocessing.Energy"
        assert by_attribute.loc[10, "source_name"] == "D1_SUBSTRATE"
        assert by_attribute.loc[10, "physical_name"] == "D1_SUBSTRATE"
        assert by_attribute.loc[10, "entry_name"] == "substrate"
        assert by_attribute.loc[10, "role"] == "dielectric_volume"
        assert by_attribute.loc[10, "material_name"] == "silicon"
        assert by_attribute.loc[10, "permittivity"] == pytest.approx(11.45)
        assert by_attribute.loc[10, "loss_tangent"] == pytest.approx(1.0e-6)
        assert by_attribute.loc[10, "conductivity"] == pytest.approx(2.0)
        assert by_attribute.loc[10, "volume_name"] == "substrate"
        assert by_attribute.loc[10, "stack_material_name"] == "Si"
        assert by_attribute.loc[10, "matched_material_name"] == "Si"
        assert by_attribute.loc[10, "material_model_type"] == "constant"
        assert by_attribute.loc[10, "material_model_source"] == (
            "test PDK material overlay"
        )
        assert bool(by_attribute.loc[10, "material_within_validity"])
        assert by_attribute.loc[10, "material_frequency_ghz"] == pytest.approx(5.0)

    def test_load_domain_material_summary_keeps_unmapped_material_attributes(
        self, indexed_report_dir: Path
    ) -> None:
        import pandas as pd

        summary = load_domain_material_summary(indexed_report_dir)

        by_attribute = summary.set_index("material_attribute")
        assert pd.isna(by_attribute.loc[99, "domain_index"])
        assert by_attribute.loc[99, "source_name"] == "Attribute 99"
        assert pd.isna(by_attribute.loc[99, "physical_name"])
        assert by_attribute.loc[99, "attributes"] == (99,)
        assert by_attribute.loc[99, "permittivity"] == pytest.approx(4.2)
        assert by_attribute.loc[99, "stack_material_name"] == "custom"
        assert pd.isna(by_attribute.loc[99, "matched_material_name"])
        assert "not found" in by_attribute.loc[99, "material_validity_note"]

    def test_load_domain_energy_summary_keeps_unmapped_indices(
        self, indexed_report_dir: Path
    ) -> None:
        summary = load_domain_energy_summary(indexed_report_dir)

        by_index = summary.set_index("domain_index")
        assert by_index.loc[1, "mode_index"] == 1
        assert by_index.loc[1, "source_name"] == "D1_SUBSTRATE"
        assert by_index.loc[1, "physical_name"] == "D1_SUBSTRATE"
        assert by_index.loc[1, "E_elec_j"] == pytest.approx(2.0)
        assert by_index.loc[1, "p_elec"] == pytest.approx(0.5)
        assert by_index.loc[99, "source_name"] == "Index 99"
        assert by_index.loc[99, "E_elec_j"] == pytest.approx(0.0)

    def test_summarize_domain_loss_joins_effective_material_loss(
        self, indexed_report_dir: Path
    ) -> None:
        domain_energy = load_domain_energy_summary(indexed_report_dir)
        domain_materials = load_domain_material_summary(indexed_report_dir)

        summary = summarize_domain_loss(
            domain_energy,
            domain_materials,
            frequency_ghz=5.0,
        )

        by_index = summary.set_index("domain_index")
        assert by_index.loc[1, "source_name"] == "D1_SUBSTRATE"
        assert by_index.loc[1, "material_name"] == "silicon"
        assert by_index.loc[1, "material_attribute"] == 10
        assert by_index.loc[1, "material_permittivity"] == pytest.approx(11.45)
        assert by_index.loc[1, "loss_tangent"] == pytest.approx(1.0e-6)
        assert by_index.loc[1, "p_elec"] == pytest.approx(0.5)
        assert by_index.loc[1, "inverse_q"] == pytest.approx(5.0e-7)
        assert by_index.loc[1, "q_equivalent"] == pytest.approx(2.0e6)
        assert by_index.loc[1, "gamma_hz"] == pytest.approx(5.0e9 * 5.0e-7)
        assert by_index.loc[1, "gamma_rad_per_s"] == pytest.approx(
            2.0 * np.pi * 5.0e9 * 5.0e-7
        )
        assert by_index.loc[1, "t1_us"] == pytest.approx(
            1.0e6 / by_index.loc[1, "gamma_rad_per_s"]
        )
        assert by_index.loc[99, "source_name"] == "Index 99"
        assert by_index.loc[99, "loss_tangent"] == pytest.approx(0.0)
        assert by_index.loc[99, "inverse_q"] == pytest.approx(0.0)

    def test_load_surface_q_summary_and_interface_totals(
        self, indexed_report_dir: Path
    ) -> None:
        summary = load_surface_q_summary(indexed_report_dir)

        row = summary.iloc[0]
        assert row["surface_index"] == 2
        assert row["interface_type"] == "MA"
        assert row["source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert row["p_surf"] == pytest.approx(1.0e-7)
        assert row["q_surf"] == pytest.approx(2.0e6)
        assert row["inverse_q"] == pytest.approx(5.0e-7)

        by_interface = summarize_surface_q_by_interface(summary).set_index(
            "interface_type"
        )
        assert by_interface.loc["MA", "surface_count"] == 1
        assert by_interface.loc["MA", "p_surf_sum"] == pytest.approx(1.0e-7)
        assert by_interface.loc["MA", "inverse_q_sum"] == pytest.approx(5.0e-7)
        assert by_interface.loc["MA", "q_equivalent"] == pytest.approx(2.0e6)
        assert by_interface.loc["MA", "p_surf_fraction"] == pytest.approx(1.0)
        assert by_interface.loc["MS", "surface_count"] == 0
        assert by_interface.loc["SA", "surface_count"] == 0

    def test_load_surface_q_summary_skips_native_mask_rows(
        self, tmp_path: Path
    ) -> None:
        palace_dir = tmp_path / "results" / "palace"
        palace_dir.mkdir(parents=True)
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        (metadata_dir / "palace_index_map.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "section": "Boundaries.Postprocessing.Dielectric",
                            "index": 1,
                            "entry_name": "native_mask_dielectric_ms_50nm",
                            "role": "boundary_surface",
                            "attributes": [4, 5],
                            "physical_names": ["MS__metal__substrate"],
                            "dimension": 2,
                            "Type": "MS",
                            "metadata": {"interface_type": "MS", "mask_margin_nm": 50},
                        },
                        {
                            "section": "Boundaries.Postprocessing.Dielectric",
                            "index": 2,
                            "entry_name": "regular_ma",
                            "role": "boundary_surface",
                            "attributes": [6],
                            "physical_names": ["MA__metal__air"],
                            "dimension": 2,
                            "Type": "MA",
                        },
                    ],
                }
            )
        )
        (palace_dir / "surface-Q.csv").write_text(
            "i, p_surf[1], Q_surf[1], p_surf[2], Q_surf[2]\n"
            "1, 1.0e-5, 1.0e5, 2.0e-7, 5.0e6\n"
        )

        summary = load_surface_q_summary(tmp_path)

        assert summary["surface_index"].tolist() == [2]
        assert summary["physical_name"].tolist() == ["MA__metal__air"]

    def test_summarize_surface_loss_joins_interface_parameters_and_rates(
        self, indexed_report_dir: Path
    ) -> None:
        surface_q = load_surface_q_summary(indexed_report_dir)
        dielectric_interfaces = load_dielectric_interface_summary(indexed_report_dir)

        summary = summarize_surface_loss(
            surface_q,
            dielectric_interfaces,
            frequency_ghz=5.0,
        )

        by_index = summary.set_index("surface_index")
        assert by_index.loc[2, "source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"
        assert by_index.loc[2, "interface_type"] == "MA"
        assert by_index.loc[2, "preset_name"] == "public_ma"
        assert by_index.loc[2, "preset_source"] == "test source"
        assert by_index.loc[2, "thickness"] == pytest.approx(0.002)
        assert by_index.loc[2, "permittivity"] == pytest.approx(10.0)
        assert by_index.loc[2, "loss_tangent"] == pytest.approx(0.0033)
        assert by_index.loc[2, "p_surf"] == pytest.approx(1.0e-7)
        assert by_index.loc[2, "q_surf"] == pytest.approx(2.0e6)
        assert by_index.loc[2, "inverse_q"] == pytest.approx(5.0e-7)
        assert by_index.loc[2, "q_equivalent"] == pytest.approx(2.0e6)
        assert by_index.loc[2, "gamma_hz"] == pytest.approx(5.0e9 * 5.0e-7)

    def test_summarize_surface_loss_carries_surface_epr_channel_metadata(
        self,
    ) -> None:
        import pandas as pd

        surface_q = pd.DataFrame(
            {
                "source_index": [1, 1, 1],
                "surface_index": [1, 2, 3],
                "p_surf": [1.0e-7, 2.0e-7, 1.0e-7],
                "q_surf": [2.0e6, 4.0e6, 1.0e6],
                "inverse_q": [5.0e-7, 2.5e-7, 1.0e-6],
            }
        )
        interfaces = pd.DataFrame(
            {
                "surface_index": [1, 2, 3],
                "loss_channel": ["MS", "SA", None],
            }
        )

        surface_loss = summarize_surface_loss(surface_q, interfaces)
        budget = summarize_loss_channel_budget(surface_loss)

        by_channel = budget.set_index("loss_channel")
        assert surface_loss.set_index("surface_index").loc[1, "loss_channel"] == "MS"
        assert by_channel.loc["MS", "inverse_q"] == pytest.approx(5.0e-7)
        assert by_channel.loc["SA", "inverse_q"] == pytest.approx(2.5e-7)
        assert by_channel.loc["MS", "loss_fraction"] == pytest.approx(2.0 / 3.0)
        assert by_channel.loc["SA", "loss_fraction"] == pytest.approx(1.0 / 3.0)
        assert set(budget["loss_channel"]) == {"MS", "SA"}

    def test_source_aware_surface_epr_budget_uses_interface_rows(
        self,
    ) -> None:
        import pandas as pd

        surface_q = pd.DataFrame(
            {
                "surface_index": [1, 2, 3],
                "p_surf": [1.0e-7, 9.0e-7, 2.0e-7],
                "q_surf": [1.0e7, 1.0e6, 5.0e6],
                "inverse_q": [1.0e-7, 9.0e-7, 2.0e-7],
            }
        )
        interfaces = pd.DataFrame(
            {
                "surface_index": [1, 2, 3],
                "source_entry_name": [
                    "D0_TOP_M1_pec_0",
                    "D0_TOP_M1_pec_0",
                    "D0_TOP_M1_pec_0",
                ],
                "loss_channel": ["MS", "MS", "MA"],
            }
        )

        surface_loss = summarize_surface_loss(surface_q, interfaces)
        channel_budget = summarize_loss_channel_budget(surface_loss)
        loss_budget = summarize_loss_budget(
            pd.DataFrame(),
            surface_loss,
            frequency_ghz=5.0,
        )

        surface_rows = surface_loss.set_index("surface_index")
        assert surface_rows.loc[1, "source_entry_name"] == "D0_TOP_M1_pec_0"

        by_channel = channel_budget.set_index("loss_channel")
        assert set(channel_budget["loss_channel"]) == {"MS", "MA"}
        assert by_channel.loc["MS", "inverse_q"] == pytest.approx(1.0e-6)
        assert by_channel.loc["MA", "inverse_q"] == pytest.approx(2.0e-7)
        assert loss_budget.iloc[0]["surface_inverse_q_sum"] == pytest.approx(1.2e-6)

    def test_summarize_loss_budget_combines_domain_and_surface_loss(
        self, eigenmode_report_dir: Path
    ) -> None:
        modes = load_eigenmodes(eigenmode_report_dir)
        domain_loss = summarize_domain_loss(
            load_domain_energy_summary(eigenmode_report_dir),
            load_domain_material_summary(eigenmode_report_dir),
            modes=modes,
        )
        surface_loss = summarize_surface_loss(
            load_surface_q_summary(eigenmode_report_dir),
            load_dielectric_interface_summary(eigenmode_report_dir),
            modes=modes,
        )

        budget = summarize_loss_budget(domain_loss, surface_loss, modes=modes)

        row = budget.set_index("mode_index").loc[1]
        assert row["frequency_ghz"] == pytest.approx(6.3)
        assert row["q_eig"] == pytest.approx(110.0)
        assert row["inverse_q_eig"] == pytest.approx(1.0 / 110.0)
        assert row["domain_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert row["surface_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert row["total_inverse_q_sum"] == pytest.approx(1.0e-6)
        assert row["eig_with_surface_inverse_q_sum"] == pytest.approx(
            1.0 / 110.0 + 5.0e-7
        )
        assert row["q_total"] == pytest.approx(1.0e6)
        assert row["q_eig_with_surface"] == pytest.approx(1.0 / (1.0 / 110.0 + 5.0e-7))
        assert row["domain_vs_eig_relative_error"] == pytest.approx(
            (5.0e-7 - 1.0 / 110.0) / (1.0 / 110.0)
        )

    def test_summarize_loss_budget_keeps_modes_separate(
        self, indexed_report_dir: Path
    ) -> None:
        palace_dir = indexed_report_dir / "results" / "palace"
        _write_eig_csv(
            palace_dir / "eig.csv",
            [
                [1, 5.0, 0.0, 2.0e6, 0.0, 0.0],
                [2, 6.0, 0.0, 4.0e6, 0.0, 0.0],
            ],
        )
        (palace_dir / "domain-E.csv").write_text(
            "m, E_elec[1] (J), p_elec[1]\n1, 2.0, 0.5\n2, 1.0, 0.25\n"
        )
        (palace_dir / "surface-Q.csv").write_text(
            "m, p_surf[2], Q_surf[2]\n1, 1.0e-7, 2.0e6\n2, 2.0e-7, 4.0e6\n"
        )

        modes = load_eigenmodes(indexed_report_dir)
        domain_loss = summarize_domain_loss(
            load_domain_energy_summary(indexed_report_dir),
            load_domain_material_summary(indexed_report_dir),
            modes=modes,
        )
        surface_loss = summarize_surface_loss(
            load_surface_q_summary(indexed_report_dir),
            load_dielectric_interface_summary(indexed_report_dir),
            modes=modes,
        )
        budget = summarize_loss_budget(domain_loss, surface_loss, modes=modes)

        by_mode = budget.set_index("mode_index")
        assert by_mode.loc[1, "frequency_ghz"] == pytest.approx(5.0)
        assert by_mode.loc[1, "domain_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert by_mode.loc[1, "surface_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert by_mode.loc[1, "total_inverse_q_sum"] == pytest.approx(1.0e-6)
        assert by_mode.loc[2, "frequency_ghz"] == pytest.approx(6.0)
        assert by_mode.loc[2, "domain_inverse_q_sum"] == pytest.approx(2.5e-7)
        assert by_mode.loc[2, "surface_inverse_q_sum"] == pytest.approx(2.5e-7)
        assert by_mode.loc[2, "total_inverse_q_sum"] == pytest.approx(5.0e-7)

    def test_load_port_epr_summary_tracks_signed_and_abs_participation(
        self, indexed_report_dir: Path
    ) -> None:
        summary = load_port_epr_summary(indexed_report_dir)

        row = summary.iloc[0]
        assert row["port_index"] == 3
        assert row["mode_index"] == 1
        assert row["source_name"] == "P1"
        assert row["postprocessing_type"] == "Power"
        assert row["p_port"] == pytest.approx(-2.5e-4)
        assert row["abs_p_port"] == pytest.approx(2.5e-4)
        assert row["abs_p_port_fraction"] == pytest.approx(1.0)

    def test_summary_helpers_accept_results_dict(
        self, indexed_report_dir: Path
    ) -> None:
        results = {
            "surface-Q.csv": indexed_report_dir
            / "results"
            / "palace"
            / "surface-Q.csv",
            "palace_index_map.json": indexed_report_dir
            / "metadata"
            / "palace_index_map.json",
        }

        summary = load_surface_q_summary(results)

        assert summary.iloc[0]["source_name"] == "MA:D1_TOP_M1___D1_SUBSTRATE"


class TestElectrostaticReport:
    """Tests for composed Palace electrostatic report bundles."""

    def test_load_electrostatic_report_composes_loss_without_t1(
        self,
        electrostatic_report_dir: Path,
    ) -> None:
        report = load_electrostatic_report(electrostatic_report_dir)

        assert isinstance(report, ElectrostaticReport)
        assert report.capacitance.terminal_names == ("left", "right")
        assert report.mutual_capacitance is not None
        assert report.inverse_capacitance is not None
        assert report.terminal_c_pass_summary.iloc[0]["n_elements"] == 4
        material_rows = report.domain_materials.set_index("material_attribute")
        assert material_rows.loc[10, "source_name"] == "substrate"
        assert material_rows.loc[10, "loss_tangent"] == pytest.approx(1.0e-6)
        interface_rows = report.dielectric_interfaces.set_index("surface_index")
        assert interface_rows.loc[2, "source_name"] == "MA:top_metal__substrate"
        assert interface_rows.loc[2, "loss_tangent"] == pytest.approx(0.0033)

        domain_loss = report.domain_loss.set_index("source_index")
        assert domain_loss.loc[1, "source_name"] == "substrate"
        assert domain_loss.loc[1, "inverse_q"] == pytest.approx(5.0e-7)
        assert domain_loss.loc[2, "inverse_q"] == pytest.approx(2.5e-7)
        assert "t1_us" not in report.domain_loss.columns

        surface_loss = report.surface_loss.set_index("source_index")
        assert surface_loss.loc[1, "source_name"] == "MA:top_metal__substrate"
        assert surface_loss.loc[1, "inverse_q"] == pytest.approx(5.0e-7)
        assert surface_loss.loc[2, "inverse_q"] == pytest.approx(2.5e-7)
        assert "t1_us" not in report.surface_loss.columns

        domain_metrics = report.domain_epr_loss.to_epr_dataframe().set_index(
            "source_index"
        )
        assert domain_metrics.loc[1, "participation"] == pytest.approx(0.5)
        assert domain_metrics.loc[1, "t1_us"] is None
        surface_metrics = report.surface_epr_loss.to_epr_dataframe().set_index(
            "source_index"
        )
        assert surface_metrics.loc[1, "participation"] == pytest.approx(1.0e-7)
        assert surface_metrics.loc[1, "t1_us"] is None

        budget = report.loss_budget.set_index("source_index")
        assert budget.loc[1, "domain_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert budget.loc[1, "surface_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert budget.loc[1, "total_inverse_q_sum"] == pytest.approx(1.0e-6)
        assert budget.loc[2, "total_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert "t1_us" not in report.loss_budget.columns
        assert isinstance(report.loss, ReportLoss)
        assert report.loss.domain.to_dataframe().equals(report.domain_loss)
        assert report.loss.surface.to_dataframe().equals(report.surface_loss)
        assert report.loss.budget.to_dataframe().equals(report.loss_budget)

        by_interface = report.surface_interface_summary.set_index("interface_type")
        assert by_interface.loc["MA", "surface_count"] == 2
        assert by_interface.loc["MA", "inverse_q_sum"] == pytest.approx(7.5e-7)
        assert report.missing_reports == ()
        assert bool(report.sources.set_index("name").loc["config.json", "loaded"])

    def test_load_electrostatic_report_adds_t1_for_explicit_frequency(
        self,
        electrostatic_report_dir: Path,
    ) -> None:
        report = load_electrostatic_report(electrostatic_report_dir, frequency_ghz=5.0)

        domain_row = report.domain_loss.set_index("source_index").loc[1]
        assert domain_row["gamma_hz"] == pytest.approx(5.0e9 * 5.0e-7)
        assert domain_row["gamma_rad_per_s"] == pytest.approx(
            2.0 * np.pi * 5.0e9 * 5.0e-7
        )
        assert domain_row["t1_us"] == pytest.approx(
            1.0e6 / domain_row["gamma_rad_per_s"]
        )

        budget_row = report.loss_budget.set_index("source_index").loc[1]
        assert budget_row["gamma_hz"] == pytest.approx(5.0e9 * 1.0e-6)
        assert budget_row["t1_us"] == pytest.approx(
            1.0e6 / (2.0 * np.pi * 5.0e9 * 1.0e-6)
        )

    def test_load_electrostatic_report_allows_missing_optional_epr(
        self,
        terminal_matrix_dir: Path,
    ) -> None:
        report = load_electrostatic_report(terminal_matrix_dir)

        assert report.capacitance.terminal_names == ("left", "right")
        assert report.domain_materials.empty
        assert report.dielectric_interfaces.empty
        assert report.domain_energy.empty
        assert report.domain_loss.empty
        assert report.surface_q.empty
        assert report.surface_loss.empty
        assert report.loss_budget.empty
        assert report.missing_reports == (
            "config.json",
            "domain-E.csv",
            "surface-Q.csv",
        )

    def test_load_electrostatic_report_requires_bulk_and_surface_epr(
        self,
        terminal_matrix_dir: Path,
    ) -> None:
        with pytest.raises(
            FileNotFoundError,
            match=r"domain-E\.csv, surface-Q\.csv",
        ):
            load_electrostatic_report(terminal_matrix_dir, require_epr=True)

    def test_load_electrostatic_report_rejects_invalid_frequency(
        self,
        electrostatic_report_dir: Path,
    ) -> None:
        with pytest.raises(ValueError, match="frequency_ghz"):
            load_electrostatic_report(electrostatic_report_dir, frequency_ghz=0.0)

    def test_load_electrostatic_report_reads_iteration_history_from_run_folder(
        self,
        electrostatic_report_dir: Path,
    ) -> None:
        palace_dir = electrostatic_report_dir / "results" / "palace"
        metadata_dir = electrostatic_report_dir / "metadata"
        index_map_path = metadata_dir / "palace_index_map.json"
        index_map = json.loads(index_map_path.read_text())
        for entry in index_map["entries"]:
            if (
                entry["section"] == "Boundaries.Postprocessing.Dielectric"
                and entry["index"] == 2
            ):
                entry["metadata"] = {
                    "source_entry_name": "left",
                }
        index_map["entries"].append(
            {
                "section": "Boundaries.Postprocessing.Dielectric",
                "index": 3,
                "entry_name": "ma_total_right",
                "role": "boundary_surface",
                "attributes": [21],
                "physical_names": ["MA:right_metal__substrate"],
                "dimension": 2,
                "metadata": {
                    "source_entry_name": "right",
                },
            }
        )
        index_map_path.write_text(json.dumps(index_map))
        config_path = electrostatic_report_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["Boundaries"]["Postprocessing"]["Dielectric"].append(
            {
                "Index": 3,
                "Attributes": [21],
                "Type": "MA",
                "Thickness": 0.002,
                "Permittivity": 10.0,
                "LossTan": 0.0033,
            }
        )
        config_path.write_text(json.dumps(config))
        iteration01 = palace_dir / "iteration01"
        iteration01.mkdir()
        _write_terminal_matrix_csv(
            iteration01 / "terminal-C.csv",
            "C",
            [
                [0.5e-15, -1.0e-15],
                [-1.0e-15, 2.0e-15],
            ],
        )
        (iteration01 / "domain-E.csv").write_text(
            "i, E_elec[1] (J), p_elec[1]\n1, 1.0, 0.4\n"
        )
        (iteration01 / "surface-Q.csv").write_text(
            "i, p_surf[2], Q_surf[2], p_surf[3], Q_surf[3]\n"
            "1, 1.0e-7, 1.0e7, 2.0e-7, 5.0e6\n"
        )
        iteration02 = palace_dir / "iteration02"
        iteration02.mkdir()
        (iteration02 / "domain-E.csv").write_text(
            "i, E_elec[1] (J), p_elec[1]\n1, 1.5, 0.45\n"
        )
        (iteration02 / "surface-Q.csv").write_text(
            "i, p_surf[2], Q_surf[2], p_surf[3], Q_surf[3]\n"
            "1, 2.0e-7, 5.0e6, 3.0e-7, 3.333333e6\n"
        )

        report = load_electrostatic_report(electrostatic_report_dir)

        assert report.terminal_c_history["pass_index"].drop_duplicates().tolist() == [
            1,
            2,
        ]
        assert set(report.terminal_c_history["label"]) == {"Pass 1", "Final"}
        sources = report.sources.set_index("name")
        assert bool(sources.loc["iteration*/terminal-C.csv", "loaded"])
        assert "loaded 1 AMR iteration files" in str(
            sources.loc["iteration*/terminal-C.csv", "message"]
        )
        assert bool(sources.loc["iteration*/domain-E.csv", "loaded"])
        assert bool(sources.loc["iteration*/surface-Q.csv", "loaded"])
        domain_convergence = report.domain_epr_convergence
        assert domain_convergence.columns.tolist() == [
            "pass_index",
            "source_index",
            "sample_column",
            "sample_value",
            "domain_index",
            "source_name",
            "physical_name",
            "domain_epr_abs",
        ]
        assert domain_convergence["pass_index"].tolist() == [1, 2, 3, 3]
        final_domain = domain_convergence.loc[
            domain_convergence["pass_index"] == 3
        ].set_index("source_index")
        assert final_domain.loc[1, "domain_epr_abs"] == pytest.approx(0.5)
        assert final_domain.loc[2, "domain_epr_abs"] == pytest.approx(0.25)

        convergence = report.surface_epr_convergence
        assert convergence.columns.tolist() == [
            "pass_index",
            "source_index",
            "sample_column",
            "sample_value",
            "surface_index",
            "source_name",
            "physical_name",
            "entry_name",
            "interface_type",
            "surface_epr_abs",
        ]
        by_surface = convergence.set_index(
            ["surface_index", "pass_index", "sample_value"]
        )
        assert by_surface.loc[(2, 1, 1.0), "source_name"] == "left"
        assert by_surface.loc[(3, 1, 1.0), "source_name"] == "right"
        assert by_surface.loc[(2, 1, 1.0), "surface_epr_abs"] == pytest.approx(1.0e-7)
        assert by_surface.loc[(3, 1, 1.0), "surface_epr_abs"] == pytest.approx(2.0e-7)
        assert by_surface.loc[(2, 2, 1.0), "surface_epr_abs"] == pytest.approx(2.0e-7)
        assert by_surface.loc[(3, 2, 1.0), "surface_epr_abs"] == pytest.approx(3.0e-7)

    def test_electrostatic_report_loader_stays_assembly_owned(self) -> None:
        import gsim.palace as palace
        import gsim.palace.resolve as resolve
        import gsim.palace.resolve.assembly as assembly

        assert palace.ElectrostaticReport is ElectrostaticReport
        assert assembly.load_electrostatic_report is load_electrostatic_report
        assert not hasattr(resolve, "load_electrostatic_report")
        assert not hasattr(palace, "load_electrostatic_report")


class TestTerminalMatrix:
    """Tests for electrostatic terminal matrix loading through index maps."""

    def test_load_terminal_matrix_names_from_index_map(
        self, terminal_matrix_dir: Path
    ) -> None:
        matrix = load_terminal_matrix(terminal_matrix_dir, "C")

        assert matrix.matrix_kind == "C"
        assert matrix.terminal_names == ("left", "right")
        assert list(matrix.dataframe.index) == ["left", "right"]
        assert list(matrix.dataframe.columns) == ["left", "right"]
        assert matrix.dataframe.loc["left", "right"] == pytest.approx(-2.0e-15)
        assert matrix.dataframe.attrs["source_unit"] == "F"
        assert matrix.display_dataframe.loc["left", "left"] == pytest.approx(1.0)
        assert matrix.display_dataframe.attrs["display_unit"] == "fF"

        table = matrix.to_long_dataframe()
        left_right = table.set_index("element").loc["left -> right"]
        assert left_right["row_index"] == 1
        assert left_right["column_index"] == 2
        assert not left_right["is_diagonal"]
        assert left_right["value_si"] == pytest.approx(-2.0e-15)
        assert left_right["display_value"] == pytest.approx(-2.0)

    def test_load_terminal_matrix_accepts_results_dict(
        self, terminal_matrix_dir: Path
    ) -> None:
        results = {
            "terminal-Cm.csv": terminal_matrix_dir
            / "results"
            / "palace"
            / "terminal-Cm.csv",
            "palace_index_map.json": terminal_matrix_dir
            / "metadata"
            / "palace_index_map.json",
        }

        matrix = load_terminal_matrix(results, "Cm")

        assert matrix.matrix_kind == "Cm"
        assert matrix.dataframe.loc["left", "right"] == pytest.approx(2.0e-15)
        assert matrix.display_dataframe.loc["left", "right"] == pytest.approx(2.0)

    def test_load_terminal_matrix_supports_cinv_units(
        self, terminal_matrix_dir: Path
    ) -> None:
        matrix = load_terminal_matrix(terminal_matrix_dir, "Cinv")

        assert matrix.source_unit == "1/F"
        assert matrix.display_unit == "1/F"
        assert matrix.display_dataframe.loc["right", "right"] == pytest.approx(4.0e15)

    def test_load_terminal_matrix_accepts_explicit_terminal_names(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "terminal-C.csv"
        _write_terminal_matrix_csv(
            csv_path,
            "C",
            [
                [1.0e-15, 0.0],
                [0.0, 2.0e-15],
            ],
        )

        matrix = load_terminal_matrix(csv_path, terminal_names=("top", "bottom"))

        assert matrix.terminal_names == ("top", "bottom")
        assert matrix.display_dataframe.loc["bottom", "bottom"] == pytest.approx(2.0)

    def test_load_terminal_matrix_rejects_terminal_label_mismatch(
        self, terminal_matrix_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="Terminal label count"):
            load_terminal_matrix(
                terminal_matrix_dir,
                "C",
                terminal_names=("left", "right", "readout"),
            )


class TestTerminalMatrixHistory:
    """Tests for electrostatic terminal matrix convergence histories."""

    def test_history_deduplicates_matching_final_and_summarizes(
        self, terminal_matrix_dir: Path
    ) -> None:
        palace_dir = terminal_matrix_dir / "results" / "palace"
        iteration01 = palace_dir / "iteration01"
        iteration02 = palace_dir / "iteration02"
        iteration01.mkdir()
        iteration02.mkdir()
        pass1 = [
            [1.0e-15, -0.1e-15],
            [-0.1e-15, 2.0e-15],
        ]
        pass2 = [
            [1.5e-15, -0.2e-15],
            [-0.2e-15, 3.0e-15],
        ]
        _write_terminal_matrix_csv(iteration01 / "terminal-C.csv", "C", pass1)
        _write_terminal_matrix_csv(iteration02 / "terminal-C.csv", "C", pass2)
        _write_terminal_matrix_csv(palace_dir / "terminal-C.csv", "C", pass2)

        history = load_terminal_matrix_history(palace_dir, "C")

        assert history["pass_index"].drop_duplicates().tolist() == [1, 2]
        assert len(history) == 8
        left_left_pass2 = history.loc[
            (history["pass_index"] == 2)
            & (history["row_terminal"] == "left")
            & (history["column_terminal"] == "left")
        ].iloc[0]
        assert left_left_pass2["display_value"] == pytest.approx(1.5)
        assert left_left_pass2["display_delta_to_previous"] == pytest.approx(0.5)
        assert left_left_pass2["display_delta_to_final"] == pytest.approx(0.0)

        summary = summarize_terminal_matrix_history(history)
        pass2_summary = summary.loc[summary["pass_index"] == 2].iloc[0]
        assert pass2_summary["n_elements"] == 4
        assert pass2_summary["n_diagonal_elements"] == 2
        assert pass2_summary["n_off_diagonal_elements"] == 2
        assert pass2_summary["max_abs_display_delta_to_previous"] == pytest.approx(1.0)

    def test_history_accepts_run_folder_not_only_palace_output_dir(
        self, terminal_matrix_dir: Path
    ) -> None:
        palace_dir = terminal_matrix_dir / "results" / "palace"
        iteration01 = palace_dir / "iteration01"
        iteration02 = palace_dir / "iteration02"
        iteration01.mkdir()
        iteration02.mkdir()
        _write_terminal_matrix_csv(
            iteration01 / "terminal-C.csv",
            "C",
            [
                [1.0e-15, -0.1e-15],
                [-0.1e-15, 2.0e-15],
            ],
        )
        _write_terminal_matrix_csv(
            iteration02 / "terminal-C.csv",
            "C",
            [
                [1.5e-15, -0.2e-15],
                [-0.2e-15, 3.0e-15],
            ],
        )

        history = load_terminal_matrix_history(terminal_matrix_dir, "C")

        assert history["pass_index"].drop_duplicates().tolist() == [1, 2, 3]
        assert set(history["label"]) == {"Pass 1", "Pass 2", "Final"}
        assert history["row_terminal"].drop_duplicates().tolist() == ["left", "right"]

    def test_history_appends_nonmatching_final(self, terminal_matrix_dir: Path) -> None:
        palace_dir = terminal_matrix_dir / "results" / "palace"
        iteration01 = palace_dir / "iteration01"
        iteration01.mkdir()
        _write_terminal_matrix_csv(
            iteration01 / "terminal-Cm.csv",
            "Cm",
            [
                [0.0, 1.0e-15],
                [1.0e-15, 0.0],
            ],
        )

        history = load_terminal_matrix_history(palace_dir, "Cm")

        assert history["pass_index"].drop_duplicates().tolist() == [1, 2]
        final_rows = history.loc[history["is_final"]]
        assert len(final_rows) == 4
        assert set(final_rows["label"]) == {"Final"}
        final_left_right = final_rows.loc[
            (final_rows["row_terminal"] == "left")
            & (final_rows["column_terminal"] == "right")
        ].iloc[0]
        assert final_left_right["display_value"] == pytest.approx(2.0)
        assert final_left_right["display_delta_to_previous"] == pytest.approx(1.0)


class TestSParamsSave:
    """Tests for SParams NumPy export."""

    def test_save_npz_exports_numpy_artifact(
        self,
        sim_dir: Path,
        tmp_path: Path,
    ) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "cached")
        assert out.suffix == ".npz"
        assert out.exists()

        exported = np.load(out, allow_pickle=False)
        assert list(exported["port_names"]) == sp.port_names
        np.testing.assert_allclose(exported["freq"], sp.freq)
        for key in sp._data:
            to_port, from_port = key
            np.testing.assert_allclose(
                exported[f"S_{to_port}_{from_port}_db"],
                sp[key].db,
            )
            np.testing.assert_allclose(
                exported[f"S_{to_port}_{from_port}_deg"],
                sp[key].deg,
            )

    def test_adds_npz_suffix(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "no_ext")
        assert out.name == "no_ext.npz"


def _write_terminal_matrix_csv(
    path: Path,
    matrix_kind: str,
    values: list[list[float]],
) -> None:
    headers = {
        "C": "C[i][{index}] (F)",
        "Cm": "C_m[i][{index}] (F)",
        "Cinv": "C_inv[i][{index}] (1/F)",
    }
    lines = [
        ",".join(
            ["i"]
            + [
                headers[matrix_kind].format(index=index)
                for index in range(1, len(values) + 1)
            ]
        )
    ]
    for row_index, row in enumerate(values, start=1):
        lines.append(
            ",".join([f"{float(row_index):.2e}"] + [str(value) for value in row])
        )
    path.write_text("\n".join(lines) + "\n")


def _write_eig_csv(path: Path, rows: list[list[float]]) -> None:
    lines = [
        "m, Re{f} (GHz), Im{f} (GHz), Q, Error (Bkwd.), Error (Abs.)",
    ]
    lines.extend(", ".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n")
