"""Tests for gsim.palace.results — S-parameter loading with port-name mapping."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import (
    DrivenReport,
    EigenmodeReport,
    Eigenmodes,
    ElectrostaticReport,
    PalaceRunSummary,
    PalaceSweepSummary,
    SParams,
    get_port_map,
    load_dielectric_interface_summary,
    load_domain_energy_summary,
    load_domain_material_summary,
    load_driven_report,
    load_eigenmode_history,
    load_eigenmode_report,
    load_eigenmodes,
    load_electrostatic_report,
    load_indexed_csv,
    load_palace_run_summary,
    load_palace_sweep_summary,
    load_port_epr_summary,
    load_postprocessing_index_map,
    load_sparams,
    load_surface_q_summary,
    load_terminal_matrix,
    load_terminal_matrix_history,
    summarize_domain_loss,
    summarize_eigenmode_history,
    summarize_loss_budget,
    summarize_surface_loss,
    summarize_surface_q_by_interface,
    summarize_terminal_matrix_history,
)


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace output directory."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    port_info = {
        "ports": [
            {"portnumber": 1, "name": "o1", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "name": "o2", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 3, "name": "o3", "Z0": 50.0, "type": "lumped"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))

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
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

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
    (tmp_path / "palace_index_map.json").write_text(json.dumps(index_map))
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
    (tmp_path / "palace_material_resolution.json").write_text(
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
    result_dir = result_dir or run_dir / "output" / "palace"
    result_dir.mkdir(parents=True)

    config = json.loads((source / "config.json").read_text())
    config["Problem"] = {"Type": "Eigenmode"}
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / mesh_name).write_text("$MeshFormat\n")
    (run_dir / "mesh_manifest.json").write_text(
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
        shutil.copy(source / name, run_dir / name)
    (result_dir / "palace_run_metadata.json").write_text(
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
    for csv_path in (source / "output" / "palace").glob("*.csv"):
        shutil.copy(csv_path, result_dir / csv_path.name)


@pytest.fixture
def driven_report_dir(indexed_report_dir: Path) -> Path:
    """Create a Palace driven report output with S-parameters and port EPR."""
    palace_dir = indexed_report_dir / "output" / "palace"
    port_info = {
        "ports": [
            {"portnumber": 1, "name": "readout", "Z0": 50.0, "type": "cpw"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (indexed_report_dir / "port_information.json").write_text(json.dumps(port_info))
    (palace_dir / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.)\n"
        "5.0, -12.0, -33.0\n"
        "6.0, -6.0, -45.0\n"
    )
    return indexed_report_dir


@pytest.fixture
def eigenmode_report_dir(indexed_report_dir: Path) -> Path:
    """Create a Palace eigenmode report output with AMR and EPR tables."""
    palace_dir = indexed_report_dir / "output" / "palace"
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
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

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
    (tmp_path / "palace_index_map.json").write_text(json.dumps(index_map))
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
    palace_dir = terminal_matrix_dir / "output" / "palace"
    index_map_path = terminal_matrix_dir / "palace_index_map.json"
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
    palace_dir = tmp_path / "output" / "palace"
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
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    port_info = {
        "ports": [
            {"portnumber": 1, "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "Z0": 50.0, "type": "cpw"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))

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

    def test_plot_interactive_runs(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        fig = sp.plot_interactive()
        assert len(fig.data) == len(sp.keys())

    def test_plot_interactive_labels(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        fig = sp.plot_interactive()
        names = [trace.name for trace in fig.data]
        # 3 ports -> S11, S21, S31 etc.
        assert "S11" in names
        assert "S21" in names

    def test_plot_interactive_visibility(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        fig = sp.plot_interactive()
        # First excitation column (Si1) should be visible
        for trace in fig.data:
            if trace.name.endswith("1"):  # S11, S21, S31
                assert trace.visible is True
            else:
                assert trace.visible == "legendonly"


class TestLoadSparamsSource:
    """Tests for source resolution (dir, subdir, dict)."""

    def test_accepts_dir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert len(sp.freq) == 2

    def test_accepts_palace_subdir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir / "output" / "palace")
        assert len(sp.freq) == 2

    def test_accepts_results_dict(self, sim_dir: Path) -> None:
        results = {
            "port-S.csv": sim_dir / "output" / "palace" / "port-S.csv",
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


class TestGetPortMap:
    """Tests for get_port_map."""

    def test_returns_mapping(self, sim_dir: Path) -> None:
        pm = get_port_map(sim_dir)
        assert pm == {1: "o1", 2: "o2", 3: "o3"}

    def test_legacy_numeric_fallback(self, sim_dir_no_names: Path) -> None:
        pm = get_port_map(sim_dir_no_names)
        assert pm == {1: "p1", 2: "p2"}


class TestPalaceRunSummary:
    """Tests for reusable Palace run artifact summaries."""

    def test_load_palace_run_summary_records_handoff_and_results(
        self,
        indexed_report_dir: Path,
    ) -> None:
        config_path = indexed_report_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["Problem"] = {"Type": "Eigenmode"}
        config_path.write_text(json.dumps(config))

        (indexed_report_dir / "palace.msh").write_text("$MeshFormat\n")
        (indexed_report_dir / "mesh_manifest.json").write_text(
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
        (indexed_report_dir / "palace_run_metadata.json").write_text(
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
            "domain-E.csv": indexed_report_dir / "output" / "palace" / "domain-E.csv",
            "config.json": config_path,
            "palace_index_map.json": indexed_report_dir / "palace_index_map.json",
            "palace_material_resolution.json": indexed_report_dir
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


class TestPalaceSweepSummary:
    """Tests for reusable point-local Palace sweep summaries."""

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
                            "parameters": {"gap_um": 6.0},
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
        assert point.parameters == {"gap_um": 6.0}
        assert point.run_summary.results["domain-E.csv"].present
        assert point.run_summary.runtime["status"] == "completed"

        as_dict = summary.to_dict()
        assert as_dict["point_count"] == 1
        assert as_dict["points"][0]["missing_artifacts"] == []

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
        assert report.port_epr.iloc[0]["source_name"] == "P1"
        assert report.port_epr.iloc[0]["p_port"] == pytest.approx(-2.5e-4)
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
        assert bool(sources.loc["port-EPR.csv", "loaded"])

    def test_load_driven_report_allows_missing_optional_reports(
        self,
        sim_dir: Path,
    ) -> None:
        report = load_driven_report(sim_dir)

        assert report.sparams.port_names == ["o1", "o2", "o3"]
        assert report.port_epr.empty
        assert report.domain_materials.empty
        assert report.dielectric_interfaces.empty
        assert report.index_map.empty
        assert report.missing_reports == (
            "palace_index_map.json",
            "config.json",
            "port-EPR.csv",
        )
        sources = report.sources.set_index("name")
        assert bool(sources.loc["port-S.csv", "loaded"])
        assert not bool(sources.loc["port-EPR.csv", "loaded"])

    def test_load_driven_report_can_require_port_epr(
        self,
        sim_dir: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="port-EPR"):
            load_driven_report(sim_dir, require_port_epr=True)


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
            "eig.csv": eigenmode_dir / "output" / "palace" / "eig.csv",
        }

        eigenmodes = load_eigenmodes(results)

        assert eigenmodes.source_path == results["eig.csv"]
        np.testing.assert_allclose(eigenmodes.freq_real_ghz, [6.1, 7.2])

    def test_load_eigenmodes_accepts_csv_path(self, eigenmode_dir: Path) -> None:
        csv_path = eigenmode_dir / "output" / "palace" / "eig.csv"

        eigenmodes = load_eigenmodes(csv_path)

        assert eigenmodes.source_path == csv_path
        assert eigenmodes.n_modes == 2

    def test_load_eigenmodes_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"eig\.csv"):
            load_eigenmodes(tmp_path)

    def test_load_eigenmodes_fills_optional_columns(self, tmp_path: Path) -> None:
        palace_dir = tmp_path / "output" / "palace"
        palace_dir.mkdir(parents=True)
        (palace_dir / "eig.csv").write_text("m, Re{f} (GHz)\n1, 5.5\n")

        eigenmodes = load_eigenmodes(tmp_path)

        assert eigenmodes.freq_imag_ghz.tolist() == [0.0]
        assert np.isnan(eigenmodes.q[0])

    def test_load_eigenmode_history_deduplicates_final_and_summarizes(
        self,
        tmp_path: Path,
    ) -> None:
        palace_dir = tmp_path / "output" / "palace"
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
        palace_dir = tmp_path / "output" / "palace"
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
            "domain-E.csv": indexed_report_dir / "output" / "palace" / "domain-E.csv",
            "palace_index_map.json": indexed_report_dir / "palace_index_map.json",
        }

        result = load_indexed_csv(results, "domain-E.csv")

        assert "E_elec[D1_SUBSTRATE] (J)" in result.dataframe.columns

    def test_load_indexed_csv_requires_unknown_section(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "custom.csv"
        csv_path.write_text("x[1]\n1\n")
        (tmp_path / "palace_index_map.json").write_text(
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
        palace_dir = indexed_report_dir / "output" / "palace"
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
            "surface-Q.csv": indexed_report_dir / "output" / "palace" / "surface-Q.csv",
            "palace_index_map.json": indexed_report_dir / "palace_index_map.json",
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

        budget = report.loss_budget.set_index("source_index")
        assert budget.loc[1, "domain_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert budget.loc[1, "surface_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert budget.loc[1, "total_inverse_q_sum"] == pytest.approx(1.0e-6)
        assert budget.loc[2, "total_inverse_q_sum"] == pytest.approx(5.0e-7)
        assert "t1_us" not in report.loss_budget.columns

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

    def test_electrostatic_report_is_publicly_exported(self) -> None:
        import gsim.palace as palace

        assert palace.ElectrostaticReport is ElectrostaticReport
        assert palace.load_electrostatic_report is load_electrostatic_report


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
            / "output"
            / "palace"
            / "terminal-Cm.csv",
            "palace_index_map.json": terminal_matrix_dir / "palace_index_map.json",
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
        palace_dir = terminal_matrix_dir / "output" / "palace"
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

    def test_history_appends_nonmatching_final(self, terminal_matrix_dir: Path) -> None:
        palace_dir = terminal_matrix_dir / "output" / "palace"
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


class TestSParamsSaveLoad:
    """Tests for SParams save_npz/from_file round-trip."""

    def test_round_trip(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "cached")
        assert out.suffix == ".npz"
        assert out.exists()

        loaded = SParams.from_file(out)
        assert loaded.port_names == sp.port_names
        assert len(loaded.freq) == len(sp.freq)
        np.testing.assert_allclose(loaded.freq, sp.freq)
        for key in sp._data:
            np.testing.assert_allclose(loaded[key].db, sp[key].db)
            np.testing.assert_allclose(loaded[key].deg, sp[key].deg)

    def test_adds_npz_suffix(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "no_ext")
        assert out.name == "no_ext.npz"

    def test_from_file_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SParams.from_file(tmp_path / "nonexistent.npz")


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
