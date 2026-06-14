"""Tests for gsim.palace.results — S-parameter loading with port-name mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import (
    SParams,
    get_port_map,
    load_indexed_csv,
    load_postprocessing_index_map,
    load_sparams,
    load_terminal_matrix,
    load_terminal_matrix_history,
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
            },
        ],
    }
    (tmp_path / "palace_index_map.json").write_text(json.dumps(index_map))
    (palace_dir / "domain-E.csv").write_text(
        "m, E_elec[1] (J), p_elec[1], E_elec[99] (J)\n1, 2.0, 0.5, 0.0\n"
    )
    (palace_dir / "surface-Q.csv").write_text(
        "m, p_surf[2], Q_surf[2]\n1, 1.0e-7, 2.0e6\n"
    )
    return tmp_path


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


class TestIndexedCsv:
    """Tests for indexed Palace CSV loading through palace_index_map.json."""

    def test_loads_postprocessing_index_map(self, indexed_report_dir: Path) -> None:
        index_map = load_postprocessing_index_map(indexed_report_dir)

        entry = index_map.entry_for_index("Domains.Postprocessing.Energy", 1)
        assert entry is not None
        assert entry.primary_physical_name == "D1_SUBSTRATE"
        assert entry.metadata == {"material": "silicon"}

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
