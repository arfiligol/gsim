"""Tests for Palace report aggregation and generic display primitives."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import get_type_hints

import pytest

from gsim.palace import EigenmodeSim, PalaceRunHandle
from gsim.palace.display import (
    DisplayValue,
    PlotlyFigure,
    collect_visualizations,
    make_bar_figure,
    make_heatmap_figure,
    make_trace_figure,
    make_trace_subplot_figure,
)
from gsim.palace.resolve import (
    PalaceResolvedResult,
    PalaceResultBundle,
    resolve_palace_result,
)
from gsim.palace.resolve.assembly import (
    load_driven_report,
    load_electrostatic_report,
)
from gsim.palace.results import (
    DrivenReport,
    EigenmodeReport,
    Eigenmodes,
    ElectrostaticReport,
    SParams,
    TerminalMatrix,
)


def test_resolve_palace_result_then_load_report_for_driven(
    tmp_path: Path,
) -> None:
    """Result resolution is separate from typed report composition."""
    source = _write_driven_fixture(tmp_path)

    resolved = resolve_palace_result(source)
    bundle = resolved.load_report()

    assert isinstance(resolved, PalaceResolvedResult)
    assert resolved.problem_type == "Driven"
    assert resolved.artifacts.summary.problem_type == "Driven"
    assert isinstance(bundle, PalaceResultBundle)
    assert bundle.problem_type == "Driven"
    assert bundle.report_status == "loaded"
    assert bundle.report is not None
    report = bundle.require_report()
    assert isinstance(report, DrivenReport)
    assert report is bundle.report
    assert "s_parameters_trace_plot" in report.sparams.visualize()
    assert bundle.report.run_summary is bundle.run_summary
    assert "port-S.csv" in bundle.artifacts.result_names


def test_resolved_result_load_report_wraps_resolve_and_report_loading(
    tmp_path: Path,
) -> None:
    """The resolved result composes resolve metadata with a problem report."""
    source = _write_driven_fixture(tmp_path)

    bundle = resolve_palace_result(source).load_report()

    assert isinstance(bundle, PalaceResultBundle)
    assert bundle.problem_type == "Driven"
    assert bundle.report_status == "loaded"


def test_result_bundle_require_report_raises_when_report_missing(
    tmp_path: Path,
) -> None:
    """Typed report access raises with resolve context when loading failed."""
    _write_config(tmp_path, "Driven")

    bundle = resolve_palace_result(tmp_path, problem_type="Driven").load_report()

    assert bundle.report is None
    assert bundle.report_status == "missing"
    with pytest.raises(FileNotFoundError, match=r"port-S\.csv not found"):
        bundle.require_report()


def test_report_model_type_hints_are_runtime_introspectable() -> None:
    """Report annotations support reviewer probes such as get_type_hints."""
    driven_hints = get_type_hints(DrivenReport)
    eigenmode_hints = get_type_hints(EigenmodeReport)
    electrostatic_hints = get_type_hints(ElectrostaticReport)

    assert driven_hints["sparams"] is SParams
    assert eigenmode_hints["eigenmodes"] is Eigenmodes
    assert electrostatic_hints["terminal_c"] is TerminalMatrix


def test_driven_report_aggregates_typed_data_visualizers(tmp_path: Path) -> None:
    """Driven reports expose S-parameter typed-data tables and figures."""
    source = _write_driven_fixture(tmp_path)
    report = load_driven_report(source)

    items = report.result_items()

    assert isinstance(report, DrivenReport)
    assert report.performance is not None
    assert report.benchmark is not None
    assert "s_parameters_table" in items
    assert "s_parameters_trace_plot" in items
    assert "simulation_performance_table" in items
    assert "simulation_benchmark_table" in items
    assert "s_parameters_trace_plot" in report.sparams.figures()


def test_eigenmode_loss_typed_data_is_report_owned(tmp_path: Path) -> None:
    """Loss tables are typed data owned by the Eigenmode report."""
    source = _write_eigenmode_fixture(tmp_path)
    bundle = resolve_palace_result(source).load_report()

    assert bundle.report is not None
    items = bundle.report.result_items()

    assert "domain_epr_loss_table" in items
    assert "surface_epr_loss_table" in items
    assert "domain_inverse_q_bar_plot" in items
    assert "surface_inverse_q_bar_plot" in items
    assert items["domain_epr_loss_table"].iloc[0]["participation"] == pytest.approx(0.5)
    assert items["surface_epr_loss_table"].iloc[0]["participation"] == pytest.approx(
        1.0e-7
    )


def test_electrostatic_report_aggregates_terminal_matrix_visualizers(
    tmp_path: Path,
) -> None:
    """Electrostatic reports expose terminal-matrix typed data and history."""
    source = _write_electrostatic_fixture(tmp_path)
    report = load_electrostatic_report(source)

    items = report.result_items()

    assert report.problem_type == "Electrostatic"
    assert "terminal_C_table" in items
    assert "terminal_C_heatmap" in items
    assert "terminal_C_history_table" in items
    assert "terminal_C_pass_summary_table" in items


def test_display_primitives_are_problem_neutral() -> None:
    """Display helpers build generic Plotly figures without Palace report policy."""
    trace = make_trace_figure(
        [{"x": [1, 2], "y": [3, 4], "name": "trace"}],
        title="Trace",
        x_title="x",
        y_title="y",
    )
    heatmap = make_heatmap_figure(
        [[1, 2], [3, 4]],
        x=["a", "b"],
        y=["a", "b"],
        title="Matrix",
    )

    assert trace.layout.title.text == "Trace"
    assert heatmap.layout.title.text == "Matrix"


def test_display_figure_helpers_have_concrete_return_types() -> None:
    """Public Display figure helpers keep Plotly return types reviewable."""
    helpers = (
        make_bar_figure,
        make_heatmap_figure,
        make_trace_figure,
        make_trace_subplot_figure,
    )

    for helper in helpers:
        assert get_type_hints(helper)["return"] is PlotlyFigure


def test_display_collection_rejects_duplicate_semantic_keys() -> None:
    """Report aggregation fails when typed data repeats a display key."""

    class DuplicateVisualizer:
        def visualize(self) -> dict[str, DisplayValue]:
            return {"same_table": object()}

    with pytest.raises(ValueError, match="Duplicate Palace display item"):
        collect_visualizations((DuplicateVisualizer(), DuplicateVisualizer()))


def test_generate_handoff_package_writes_config_and_package_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handoff generation writes config and package metadata by default."""
    sim = EigenmodeSim()
    sim.set_output_dir(tmp_path)

    def fake_write_config(_self: EigenmodeSim, **kwargs: object) -> Path:
        assert kwargs["validate_mesh"] is True
        _write_config(tmp_path, "Eigenmode")
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "palace.msh").write_text("mesh\n")
        (metadata_dir / "mesh_manifest.json").write_text(
            json.dumps({"schema_version": 1, "entries": []})
        )
        (metadata_dir / "palace_index_map.json").write_text(
            json.dumps({"schema_version": 1, "entries": []})
        )
        (metadata_dir / "palace_material_resolution.json").write_text(
            json.dumps({"schema_version": 1, "materials": []})
        )
        return tmp_path / "config.json"

    monkeypatch.setattr(EigenmodeSim, "write_config", fake_write_config)

    handle = sim.generate_handoff_package()

    assert isinstance(handle, PalaceRunHandle)
    assert handle.problem_type == "Eigenmode"
    assert handle.status == "packaged"
    assert handle.kind == "handoff"
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metadata" / "palace_handoff_metadata.json").exists()
    assert (tmp_path / "metadata" / "palace_handoff_archive_manifest.json").exists()
    assert (tmp_path.parent / f"{tmp_path.name}-palace.tar.gz").exists()
    resolved = resolve_palace_result(handle.run_folder, problem_type="Eigenmode")
    assert not resolved.run_summary.missing_artifacts
    assert resolved.run_summary.handoff["status"] == "packaged"
    assert resolved.run_summary.handoff["archive_present"] is True
    assert resolved.run_summary.handoff["archive_manifest_present"] is True


def test_generate_handoff_package_records_packaged_script_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handoff package metadata can point at a generated batch script."""
    sim = EigenmodeSim()
    sim.set_output_dir(tmp_path)

    def fake_write_config(_self: EigenmodeSim, **_kwargs: object) -> Path:
        _write_config(tmp_path, "Eigenmode")
        (tmp_path / "palace.msh").write_text("mesh\n")
        (tmp_path / "run_palace.sbatch").write_text("#!/bin/bash\n")
        return tmp_path / "config.json"

    monkeypatch.setattr(EigenmodeSim, "write_config", fake_write_config)

    handle = sim.generate_handoff_package(
        launcher={"kind": "slurm", "submission": "manual"},
        script_path="run_palace.sbatch",
        command={"argv": ["sbatch", "run_palace.sbatch"]},
    )

    assert handle.kind == "slurm"
    assert handle.script_path == tmp_path / "run_palace.sbatch"
    resolved = resolve_palace_result(handle.run_folder, problem_type="Eigenmode")
    assert resolved.run_summary.handoff["script"] == {"path": "run_palace.sbatch"}
    archive_path = tmp_path.parent / f"{tmp_path.name}-palace.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert f"{tmp_path.name}/run_palace.sbatch" in names


def test_sim_generate_handoff_package_returns_run_handle_for_existing_outputs(
    tmp_path: Path,
) -> None:
    """Handoff packaging stays in Run Stage even when result files exist."""
    source = _write_eigenmode_fixture(tmp_path)
    sim = EigenmodeSim()
    sim.set_output_dir(source)

    handle = sim.generate_handoff_package(write_config=False)

    assert isinstance(handle, PalaceRunHandle)
    bundle = resolve_palace_result(
        handle.run_folder, problem_type="Eigenmode"
    ).load_report()
    assert isinstance(bundle, PalaceResultBundle)
    assert bundle.problem_type == "Eigenmode"
    assert bundle.report_status == "loaded"
    assert bundle.report is not None
    assert bundle.report.benchmark is not None
    assert (source / "metadata" / "palace_handoff_metadata.json").exists()
    assert (source / "metadata" / "palace_handoff_archive_manifest.json").exists()
    assert (source.parent / f"{source.name}-palace.tar.gz").exists()
    assert "eig.csv" in bundle.artifacts.result_names


def _write_config(source: Path, problem_type: str) -> None:
    (source / "config.json").write_text(
        json.dumps(
            {
                "Problem": {"Type": problem_type},
                "Domains": {"Materials": []},
                "Boundaries": {},
            }
        )
    )


def _write_driven_fixture(source: Path) -> Path:
    palace_dir = source / "results" / "palace"
    palace_dir.mkdir(parents=True)
    metadata_dir = source / "metadata"
    metadata_dir.mkdir()
    _write_config(source, "Driven")
    (metadata_dir / "port_information.json").write_text(
        json.dumps(
            {
                "ports": [{"portnumber": 1, "name": "o1", "Z0": 50.0}],
                "unit": 1e-6,
            }
        )
    )
    (metadata_dir / "palace_index_map.json").write_text(
        json.dumps({"schema_version": 1, "entries": []})
    )
    (palace_dir / "port-S.csv").write_text(
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.)\n"
        "5.0, -10.0, -45.0\n"
        "6.0, -8.0, -50.0\n"
    )
    return source


def _write_eigenmode_fixture(source: Path) -> Path:
    palace_dir = source / "results" / "palace"
    iteration_dir = palace_dir / "iteration01"
    iteration_dir.mkdir(parents=True)
    metadata_dir = source / "metadata"
    metadata_dir.mkdir()
    _write_config(source, "Eigenmode")
    (source / "config.json").write_text(
        json.dumps(
            {
                "Problem": {"Type": "Eigenmode"},
                "Domains": {
                    "Materials": [
                        {
                            "Attributes": [10],
                            "Name": "silicon",
                            "Permittivity": 11.45,
                            "LossTan": 1.0e-6,
                        }
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
                            }
                        ]
                    }
                },
            }
        )
    )
    (metadata_dir / "palace_index_map.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "section": "Domains.Postprocessing.Energy",
                        "index": 1,
                        "entry_name": "substrate",
                        "role": "dielectric_volume",
                        "attributes": [10],
                        "physical_names": ["substrate"],
                        "dimension": 3,
                    },
                    {
                        "section": "Boundaries.Postprocessing.Dielectric",
                        "index": 2,
                        "entry_name": "ma_interface",
                        "role": "boundary_surface",
                        "attributes": [20],
                        "physical_names": ["MA:metal__substrate"],
                        "dimension": 2,
                        "Type": "MA",
                    },
                ],
            }
        )
    )
    _write_eig_csv(iteration_dir / "eig.csv", frequency=5.0, q_factor=100.0)
    _write_eig_csv(palace_dir / "eig.csv", frequency=5.2, q_factor=120.0)
    (palace_dir / "domain-E.csv").write_text(
        "m, E_elec[1] (J), p_elec[1]\n1, 2.0, 0.5\n"
    )
    (palace_dir / "surface-Q.csv").write_text(
        "m, p_surf[2], Q_surf[2]\n1, 1.0e-7, 2.0e6\n"
    )
    return source


def _write_electrostatic_fixture(source: Path) -> Path:
    palace_dir = source / "results" / "palace"
    iteration_dir = palace_dir / "iteration01"
    iteration_dir.mkdir(parents=True)
    metadata_dir = source / "metadata"
    metadata_dir.mkdir()
    _write_config(source, "Electrostatic")
    (metadata_dir / "palace_index_map.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "section": "Boundaries.Terminal",
                        "index": 1,
                        "entry_name": "left",
                        "role": "pec_surface",
                        "attributes": [11],
                        "physical_names": ["left"],
                        "dimension": 2,
                        "terminal_name": "left",
                    },
                    {
                        "section": "Boundaries.Terminal",
                        "index": 2,
                        "entry_name": "right",
                        "role": "pec_surface",
                        "attributes": [12],
                        "physical_names": ["right"],
                        "dimension": 2,
                        "terminal_name": "right",
                    },
                ],
            }
        )
    )
    _write_terminal_c(iteration_dir / "terminal-C.csv", scale=1.0)
    _write_terminal_c(palace_dir / "terminal-C.csv", scale=2.0)
    return source


def _write_eig_csv(path: Path, *, frequency: float, q_factor: float) -> None:
    path.write_text(
        "m, Re{f} (GHz), Im{f} (GHz), Q, Error (Bkwd.), Error (Abs.)\n"
        f"1, {frequency}, 0.0, {q_factor}, 0.0, 0.0\n"
    )


def _write_terminal_c(path: Path, *, scale: float) -> None:
    path.write_text(
        "i, C[i][1] (F), C[i][2] (F)\n"
        f"1.00e+00, {scale * 1.0e-15}, {-scale * 2.0e-15}\n"
        f"2.00e+00, {-scale * 2.0e-15}, {scale * 4.0e-15}\n"
    )
