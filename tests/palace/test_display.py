"""Tests for Palace report aggregation and generic display primitives."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest

from gsim.palace import EigenmodeSim, PalaceRunHandle
from gsim.palace.display import (
    DisplayValue,
    PlotlyFigure,
    collect_visualizations,
    display_items,
    make_bar_figure,
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
    DomainEnergy,
    DrivenReport,
    EigenmodeReport,
    Eigenmodes,
    ElectrostaticReport,
    PortEpr,
    SParams,
    SurfaceQ,
    TerminalMatrix,
)
from gsim.palace.results.loss import DomainLoss, LossBudget, ReportLoss, SurfaceLoss


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


def test_typed_data_public_display_api_is_visualize_only(tmp_path: Path) -> None:
    """Typed Data shares ``visualize()`` without mandatory empty view hooks."""
    import pandas as pd

    reports = (
        load_driven_report(_write_driven_fixture(tmp_path / "driven")),
        resolve_palace_result(_write_eigenmode_fixture(tmp_path / "eigen"))
        .load_report()
        .require_report(),
        load_electrostatic_report(_write_electrostatic_fixture(tmp_path / "static")),
    )

    for report in reports:
        for typed_data in report.typed_results():
            visualize = typed_data.visualize
            items = visualize()
            assert isinstance(items, dict)
            assert items

    for table in (
        DomainEnergy(pd.DataFrame({"value": [1]})),
        SurfaceQ(pd.DataFrame({"value": [1]})),
        PortEpr(pd.DataFrame({"value": [1]})),
    ):
        assert list(table.visualize()) == [f"{table.table_name}_table"]
        assert not hasattr(table, "figures")


def test_dataframe_backed_results_copy_and_save_canonical_tables(
    tmp_path: Path,
) -> None:
    """Shared dataframe mechanics stay below Typed Data semantics."""
    import pandas as pd

    source = pd.DataFrame({"value": [1.0]})
    table = DomainEnergy(source)
    source.loc[0, "value"] = 9.0

    assert table.to_dataframe().loc[0, "value"] == pytest.approx(1.0)
    table_frame = table.to_dataframe()
    table_frame.loc[0, "value"] = 3.0
    assert table.to_dataframe().loc[0, "value"] == pytest.approx(1.0)

    csv_path = table.save_csv(tmp_path / "domain.csv")
    assert csv_path.exists()
    assert "value" in csv_path.read_text()


def test_driven_report_aggregates_typed_data_visualizers(tmp_path: Path) -> None:
    """Driven reports expose S-parameter typed-data tables and figures."""
    source = _write_driven_fixture(tmp_path)
    report = load_driven_report(source)

    items = report.result_items()
    benchmark_items = report.benchmark_items()

    assert isinstance(report, DrivenReport)
    assert report.benchmark is not None
    assert "s_parameters_table" in items
    assert "s_parameters_trace_plot" in items
    assert "simulation_benchmark_table" not in items
    assert benchmark_items == {}
    assert "s_parameters_trace_plot" in report.sparams.figures()


def test_eigenmode_loss_typed_data_is_report_owned(tmp_path: Path) -> None:
    """Loss tables are typed data owned by the Eigenmode report."""
    source = _write_eigenmode_fixture(tmp_path)
    bundle = resolve_palace_result(source).load_report()

    assert bundle.report is not None
    items = bundle.report.result_items()

    assert "domain_epr_summary_table" in items
    assert "surface_epr_summary_table" in items
    assert "loss_budget_table" in items
    assert "epr_convergence_trace_plot" not in items
    assert "epr_absolute_summary_table" not in items
    assert "epr_normalized_percentage_table" not in items
    assert "domain_epr_loss_table" not in items
    assert "surface_epr_loss_table" not in items
    assert "domain_inverse_q_bar_plot" not in items
    assert "surface_inverse_q_bar_plot" not in items
    domain_summary = cast(Any, items["domain_epr_summary_table"])
    surface_summary = cast(Any, items["surface_epr_summary_table"])
    assert domain_summary.iloc[0]["participation"] == pytest.approx(0.5)
    assert surface_summary.iloc[0]["participation"] == pytest.approx(1.0e-7)
    assert domain_summary.iloc[0]["participation_percent"] == pytest.approx(100.0)
    assert surface_summary.iloc[0]["participation_percent"] == pytest.approx(100.0)


def test_epr_compact_tables_normalize_participation_per_source_index() -> None:
    """Compact EPR percentages are local to each electrostatic source."""
    import pandas as pd

    loss = ReportLoss(
        domain=DomainLoss(
            pd.DataFrame(
                {
                    "source_index": [1, 1, 2],
                    "source_name": ["substrate", "vacuum", "substrate"],
                    "p_elec": [1.0, 3.0, 10.0],
                }
            )
        ),
        surface=SurfaceLoss(pd.DataFrame()),
        budget=LossBudget(dataframe=pd.DataFrame()),
    )

    table = loss.tables()["domain_epr_summary_table"].set_index(
        ["source_index", "source_name"]
    )

    assert table.loc[(1, "substrate"), "participation_percent"] == pytest.approx(25.0)
    assert table.loc[(1, "vacuum"), "participation_percent"] == pytest.approx(75.0)
    assert table.loc[(2, "substrate"), "participation_percent"] == pytest.approx(100.0)


def test_electrostatic_report_aggregates_terminal_matrix_visualizers(
    tmp_path: Path,
) -> None:
    """Electrostatic reports expose terminal-matrix typed data and history."""
    source = _write_electrostatic_fixture(tmp_path)
    report = load_electrostatic_report(source)

    items = report.result_items()

    assert report.problem_type == "Electrostatic"
    assert "terminal_C_table" in items
    assert "terminal_C_convergence_trace_plot" in items
    assert "terminal_C_delta_trace_plot" in items
    assert "terminal_C_heatmap" not in items
    assert "terminal_C_long_table" not in items
    assert "terminal_C_history_table" not in items
    assert "terminal_C_pass_summary_table" not in items


def test_electrostatic_report_adds_surface_epr_convergence_plot(
    tmp_path: Path,
) -> None:
    """Electrostatic loss display owns EPR convergence presentation."""
    source = _write_electrostatic_fixture(tmp_path)
    report = load_electrostatic_report(source)

    items = report.result_items()

    domain_figure = cast(Any, items["domain_epr_convergence_trace_plot"])
    assert {trace.name for trace in domain_figure.data} == {"source 1 substrate"}
    for trace in domain_figure.data:
        assert list(trace.x) == [1, 2, 3]

    figure = cast(Any, items["surface_epr_convergence_trace_plot"])
    assert {trace.name for trace in figure.data} == {
        "source 1 surface 2 i=1 MS MS:metal__substrate",
        "source 1 surface 3 i=1 MS MS:metal__air",
        "source 1 surface 4 i=1 MA MA:metal__air",
    }
    for trace in figure.data:
        assert list(trace.x) == [1, 2, 3]
    summary = cast(Any, items["surface_epr_summary_table"])
    assert "loss_channel" not in summary.columns
    assert set(summary["interface_type"]) == {"MS", "MA"}
    assert "attributes" in summary.columns


def test_surface_epr_summary_table_keeps_source_provenance() -> None:
    """Surface summary tables keep resolved source provenance."""
    import pandas as pd

    loss = ReportLoss(
        domain=DomainLoss(pd.DataFrame()),
        surface=SurfaceLoss(
            pd.DataFrame(
                {
                    "source_index": [1, 2],
                    "physical_name": ["MS:left", "MS:right"],
                    "attributes": [(20,), (21,)],
                    "interface_type": ["MS", "MS"],
                    "loss_channel": ["MS", "MS"],
                    "p_surf": [1.0e-7, 2.0e-7],
                }
            )
        ),
        budget=LossBudget(dataframe=pd.DataFrame()),
    )
    legacy_loss = ReportLoss(
        domain=DomainLoss(pd.DataFrame()),
        surface=SurfaceLoss(
            pd.DataFrame(
                {
                    "source_index": [1],
                    "interface_type": ["MS"],
                    "loss_channel": ["MS"],
                    "p_surf": [1.0e-7],
                }
            )
        ),
        budget=LossBudget(dataframe=pd.DataFrame()),
    )

    table = loss.tables()["surface_epr_summary_table"]
    legacy_table = legacy_loss.tables()["surface_epr_summary_table"]

    assert table["participation"].tolist() == pytest.approx([1.0e-7, 2.0e-7])
    assert "loss_channel" not in table.columns
    assert table["physical_name"].tolist() == ["MS:left", "MS:right"]
    assert table["attributes"].tolist() == [(20,), (21,)]
    assert legacy_table["participation"].tolist() == pytest.approx([1.0e-7])


def test_display_primitives_are_problem_neutral() -> None:
    """Display helpers build generic Plotly figures without Palace report policy."""
    trace = make_trace_figure(
        [{"x": [1, 2], "y": [3, 4], "name": "trace"}],
        title="Trace",
        x_title="x",
        y_title="y",
    )

    assert trace.layout.title.text == "Trace"


def test_display_items_does_not_duplicate_plotly_titles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plotly figures own their title; tables still get display headings."""
    import pandas as pd

    shown: list[object] = []
    monkeypatch.setattr("IPython.display.display", shown.append)

    display_items(
        {
            "table_item": pd.DataFrame({"value": [1]}),
            "plot_item": make_trace_figure(
                [{"x": [1], "y": [1], "name": "trace"}],
                title="Figure title",
            ),
        }
    )

    markdown = [str(getattr(value, "data", "")) for value in shown]
    assert "#### Table Item" in markdown
    assert "#### Plot Item" not in markdown


def test_display_items_formats_epr_summary_tables_at_display_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EPR summary values render compactly without mutating returned dataframes."""
    import pandas as pd

    frame = pd.DataFrame({"participation": [1.0e-7], "inverse_q": [2.0e-6]})
    shown: list[object] = []
    monkeypatch.setattr("IPython.display.display", shown.append)

    display_items({"surface_epr_summary_table": frame})

    assert shown[1].__class__.__name__ == "Styler"
    assert frame["participation"].iloc[0] == pytest.approx(1.0e-7)


def test_display_items_applies_plotly_layout_and_render_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plotly layout is cosmetic; Plotly config is passed at render time."""
    figure = make_trace_figure(
        [{"x": [1], "y": [1], "name": "trace"}],
        title="Semantic title",
    )
    shown: list[tuple[object, dict[str, object]]] = []

    def fake_show(value: object, **kwargs: object) -> None:
        shown.append((value, kwargs))

    monkeypatch.setattr("plotly.io.show", fake_show)

    display_items(
        {"plot_item": figure},
        figure_layout={"height": 480},
        plotly_config={"responsive": True, "displaylogo": False},
    )

    assert figure.layout.title.text == "Semantic title"
    assert figure.layout.height == 480
    assert shown == [
        (
            figure,
            {"config": {"responsive": True, "displaylogo": False}},
        )
    ]


def test_show_all_results_forwards_display_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Problem reports aggregate items and leave figure styling to Display."""
    report = load_driven_report(_write_driven_fixture(tmp_path))
    calls: list[dict[str, object]] = []

    def fake_display_items(
        items: object,
        *,
        figure_layout: object = None,
        plotly_config: object = None,
    ) -> None:
        calls.append(
            {
                "items": items,
                "figure_layout": figure_layout,
                "plotly_config": plotly_config,
            }
        )

    monkeypatch.setattr(
        "gsim.palace.results.reports.base.display_items",
        fake_display_items,
    )

    returned = report.show_all_results(
        figure_layout={"template": "plotly_white"},
        plotly_config={"responsive": True},
    )

    assert returned is calls[0]["items"]
    assert "simulation_benchmark_table" not in returned
    assert calls[0]["figure_layout"] == {"template": "plotly_white"}
    assert calls[0]["plotly_config"] == {"responsive": True}


def test_show_simulation_benchmark_forwards_display_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulation benchmark display stays explicit and separately styleable."""
    report = load_driven_report(_write_driven_fixture(tmp_path))
    calls: list[dict[str, object]] = []

    def fake_display_items(
        items: object,
        *,
        figure_layout: object = None,
        plotly_config: object = None,
    ) -> None:
        calls.append(
            {
                "items": items,
                "figure_layout": figure_layout,
                "plotly_config": plotly_config,
            }
        )

    monkeypatch.setattr(
        "gsim.palace.results.reports.base.display_items",
        fake_display_items,
    )

    returned = report.show_simulation_benchmark(
        figure_layout={"height": 360},
        plotly_config={"responsive": True},
    )

    assert returned is calls[0]["items"]
    assert returned == {}
    assert calls[0]["figure_layout"] == {"height": 360}
    assert calls[0]["plotly_config"] == {"responsive": True}


def test_display_figure_helpers_have_concrete_return_types() -> None:
    """Public Display figure helpers keep Plotly return types reviewable."""
    helpers = (
        make_bar_figure,
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
    (source / "config.json").write_text(
        json.dumps(
            {
                "Problem": {"Type": "Electrostatic"},
                "Domains": {"Materials": []},
                "Boundaries": {
                    "Postprocessing": {
                        "Dielectric": [
                            {
                                "Index": 2,
                                "Attributes": [20],
                                "Type": "MS",
                                "Thickness": 0.002,
                                "Permittivity": 10.0,
                                "LossTan": 0.0033,
                            },
                            {
                                "Index": 3,
                                "Attributes": [21],
                                "Type": "MS",
                                "Thickness": 0.002,
                                "Permittivity": 10.0,
                                "LossTan": 0.0033,
                            },
                            {
                                "Index": 4,
                                "Attributes": [22],
                                "Type": "MA",
                                "Thickness": 0.002,
                                "Permittivity": 10.0,
                                "LossTan": 0.0033,
                            },
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
                    {
                        "section": "Domains.Postprocessing.Energy",
                        "index": 1,
                        "entry_name": "substrate",
                        "role": "dielectric_volume",
                        "attributes": [1],
                        "physical_names": ["substrate"],
                        "dimension": 3,
                    },
                    {
                        "section": "Boundaries.Postprocessing.Dielectric",
                        "index": 2,
                        "entry_name": "ms_total",
                        "role": "boundary_surface",
                        "attributes": [20],
                        "physical_names": ["MS:metal__substrate"],
                        "dimension": 2,
                        "metadata": {
                            "loss_channel": "MS",
                            "source_entry_name": "metal",
                        },
                    },
                    {
                        "section": "Boundaries.Postprocessing.Dielectric",
                        "index": 3,
                        "entry_name": "ms_air",
                        "role": "boundary_surface",
                        "attributes": [21],
                        "physical_names": ["MS:metal__air"],
                        "dimension": 2,
                        "metadata": {
                            "loss_channel": "MS",
                            "source_entry_name": "metal",
                        },
                    },
                    {
                        "section": "Boundaries.Postprocessing.Dielectric",
                        "index": 4,
                        "entry_name": "ma_air",
                        "role": "boundary_surface",
                        "attributes": [22],
                        "physical_names": ["MA:metal__air"],
                        "dimension": 2,
                        "metadata": {
                            "loss_channel": "MA",
                            "source_entry_name": "metal",
                        },
                    },
                ],
            }
        )
    )
    _write_terminal_c(iteration_dir / "terminal-C.csv", scale=1.0)
    _write_terminal_c(palace_dir / "terminal-C.csv", scale=2.0)
    (iteration_dir / "domain-E.csv").write_text(
        "i, E_elec[1] (J), p_elec[1]\n1, 1.0, 0.2\n"
    )
    (iteration_dir / "surface-Q.csv").write_text(
        "i, p_surf[2], Q_surf[2], p_surf[3], Q_surf[3], p_surf[4], Q_surf[4]\n"
        "1, 1.0e-7, 1.0e7, 2.0e-7, 5.0e6, 3.0e-7, 3.333333e6\n"
    )
    iteration_dir_2 = palace_dir / "iteration02"
    iteration_dir_2.mkdir()
    (iteration_dir_2 / "domain-E.csv").write_text(
        "i, E_elec[1] (J), p_elec[1]\n1, 1.5, 0.3\n"
    )
    (iteration_dir_2 / "surface-Q.csv").write_text(
        "i, p_surf[2], Q_surf[2], p_surf[3], Q_surf[3], p_surf[4], Q_surf[4]\n"
        "1, 2.0e-7, 5.0e6, 3.0e-7, 3.333333e6, 4.0e-7, 2.5e6\n"
    )
    (palace_dir / "domain-E.csv").write_text(
        "i, E_elec[1] (J), p_elec[1]\n1, 2.0, 0.3\n"
    )
    (palace_dir / "surface-Q.csv").write_text(
        "i, p_surf[2], Q_surf[2], p_surf[3], Q_surf[3], p_surf[4], Q_surf[4]\n"
        "1, 2.0e-7, 5.0e6, 3.0e-7, 3.333333e6, 4.0e-7, 2.5e6\n"
    )
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
