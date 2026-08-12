"""Compose Electrostatic problem reports from resolved Palace artifacts.

The Electrostatic report loader combines terminal matrices, matrix convergence,
EPR/loss summaries, source-status tables, and benchmark data into an
``ElectrostaticReport``. This module owns only assembly policy: which located
artifacts feed an Electrostatic report, which optional source-status rows are
recorded, and how Resolve-derived tables are passed into Typed Data objects.

It does not discover run roots, parse terminal matrix CSVs directly, define
report classes, or render notebook output. Source discovery lives in
``gsim.palace.resolve.sources``; primitive/derived loading lives in
``gsim.palace.resolve.loaders`` and ``gsim.palace.resolve.derived``; report
and visualization semantics live in ``gsim.palace.results`` and
``gsim.palace.display``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsim.palace.resolve.assembly.common import (
    REPORT_SOURCE_COLUMNS,
    coerce_report_run_summary,
    empty_dielectric_interface_summary,
    empty_domain_energy_summary,
    empty_domain_material_summary,
    empty_surface_q_summary,
    report_source_row,
)
from gsim.palace.resolve.assembly.electrostatic_terminals import (
    load_electrostatic_terminal_report_parts,
)
from gsim.palace.resolve.assembly.tables import load_optional_report_table
from gsim.palace.resolve.derived.loss import (
    summarize_domain_loss,
    summarize_surface_loss,
    validate_positive_frequency_ghz,
)
from gsim.palace.resolve.derived.loss_grouped import summarize_grouped_loss_budget
from gsim.palace.resolve.derived.materials import (
    find_optional_config_path,
    load_dielectric_interface_summary,
    load_domain_material_summary,
)
from gsim.palace.resolve.derived.participation import (
    load_domain_energy_summary,
    load_surface_q_summary,
    summarize_surface_q_by_interface,
)
from gsim.palace.resolve.loaders.index_maps import (
    empty_postprocessing_index_map_dataframe,
    find_optional_postprocessing_index_map_path,
    load_postprocessing_index_map,
    postprocessing_index_map_to_dataframe,
)
from gsim.palace.resolve.loaders.terminal import iteration_dirs
from gsim.palace.resolve.sources.path_utils import resolve_palace_output_dir
from gsim.palace.results.loss import DomainLoss, LossBudget, SurfaceLoss
from gsim.palace.results.postprocessing import DomainEnergy, SurfaceQ
from gsim.palace.results.reports.electrostatic import ElectrostaticReport

if TYPE_CHECKING:
    import pandas as pd

    from gsim.palace.resolve.sources.run_models import PalaceRunSummary


SURFACE_EPR_CONVERGENCE_COLUMNS = (
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
)
DOMAIN_EPR_CONVERGENCE_COLUMNS = (
    "pass_index",
    "source_index",
    "sample_column",
    "sample_value",
    "domain_index",
    "source_name",
    "physical_name",
    "domain_epr_abs",
)


def load_electrostatic_report(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
    terminal_names: tuple[str, ...] | list[str] | None = None,
    include_history: bool = True,
    require_epr: bool = False,
    frequency_ghz: float | None = None,
    run_summary: PalaceRunSummary | None = None,
) -> ElectrostaticReport:
    """Load Electrostatic terminal matrices plus optional indexed EPR reports.

    Electrostatic Palace outputs do not carry a resonant frequency. Loss-rate
    and T1 columns are therefore derived only when ``frequency_ghz`` is passed
    explicitly; otherwise the report keeps inverse-Q and equivalent-Q columns.
    Missing optional EPR reports produce empty typed tables and source-status
    audit rows. Missing required terminal-capacitance data is handled by the
    terminal assembly helper as a hard loader failure.
    """
    import pandas as pd

    if frequency_ghz is not None:
        validate_positive_frequency_ghz(frequency_ghz)

    source_rows: list[dict[str, Any]] = []
    resolved_index_map_path = find_optional_postprocessing_index_map_path(
        source,
        index_map_path=index_map_path,
    )
    index_map_present = (
        resolved_index_map_path is not None and resolved_index_map_path.exists()
    )
    index_map_loaded = False
    if index_map_present:
        index_map = load_postprocessing_index_map(
            source,
            index_map_path=resolved_index_map_path,
        )
        index_map_frame = postprocessing_index_map_to_dataframe(index_map)
        index_map_loaded = True
        index_message = "loaded postprocessing index map"
    else:
        index_map_frame = empty_postprocessing_index_map_dataframe()
        index_message = "not found"
    source_rows.append(
        report_source_row(
            "palace_index_map.json",
            resolved_index_map_path,
            required=False,
            present=index_map_present,
            loaded=index_map_loaded,
            message=index_message,
        )
    )

    terminal_parts = load_electrostatic_terminal_report_parts(
        source,
        index_map_path=resolved_index_map_path if index_map_present else None,
        terminal_names=terminal_names,
        include_history=include_history,
    )
    source_rows.extend(terminal_parts.source_rows)

    resolved_config_path = find_optional_config_path(source, config_path=None)
    config_present = resolved_config_path is not None and resolved_config_path.exists()
    config_loaded = False
    if config_present:
        domain_materials = load_domain_material_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        dielectric_interfaces = load_dielectric_interface_summary(
            source,
            config_path=resolved_config_path,
            index_map_path=resolved_index_map_path if index_map_present else None,
        )
        config_loaded = True
        config_message = (
            "loaded config material and interface summaries"
            if index_map_present
            else (
                "loaded config material and interface summaries without "
                "palace_index_map.json"
            )
        )
    else:
        domain_materials = empty_domain_material_summary()
        dielectric_interfaces = empty_dielectric_interface_summary()
        config_message = "not found"
    source_rows.append(
        report_source_row(
            "config.json",
            resolved_config_path,
            required=False,
            present=config_present,
            loaded=config_loaded,
            message=config_message,
        )
    )

    domain_energy, domain_source = load_optional_report_table(
        source,
        "domain-E.csv",
        loader=load_domain_energy_summary,
        empty_factory=empty_domain_energy_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(domain_source)
    surface_q, surface_source = load_optional_report_table(
        source,
        "surface-Q.csv",
        loader=load_surface_q_summary,
        empty_factory=empty_surface_q_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(surface_source)

    if require_epr:
        required_failures = [
            row["name"]
            for row in (domain_source, surface_source)
            if not bool(row["loaded"])
        ]
        if required_failures:
            failure_names = ", ".join(str(name) for name in required_failures)
            msg = f"Missing required electrostatic EPR reports: {failure_names}"
            raise FileNotFoundError(msg)

    domain_loss = summarize_domain_loss(
        domain_energy,
        domain_materials,
        frequency_ghz=frequency_ghz,
    )
    surface_loss = summarize_surface_loss(
        surface_q,
        dielectric_interfaces,
        frequency_ghz=frequency_ghz,
    )
    surface_interface_summary = summarize_surface_q_by_interface(surface_loss)
    domain_epr_convergence = load_domain_epr_convergence_for_report(
        source,
        include_history=include_history,
        index_map_path=resolved_index_map_path if index_map_present else None,
        source_rows=source_rows,
    )
    surface_epr_convergence = load_surface_epr_convergence_for_report(
        source,
        include_history=include_history,
        index_map_path=resolved_index_map_path if index_map_present else None,
        dielectric_interfaces=dielectric_interfaces,
        frequency_ghz=frequency_ghz,
        source_rows=source_rows,
    )
    loss_budget = summarize_grouped_loss_budget(
        domain_loss,
        surface_loss,
        frequency_ghz=frequency_ghz,
    )
    return ElectrostaticReport(
        terminal_c=terminal_parts.terminal_c,
        terminal_cm=terminal_parts.terminal_cm,
        terminal_cinv=terminal_parts.terminal_cinv,
        terminal_c_convergence=terminal_parts.terminal_c_convergence,
        terminal_cm_convergence=terminal_parts.terminal_cm_convergence,
        terminal_cinv_convergence=terminal_parts.terminal_cinv_convergence,
        domain_materials=domain_materials,
        dielectric_interfaces=dielectric_interfaces,
        domain_energy_result=DomainEnergy(domain_energy),
        surface_q_result=SurfaceQ(surface_q),
        surface_interface_summary=surface_interface_summary,
        domain_epr_loss=DomainLoss(domain_loss),
        surface_epr_loss=SurfaceLoss(surface_loss),
        domain_epr_convergence=domain_epr_convergence,
        surface_epr_convergence=surface_epr_convergence,
        loss_budget_result=LossBudget(dataframe=loss_budget),
        index_map=index_map_frame,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=REPORT_SOURCE_COLUMNS,
        ),
        run_summary=coerce_report_run_summary(source, run_summary),
    )


def load_domain_epr_convergence_for_report(
    source: str | Path | dict,
    *,
    include_history: bool,
    index_map_path: Path | None,
    source_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Load AMR Domain EPR convergence rows and record their source status.

    The returned dataframe is the narrow report contract consumed by
    ``ReportLoss`` for convergence figures. ``source_rows`` is extended with
    one audit row whether history is loaded, disabled, or absent.
    """
    import pandas as pd

    iteration_paths = find_report_iteration_csvs(source, "domain-E.csv")
    report_paths = append_final_report_csv(
        source,
        "domain-E.csv",
        iteration_paths=iteration_paths,
    )
    history_row = report_source_row(
        "iteration*/domain-E.csv",
        None,
        required=False,
        present=bool(iteration_paths),
        loaded=False,
        message=history_source_message(
            include_history=include_history,
            iteration_paths=iteration_paths,
            report_paths=report_paths,
            csv_name="domain-E.csv",
        ),
    )
    source_rows.append(history_row)
    if not include_history or not iteration_paths:
        return empty_domain_epr_convergence()

    frames: list[pd.DataFrame] = []
    for csv_path, pass_index in report_paths:
        source_arg: dict[str, Path] = {"domain-E.csv": csv_path}
        if index_map_path is not None:
            source_arg["palace_index_map.json"] = index_map_path
        domain_energy = load_domain_energy_summary(
            source_arg if index_map_path is not None else csv_path,
            index_map_path=index_map_path,
        )
        if domain_energy.empty or "p_elec" not in domain_energy.columns:
            continue
        frame = domain_energy.copy()
        frame["pass_index"] = pass_index
        frames.append(frame)

    if not frames:
        return empty_domain_epr_convergence()

    history = pd.concat(frames, ignore_index=True)
    history["domain_epr_abs"] = pd.to_numeric(
        history["p_elec"],
        errors="coerce",
    ).abs()
    for column in DOMAIN_EPR_CONVERGENCE_COLUMNS:
        if column not in history.columns and column != "domain_epr_abs":
            history[column] = pd.NA
    history = history.dropna(subset=["domain_epr_abs"])
    if history.empty:
        return empty_domain_epr_convergence()

    group_columns = [
        column
        for column in DOMAIN_EPR_CONVERGENCE_COLUMNS
        if column != "domain_epr_abs"
    ]
    result = history.groupby(
        group_columns,
        dropna=False,
        as_index=False,
    )["domain_epr_abs"].sum()
    history_row["loaded"] = not result.empty
    return result.loc[:, DOMAIN_EPR_CONVERGENCE_COLUMNS].sort_values(
        ["source_index", "domain_index", "source_name", "pass_index"],
        na_position="last",
    )


def load_surface_epr_convergence_for_report(
    source: str | Path | dict,
    *,
    include_history: bool,
    index_map_path: Path | None,
    dielectric_interfaces: Any,
    frequency_ghz: float | None,
    source_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Load AMR Surface EPR convergence rows and record their source status.

    Surface history uses the same loss summarization path as the final report
    so source and physical-surface provenance match the final Surface EPR table.
    ``source_rows`` is extended with one audit row whether history is loaded,
    disabled, or absent.
    """
    import pandas as pd

    iteration_paths = find_report_iteration_csvs(source, "surface-Q.csv")
    report_paths = append_final_report_csv(
        source,
        "surface-Q.csv",
        iteration_paths=iteration_paths,
    )
    history_row = report_source_row(
        "iteration*/surface-Q.csv",
        None,
        required=False,
        present=bool(iteration_paths),
        loaded=False,
        message=history_source_message(
            include_history=include_history,
            iteration_paths=iteration_paths,
            report_paths=report_paths,
            csv_name="surface-Q.csv",
        ),
    )
    source_rows.append(history_row)
    if not include_history or not iteration_paths:
        return empty_surface_epr_convergence()

    frames: list[pd.DataFrame] = []
    for csv_path, pass_index in report_paths:
        source_arg: dict[str, Path] = {"surface-Q.csv": csv_path}
        if index_map_path is not None:
            source_arg["palace_index_map.json"] = index_map_path
        surface_q = load_surface_q_summary(
            source_arg if index_map_path is not None else csv_path,
            index_map_path=index_map_path,
        )
        surface_loss = summarize_surface_loss(
            surface_q,
            dielectric_interfaces,
            frequency_ghz=frequency_ghz,
        )
        if surface_loss.empty or "p_surf" not in surface_loss.columns:
            continue
        frame = surface_loss.copy()
        frame["pass_index"] = pass_index
        frames.append(frame)

    if not frames:
        return empty_surface_epr_convergence()

    history = pd.concat(frames, ignore_index=True)
    history["surface_epr_abs"] = pd.to_numeric(
        history["p_surf"],
        errors="coerce",
    ).abs()
    for column in SURFACE_EPR_CONVERGENCE_COLUMNS:
        if column not in history.columns and column != "surface_epr_abs":
            history[column] = pd.NA
    history = history.dropna(subset=["surface_epr_abs"])
    if history.empty:
        return empty_surface_epr_convergence()

    group_columns = [
        column
        for column in SURFACE_EPR_CONVERGENCE_COLUMNS
        if column != "surface_epr_abs"
    ]
    result = history.groupby(
        group_columns,
        dropna=False,
        as_index=False,
    )["surface_epr_abs"].sum()
    history_row["loaded"] = not result.empty
    return result.loc[:, SURFACE_EPR_CONVERGENCE_COLUMNS].sort_values(
        [
            "source_index",
            "surface_index",
            "interface_type",
            "source_name",
            "sample_column",
            "sample_value",
            "pass_index",
        ],
        na_position="last",
    )


def find_report_iteration_csvs(
    source: str | Path | dict,
    csv_name: str,
) -> tuple[tuple[Path, int], ...]:
    """Return existing ``iteration*/<csv_name>`` paths sorted by adaptive pass.

    Path discovery is deliberately limited to filesystem sources. Dict sources
    represent already-located artifacts and cannot imply sibling AMR folders.
    """
    if isinstance(source, dict):
        return ()

    path = Path(source)
    base = path.parent if path.is_file() else path
    output_dir = resolve_palace_output_dir(base)
    return tuple(
        (iteration_dir / csv_name, pass_index)
        for iteration_dir, pass_index in iteration_dirs(output_dir)
        if (iteration_dir / csv_name).exists()
    )


def append_final_report_csv(
    source: str | Path | dict,
    csv_name: str,
    *,
    iteration_paths: tuple[tuple[Path, int], ...],
) -> tuple[tuple[Path, int], ...]:
    """Append the top-level final report CSV as the last convergence sample.

    The final Palace report is shown after adaptive passes only when history
    already exists. This preserves the existing no-history contract for runs
    without ``iteration*/`` directories.
    """
    if isinstance(source, dict) or not iteration_paths:
        return iteration_paths

    path = Path(source)
    base = path.parent if path.is_file() else path
    output_dir = resolve_palace_output_dir(base)
    final_path = output_dir / csv_name
    if not final_path.exists():
        return iteration_paths
    final_pass = max(pass_index for _, pass_index in iteration_paths) + 1
    return (*iteration_paths, (final_path, final_pass))


def history_source_message(
    *,
    include_history: bool,
    iteration_paths: tuple[tuple[Path, int], ...],
    report_paths: tuple[tuple[Path, int], ...],
    csv_name: str,
) -> str:
    """Return a compact source-status message for AMR history CSVs."""
    if not include_history:
        return "history disabled"
    if not iteration_paths:
        return f"no AMR iteration {csv_name} files found"
    suffix = " plus final report" if len(report_paths) > len(iteration_paths) else ""
    return f"found {len(iteration_paths)} AMR iteration files{suffix}"


def empty_domain_epr_convergence() -> pd.DataFrame:
    """Return the empty Domain EPR convergence dataframe contract."""
    import pandas as pd

    return pd.DataFrame(columns=DOMAIN_EPR_CONVERGENCE_COLUMNS)


def empty_surface_epr_convergence() -> pd.DataFrame:
    """Return the empty Surface EPR convergence dataframe contract."""
    import pandas as pd

    return pd.DataFrame(columns=SURFACE_EPR_CONVERGENCE_COLUMNS)
