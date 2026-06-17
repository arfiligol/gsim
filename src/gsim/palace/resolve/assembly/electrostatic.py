"""Compose Electrostatic problem reports from resolved Palace artifacts.

The Electrostatic report loader combines terminal matrices, matrix convergence,
EPR/loss summaries, source-status tables, and benchmark data into an
``ElectrostaticReport``.
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
from gsim.palace.results.loss import DomainLoss, LossBudget, SurfaceLoss
from gsim.palace.results.postprocessing import DomainEnergy, SurfaceQ
from gsim.palace.results.reports.electrostatic import ElectrostaticReport

if TYPE_CHECKING:
    from gsim.palace.resolve.sources.run_models import PalaceRunSummary


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
        loss_budget_result=LossBudget(dataframe=loss_budget),
        index_map=index_map_frame,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=REPORT_SOURCE_COLUMNS,
        ),
        run_summary=coerce_report_run_summary(source, run_summary),
    )
