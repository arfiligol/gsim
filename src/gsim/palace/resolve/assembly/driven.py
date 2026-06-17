"""Compose Driven problem reports from resolved Palace artifacts.

The Driven report loader reads typed S-parameter data, optional domain/surface
postprocessing tables, and audit sidecars, then returns a ``DrivenReport``. It
does not own S-parameter storage, loss table visualization, or Eigenmode-only
port-EPR semantics.
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
from gsim.palace.resolve.assembly.tables import load_optional_report_table
from gsim.palace.resolve.derived.loss import (
    summarize_domain_loss,
    summarize_surface_loss,
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
from gsim.palace.results.driven import load_sparams, resolve_sparameter_source
from gsim.palace.results.loss import DomainLoss, LossBudget, SurfaceLoss
from gsim.palace.results.postprocessing import DomainEnergy, SurfaceQ
from gsim.palace.results.reports.driven import DrivenReport

if TYPE_CHECKING:
    from gsim.palace.resolve.sources.run_models import PalaceRunSummary


def load_driven_report(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
    port_info_path: str | Path | None = None,
    run_summary: PalaceRunSummary | None = None,
) -> DrivenReport:
    """Load Driven S-parameters plus optional indexed postprocessing reports.

    This is a thin composition layer over the stricter primitive loaders. The
    final ``port-S.csv`` is required. Palace index/config/source-status artifacts
    and postprocessing CSVs are loaded independently when present and are
    reported as missing rather than forcing every Driven run to emit all report
    families.
    """
    import pandas as pd

    sparams = load_sparams(source, port_info_path=port_info_path)
    port_s_path, _ = resolve_sparameter_source(source)
    source_rows: list[dict[str, Any]] = [
        report_source_row(
            "port-S.csv",
            port_s_path,
            required=True,
            present=True,
            loaded=True,
            message="loaded driven S-parameters",
        )
    ]

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

    domain_loss = summarize_domain_loss(domain_energy, domain_materials)
    surface_loss = summarize_surface_loss(surface_q, dielectric_interfaces)
    surface_interface_summary = summarize_surface_q_by_interface(surface_loss)
    loss_budget = summarize_grouped_loss_budget(domain_loss, surface_loss)

    return DrivenReport(
        sparams=sparams,
        domain_energy_result=DomainEnergy(domain_energy),
        surface_q_result=SurfaceQ(surface_q),
        surface_interface_summary=surface_interface_summary,
        domain_epr_loss=DomainLoss(domain_loss),
        surface_epr_loss=SurfaceLoss(surface_loss),
        loss_budget_result=LossBudget(dataframe=loss_budget),
        domain_materials=domain_materials,
        dielectric_interfaces=dielectric_interfaces,
        index_map=index_map_frame,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=REPORT_SOURCE_COLUMNS,
        ),
        run_summary=coerce_report_run_summary(source, run_summary),
    )
