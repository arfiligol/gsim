"""Compose Eigenmode problem reports from resolved Palace artifacts.

The Eigenmode report loader combines eigenmode typed data, EPR tables, loss
summaries, material/index lookup data, and run benchmarks into an
``EigenmodeReport``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gsim.palace.resolve.assembly.common import (
    REPORT_SOURCE_COLUMNS,
    coerce_report_run_summary,
    empty_dielectric_interface_summary,
    empty_domain_energy_summary,
    empty_domain_material_summary,
    empty_port_epr_summary,
    empty_surface_q_summary,
    report_source_row,
)
from gsim.palace.resolve.assembly.tables import load_optional_report_table
from gsim.palace.resolve.derived.loss import (
    summarize_domain_loss,
    summarize_loss_budget,
    summarize_surface_loss,
)
from gsim.palace.resolve.derived.materials import (
    find_optional_config_path,
    load_dielectric_interface_summary,
    load_domain_material_summary,
)
from gsim.palace.resolve.derived.participation import (
    load_domain_energy_summary,
    load_port_epr_summary,
    load_surface_q_summary,
    summarize_surface_q_by_interface,
)
from gsim.palace.resolve.loaders.eigenmodes import (
    add_eigenmode_convergence_columns,
    eigenmode_history_source,
    eigenmodes_to_history_frame,
    empty_eigenmode_history,
    empty_eigenmode_pass_summary,
    find_eigenmode_iteration_csvs,
    load_eigenmode_history,
    load_eigenmodes,
    summarize_eigenmode_history,
)
from gsim.palace.resolve.loaders.index_maps import (
    empty_postprocessing_index_map_dataframe,
    find_optional_postprocessing_index_map_path,
    load_postprocessing_index_map,
    postprocessing_index_map_to_dataframe,
)
from gsim.palace.results.eigenmode import EigenmodeConvergence
from gsim.palace.results.loss import DomainLoss, LossBudget, SurfaceLoss
from gsim.palace.results.postprocessing import (
    DomainEnergy,
    PortEpr,
    SurfaceQ,
)
from gsim.palace.results.reports.eigenmode import EigenmodeReport

if TYPE_CHECKING:
    import pandas as pd

    from gsim.palace.resolve.sources.run_models import PalaceRunSummary
    from gsim.palace.results.eigenmode import Eigenmodes


def load_eigenmode_report(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
    include_final: bool = True,
    require_epr: bool = False,
    run_summary: PalaceRunSummary | None = None,
) -> EigenmodeReport:
    """Load final Eigenmode rows plus optional indexed EPR report tables.

    This is a thin composition layer over the stricter primitive loaders. The
    final ``eig.csv`` is required. Palace indexed reports are loaded
    independently when present and are reported as missing rather than forcing
    every Eigenmode run to emit all EPR families.
    """
    import pandas as pd

    eigenmodes = load_eigenmodes(source)
    source_rows: list[dict[str, Any]] = []
    source_rows.append(
        report_source_row(
            "eig.csv",
            eigenmodes.source_path,
            required=True,
            present=True,
            loaded=True,
            message="loaded final eigenmode modes",
        )
    )

    mode_history = _load_eigenmode_history_for_report(
        source,
        eigenmodes,
        include_final=include_final,
        source_rows=source_rows,
    )
    pass_summary = (
        empty_eigenmode_pass_summary()
        if mode_history.empty
        else summarize_eigenmode_history(mode_history)
    )

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
    port_epr, port_source = load_optional_report_table(
        source,
        "port-EPR.csv",
        loader=load_port_epr_summary,
        empty_factory=empty_port_epr_summary,
        index_map_path=resolved_index_map_path,
        index_map_present=index_map_present,
    )
    source_rows.append(port_source)

    if require_epr:
        required_failures = [
            row["name"]
            for row in (domain_source, surface_source)
            if not bool(row["loaded"])
        ]
        if required_failures:
            failure_names = ", ".join(str(name) for name in required_failures)
            msg = f"Missing required eigenmode EPR reports: {failure_names}"
            raise FileNotFoundError(msg)

    domain_loss = summarize_domain_loss(
        domain_energy,
        domain_materials,
        modes=eigenmodes,
    )
    surface_loss = summarize_surface_loss(
        surface_q,
        dielectric_interfaces,
        modes=eigenmodes,
    )
    surface_interface_summary = summarize_surface_q_by_interface(surface_loss)
    loss_budget = summarize_loss_budget(
        domain_loss,
        surface_loss,
        modes=eigenmodes,
    )
    return EigenmodeReport(
        eigenmodes=eigenmodes,
        convergence=EigenmodeConvergence(
            mode_history=mode_history,
            pass_summary=pass_summary,
        ),
        domain_energy_result=DomainEnergy(domain_energy),
        surface_q_result=SurfaceQ(surface_q),
        surface_interface_summary=surface_interface_summary,
        port_epr_result=PortEpr(port_epr),
        domain_epr_loss=DomainLoss(domain_loss),
        surface_epr_loss=SurfaceLoss(surface_loss),
        loss_budget_result=LossBudget(dataframe=loss_budget),
        index_map=index_map_frame,
        domain_materials=domain_materials,
        dielectric_interfaces=dielectric_interfaces,
        sources=pd.DataFrame.from_records(
            source_rows,
            columns=REPORT_SOURCE_COLUMNS,
        ),
        run_summary=coerce_report_run_summary(source, run_summary),
    )


def _load_eigenmode_history_for_report(
    source: str | Path | dict,
    eigenmodes: Eigenmodes,
    *,
    include_final: bool,
    source_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Load or synthesize Eigenmode AMR history for one report."""
    import pandas as pd

    history_source = eigenmode_history_source(source, eigenmodes.source_path)
    iteration_paths = find_eigenmode_iteration_csvs(history_source)
    source_rows.append(
        report_source_row(
            "iteration*/eig.csv",
            None,
            required=False,
            present=bool(iteration_paths),
            loaded=bool(iteration_paths),
            message=(
                f"loaded {len(iteration_paths)} AMR iteration files"
                if iteration_paths
                else "no AMR iteration eig.csv files found"
            ),
        )
    )

    if not include_final and not iteration_paths:
        return empty_eigenmode_history()

    try:
        return load_eigenmode_history(history_source, include_final=include_final)
    except (FileNotFoundError, ValueError):
        if not include_final:
            return empty_eigenmode_history()
        final_history = eigenmodes_to_history_frame(
            eigenmodes,
            iteration_index=1,
            label="Final",
            is_final=True,
            source_kind="final",
            source_iteration=None,
        )
        return cast(
            "pd.DataFrame",
            add_eigenmode_convergence_columns(pd.DataFrame(final_history)),
        )
