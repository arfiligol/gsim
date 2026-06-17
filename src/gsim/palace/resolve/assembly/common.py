"""Shared helpers for Resolve report assembly.

These helpers belong to ``gsim.palace.resolve.assembly``. They define
source-status rows and empty dataframe contracts used while assembling a
problem report from optional Palace artifacts.

They do not parse solver files, compute physics quantities, define report
models, or render notebook views.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsim.palace.resolve.derived.materials import (
    DIELECTRIC_INTERFACE_COLUMNS,
    DOMAIN_MATERIAL_COLUMNS,
)
from gsim.palace.resolve.sources.run_summary import load_palace_run_summary

if TYPE_CHECKING:
    import pandas as pd

    from gsim.palace.resolve.sources.run_models import PalaceRunSummary

REPORT_SOURCE_COLUMNS = ("name", "path", "required", "present", "loaded", "message")


def report_source_row(
    name: str,
    path: str | Path | None,
    *,
    required: bool,
    present: bool,
    loaded: bool,
    message: str,
) -> dict[str, Any]:
    """Return one source-status row for a composed problem report."""
    return {
        "name": name,
        "path": None if path is None else str(path),
        "required": required,
        "present": present,
        "loaded": loaded,
        "message": message,
    }


def coerce_report_run_summary(
    source: str | Path | dict,
    run_summary: PalaceRunSummary | None,
) -> PalaceRunSummary:
    """Return a caller-supplied run summary or resolve one from artifacts."""
    if run_summary is not None:
        return run_summary
    return load_palace_run_summary(source)


def empty_domain_material_summary() -> pd.DataFrame:
    """Return the empty domain-material summary contract."""
    import pandas as pd

    return pd.DataFrame(columns=DOMAIN_MATERIAL_COLUMNS)


def empty_dielectric_interface_summary() -> pd.DataFrame:
    """Return the empty dielectric-interface summary contract."""
    import pandas as pd

    return pd.DataFrame(columns=DIELECTRIC_INTERFACE_COLUMNS)


def empty_domain_energy_summary() -> pd.DataFrame:
    """Return the empty domain-energy summary contract."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "domain_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "E_elec_j",
            "E_mag_j",
            "p_elec",
            "p_mag",
        ]
    )


def empty_surface_q_summary() -> pd.DataFrame:
    """Return the empty surface-Q summary contract."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "surface_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "interface_type",
            "p_surf",
            "q_surf",
            "inverse_q",
        ]
    )


def empty_port_epr_summary() -> pd.DataFrame:
    """Return the empty port-EPR summary contract."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "port_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "postprocessing_type",
            "p_port",
            "abs_p_port",
            "abs_p_port_fraction",
        ]
    )


__all__ = [
    "REPORT_SOURCE_COLUMNS",
    "coerce_report_run_summary",
    "empty_dielectric_interface_summary",
    "empty_domain_energy_summary",
    "empty_domain_material_summary",
    "empty_port_epr_summary",
    "empty_surface_q_summary",
    "report_source_row",
]
