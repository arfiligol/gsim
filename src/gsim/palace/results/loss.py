"""Typed derived EPR/loss results for Palace reports.

This module owns loss-oriented Typed Data derived from Palace postprocessing
tables and material/interface metadata. Domain and surface losses use
different Palace source columns, but both expose a shared typed column language:
participation, loss tangent, inverse quality factor, equivalent quality factor,
decay rates, and T1.

Raw table-only Palace reports such as ``domain-E.csv``, ``surface-Q.csv``, and
``port-EPR.csv`` live in ``gsim.palace.results.postprocessing``. This module
starts after resolve has loaded and normalized those inputs into loss summary
dataframes.

The module does not parse Palace files or decide which postprocessing reports
are required. Primitive CSV/config loading stays in ``gsim.palace.resolve``;
generic plot/table helpers stay in ``gsim.palace.display``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from gsim.palace._shared import (
    is_missing_value,
    optional_float,
    optional_int,
    optional_str,
)
from gsim.palace.display import DisplayValue, PlotlyFigure, make_trace_figure
from gsim.palace.results.base import DataFrameResult, NamedTableResult

if TYPE_CHECKING:
    import pandas as pd

LossChannel = Literal["domain", "surface"]

DOMAIN_LOSS_COLUMNS = (
    "row_index",
    "sample_column",
    "sample_value",
    "mode_index",
    "frequency_ghz",
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
    "material_attribute",
    "material_attributes",
    "material_name",
    "material_permittivity",
    "material_loss_tangent",
    "material_conductivity",
    "material_permeability",
    "material_axes",
    "loss_tangent",
    "inverse_q",
    "q_equivalent",
    "gamma_rad_per_s",
    "gamma_per_us",
    "gamma_hz",
    "gamma_mhz",
    "t1_us",
)
SURFACE_LOSS_COLUMNS = (
    "row_index",
    "sample_column",
    "sample_value",
    "mode_index",
    "frequency_ghz",
    "surface_index",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "interface_type",
    "loss_channel",
    "source_entry_name",
    "surface_epr_summary_kind",
    "surface_epr_exclude_below_um",
    "source_aware_surface_epr_group_names",
    "surface_epr_band_names",
    "surface_epr_band_min_um",
    "surface_epr_band_max_um",
    "surface_epr_band_label",
    "preset_name",
    "preset_source",
    "p_surf",
    "q_surf",
    "inverse_q",
    "q_equivalent",
    "surface_attribute",
    "surface_attributes",
    "thickness",
    "permittivity",
    "loss_tangent",
    "gamma_rad_per_s",
    "gamma_per_us",
    "gamma_hz",
    "gamma_mhz",
    "t1_us",
)
LOSS_BUDGET_COLUMNS = (
    "mode_index",
    "frequency_ghz",
    "q_eig",
    "inverse_q_eig",
    "domain_inverse_q_sum",
    "surface_inverse_q_sum",
    "total_inverse_q_sum",
    "eig_with_surface_inverse_q_sum",
    "q_total",
    "q_eig_with_surface",
    "domain_vs_eig_relative_error",
    "gamma_rad_per_s",
    "gamma_per_us",
    "gamma_hz",
    "gamma_mhz",
    "t1_us",
)
LOSS_CHANNEL_BUDGET_COLUMNS = (
    "source_index",
    "mode_index",
    "sample_column",
    "sample_value",
    "frequency_ghz",
    "source_entry_name",
    "surface_epr_summary_kind",
    "surface_epr_exclude_below_um",
    "loss_channel",
    "inverse_q",
    "loss_fraction",
)

_COMMON_CONTEXT_COLUMNS = (
    "mode_index",
    "source_index",
    "sample_column",
    "sample_value",
    "frequency_ghz",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "interface_type",
    "loss_channel",
    "source_entry_name",
    "surface_epr_summary_kind",
    "surface_epr_exclude_below_um",
    "surface_epr_band_label",
    "material_name",
    "preset_name",
)
_COMMON_METRIC_COLUMNS = (
    "participation",
    "loss_tangent",
    "inverse_q",
    "q_equivalent",
    "gamma_rad_per_s",
    "gamma_hz",
    "gamma_mhz",
    "gamma_per_us",
    "t1_us",
)


@dataclass(frozen=True, kw_only=True)
class EprLossTable(DataFrameResult):
    """Shared typed table for domain and surface EPR loss contributions.

    The canonical dataframe is the loader-produced source table for one loss
    channel. ``to_epr_dataframe()`` exposes the review-facing semantic view:
    every row becomes one domain or surface participation contribution with
    common EPR/loss columns.

    Args:
        dataframe: Existing dataframe returned by the Palace report loaders.
        channel: Loss channel represented by this table.
        index_column: Channel-specific source index column.
        participation_column: Channel-specific participation column.
    """

    channel: LossChannel
    index_column: str
    participation_column: str

    def to_epr_dataframe(self) -> pd.DataFrame:
        """Return one semantic EPR/loss contribution per source row.

        Domain and surface Palace reports use different raw columns. This view
        normalizes them to the shared review vocabulary: source, participation,
        loss tangent, inverse-Q, equivalent Q, decay rates, and T1.
        """
        import pandas as pd

        columns = ["channel", *_COMMON_CONTEXT_COLUMNS, *_COMMON_METRIC_COLUMNS]
        if self.dataframe.empty:
            return pd.DataFrame(columns=columns)

        rows = [
            self._epr_row(cast("Mapping[str, Any]", row.to_dict()))
            for _, row in self.dataframe.iterrows()
        ]
        return cast("pd.DataFrame", pd.DataFrame.from_records(rows, columns=columns))

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return source and semantic EPR tables for direct channel review."""
        return {
            f"{self.channel}_loss_table": self.to_dataframe(),
            f"{self.channel}_epr_loss_table": self.to_epr_dataframe(),
        }

    def visualize(self) -> dict[str, DisplayValue]:
        """Return direct channel tables for notebooks."""
        return self.tables()

    def _epr_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        index_value = row.get("source_index")
        if is_missing_value(index_value):
            index_value = row.get(self.index_column)
        return {
            "channel": self.channel,
            "source_index": optional_int(index_value),
            "source_name": optional_str(row.get("source_name")),
            "mode_index": optional_int(row.get("mode_index")),
            "frequency_ghz": optional_float(row.get("frequency_ghz")),
            "physical_name": optional_str(row.get("physical_name")),
            "entry_name": optional_str(row.get("entry_name")),
            "role": optional_str(row.get("role")),
            "interface_type": optional_str(row.get("interface_type")),
            "loss_channel": optional_str(row.get("loss_channel")),
            "source_entry_name": optional_str(row.get("source_entry_name")),
            "surface_epr_summary_kind": optional_str(
                row.get("surface_epr_summary_kind")
            ),
            "surface_epr_exclude_below_um": optional_float(
                row.get("surface_epr_exclude_below_um")
            ),
            "surface_epr_band_label": optional_str(
                row.get("surface_epr_band_label")
            ),
            "material_name": optional_str(row.get("material_name")),
            "preset_name": optional_str(row.get("preset_name")),
            "participation": optional_float(row.get(self.participation_column)),
            "loss_tangent": optional_float(row.get("loss_tangent")),
            "inverse_q": optional_float(row.get("inverse_q")),
            "q_equivalent": optional_float(row.get("q_equivalent")),
            "gamma_rad_per_s": optional_float(row.get("gamma_rad_per_s")),
            "gamma_hz": optional_float(row.get("gamma_hz")),
            "gamma_mhz": optional_float(row.get("gamma_mhz")),
            "gamma_per_us": optional_float(row.get("gamma_per_us")),
            "t1_us": optional_float(row.get("t1_us")),
        }


class DomainLoss(EprLossTable):
    """Typed EPR loss table for Palace domain/material contributions."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Create a domain-loss wrapper from ``summarize_domain_loss()`` output."""
        super().__init__(
            dataframe=dataframe,
            channel="domain",
            index_column="domain_index",
            participation_column="p_elec",
        )


class SurfaceLoss(EprLossTable):
    """Typed EPR loss table for Palace surface/interface contributions.

    Surface inverse-Q values preserve the Palace ``Q_surf``-derived report
    value when present. The wrapper does not recompute inverse-Q from
    participation and loss tangent because Palace may apply solver-side surface
    conventions that are not visible in the normalized table.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Create a surface-loss wrapper from ``summarize_surface_loss()`` output."""
        super().__init__(
            dataframe=dataframe,
            channel="surface",
            index_column="surface_index",
            participation_column="p_surf",
        )


@dataclass(frozen=True, kw_only=True)
class LossBudget(NamedTableResult):
    """Typed aggregate inverse-Q budget for a Palace report.

    The budget table sums domain and surface inverse-Q contributions using the
    grouping chosen by the owning Problem Type Report. It keeps the
    loader-produced dataframe intact for report views.
    """

    table_name = "loss_budget"


@dataclass(frozen=True, kw_only=True)
class ReportLoss:
    """Notebook-facing EPR/loss summary attached to a Problem Type Report."""

    domain: DomainLoss
    surface: SurfaceLoss
    budget: LossBudget
    surface_convergence: pd.DataFrame | None = None

    @property
    def empty(self) -> bool:
        """Return whether all loss tables are empty."""
        return (
            self.domain.empty
            and self.surface.empty
            and self.budget.empty
            and (
                self.surface_convergence is None or self.surface_convergence.empty
            )
        )

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the EPR tables that are meaningful without raw CSV context."""
        return {
            "domain_epr_summary_table": _compact_epr_summary_dataframe(
                self.domain.to_epr_dataframe()
            ),
            "surface_epr_summary_table": _compact_epr_summary_dataframe(
                self.surface.to_epr_dataframe(),
                surface_only=True,
            ),
            "loss_budget_table": self.budget.to_dataframe(),
        }

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return loss figures that need report-level context."""
        figure = _surface_epr_convergence_figure(self.surface_convergence)
        if figure is None:
            return {}
        figures = {"surface_epr_inset_convergence_trace_plot": figure}
        if (
            self.surface_convergence is not None
            and "interface_type" in self.surface_convergence.columns
        ):
            for interface_type in sorted(
                self.surface_convergence["interface_type"].dropna().astype(str).unique()
            ):
                interface_frame = self.surface_convergence.loc[
                    self.surface_convergence["interface_type"].astype(str)
                    == interface_type
                ]
                interface_figure = _surface_epr_convergence_figure(interface_frame)
                if interface_figure is None:
                    continue
                interface_figure.update_layout(
                    title=f"{interface_type} Surface EPR inset convergence"
                )
                slug = "".join(
                    char.lower() if char.isalnum() else "_"
                    for char in interface_type
                ).strip("_")
                figures[f"surface_epr_inset_convergence_{slug}_trace_plot"] = (
                    interface_figure
                )
        return figures

    def visualize(self) -> dict[str, DisplayValue]:
        """Return compact EPR/loss tables and convergence figures."""
        return cast("dict[str, DisplayValue]", {**self.tables(), **self.figures()})


def _compact_epr_summary_dataframe(
    frame: pd.DataFrame,
    *,
    surface_only: bool = False,
) -> pd.DataFrame:
    """Keep only EPR columns that answer source-local participation questions."""
    import pandas as pd

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source_index",
                "source_name",
                "participation",
                "participation_percent",
            ]
        )

    result = frame.copy()
    if surface_only:
        result = _surface_total_summary_rows(result)
    values = cast(
        "pd.Series",
        pd.to_numeric(result["participation"], errors="coerce"),
    ).abs()
    result["_participation_abs"] = values
    group_columns = (
        ["source_index"]
        if _has_non_null_column(result, "source_index")
        else [
            column
            for column in (
                "mode_index",
                "sample_column",
                "sample_value",
                "frequency_ghz",
            )
            if _has_non_null_column(result, column)
        ]
    )
    if group_columns:
        denominator = result.groupby(group_columns, dropna=False)[
            "_participation_abs"
        ].transform("sum")
        result["participation_percent"] = (
            values.div(denominator.where(denominator != 0)).fillna(0.0) * 100.0
        )
    else:
        denominator = float(values.sum())
        result["participation_percent"] = (
            0.0 if denominator == 0.0 else (values / denominator) * 100.0
        )

    columns = [
        column
        for column in (
            "source_index",
            "mode_index",
            "sample_value",
            "frequency_ghz",
            "source_name",
            "physical_name",
            "interface_type",
            *(
                ()
                if surface_only
                else (
                    "loss_channel",
                    "source_entry_name",
                    "surface_epr_summary_kind",
                    "surface_epr_exclude_below_um",
                    "surface_epr_band_label",
                )
            ),
            "material_name",
            "preset_name",
            "participation",
            "participation_percent",
            "loss_tangent",
            "inverse_q",
            "q_equivalent",
            "t1_us",
        )
        if _has_non_null_column(result, column)
    ]
    return cast("pd.DataFrame", result.loc[:, columns])


def _surface_total_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "surface_epr_summary_kind" not in frame.columns:
        return frame
    kinds = frame["surface_epr_summary_kind"].astype("string")
    return cast(
        "pd.DataFrame",
        frame.loc[kinds.isna() | (kinds == "") | (kinds == "total")].copy(),
    )


def _surface_epr_convergence_figure(
    frame: pd.DataFrame | None,
) -> PlotlyFigure | None:
    if frame is None or frame.empty:
        return None
    required = {
        "pass_index",
        "interface_type",
        "surface_epr_exclude_below_um",
        "surface_epr_abs",
    }
    if not required.issubset(frame.columns):
        return None

    import pandas as pd

    data = frame.copy()
    data["pass_index"] = pd.to_numeric(data["pass_index"], errors="coerce")
    data["surface_epr_abs"] = pd.to_numeric(data["surface_epr_abs"], errors="coerce")
    data["surface_epr_exclude_below_um"] = pd.to_numeric(
        data["surface_epr_exclude_below_um"],
        errors="coerce",
    ).fillna(0.0)
    if "surface_epr_band_min_um" not in data.columns:
        data["surface_epr_band_min_um"] = data["surface_epr_exclude_below_um"]
    data["surface_epr_band_min_um"] = pd.to_numeric(
        data["surface_epr_band_min_um"],
        errors="coerce",
    ).fillna(data["surface_epr_exclude_below_um"])
    if "surface_epr_band_max_um" not in data.columns:
        data["surface_epr_band_max_um"] = pd.NA
    else:
        data["surface_epr_band_max_um"] = pd.to_numeric(
            data["surface_epr_band_max_um"],
            errors="coerce",
        )
    if "surface_epr_summary_kind" not in data.columns:
        data["surface_epr_summary_kind"] = "total"
    data["surface_epr_summary_kind"] = data["surface_epr_summary_kind"].fillna(
        "total"
    )
    data = data.dropna(subset=["pass_index", "surface_epr_abs"])
    if data.empty:
        return None

    traces = []
    for (interface_type, kind, _inset_um, min_um, max_um), group in data.groupby(
        [
            "interface_type",
            "surface_epr_summary_kind",
            "surface_epr_exclude_below_um",
            "surface_epr_band_min_um",
            "surface_epr_band_max_um",
        ],
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("pass_index")
        traces.append(
            {
                "x": ordered["pass_index"],
                "y": ordered["surface_epr_abs"],
                "name": _surface_epr_trace_name(interface_type, kind, min_um, max_um),
                "mode": "lines+markers",
            }
        )
    if not traces:
        return None
    return make_trace_figure(
        traces,
        title="Surface EPR inset convergence",
        x_title="Adaptive pass",
        y_title="abs(Surface EPR)",
    )


def _surface_epr_trace_name(
    interface_type: object,
    kind: object,
    min_um: object,
    max_um: object,
) -> str:
    interface = "" if is_missing_value(interface_type) else str(interface_type)
    prefix = interface or "Surface EPR"
    if str(kind) == "total":
        return f"{prefix} total"
    min_nm = _surface_epr_nm(min_um)
    max_nm = _surface_epr_nm(max_um)
    if str(kind) == "band" and max_nm is not None:
        return f"{prefix} band {min_nm:g}-{max_nm:g} nm"
    return f"{prefix} inset >= {min_nm:g} nm"


def _surface_epr_nm(value: object) -> float | None:
    if is_missing_value(value):
        return None
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return 0.0


def _has_non_null_column(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    series = cast("pd.Series", frame[column])
    return bool(series.notna().any())

__all__ = [
    "DOMAIN_LOSS_COLUMNS",
    "LOSS_BUDGET_COLUMNS",
    "LOSS_CHANNEL_BUDGET_COLUMNS",
    "SURFACE_LOSS_COLUMNS",
    "DomainLoss",
    "EprLossTable",
    "LossBudget",
    "ReportLoss",
    "SurfaceLoss",
]
