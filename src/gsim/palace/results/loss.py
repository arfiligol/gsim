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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from gsim.palace._shared import (
    is_missing_value,
    optional_float,
    optional_int,
    optional_str,
)
from gsim.palace.display import DisplayValue, PlotlyFigure

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
class EprLossRecord:
    """One domain or surface EPR loss contribution in shared typed columns."""

    channel: LossChannel
    source_index: int | None
    source_name: str | None
    mode_index: int | None
    frequency_ghz: float | None
    participation: float | None
    loss_tangent: float | None
    inverse_q: float | None
    q_equivalent: float | None
    gamma_rad_per_s: float | None
    gamma_hz: float | None
    gamma_mhz: float | None
    gamma_per_us: float | None
    t1_us: float | None
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return this record as a JSON-friendly mapping."""
        return {
            "channel": self.channel,
            "source_index": self.source_index,
            "source_name": self.source_name,
            "mode_index": self.mode_index,
            "frequency_ghz": self.frequency_ghz,
            "participation": self.participation,
            "loss_tangent": self.loss_tangent,
            "inverse_q": self.inverse_q,
            "q_equivalent": self.q_equivalent,
            "gamma_rad_per_s": self.gamma_rad_per_s,
            "gamma_hz": self.gamma_hz,
            "gamma_mhz": self.gamma_mhz,
            "gamma_per_us": self.gamma_per_us,
            "t1_us": self.t1_us,
        }


@dataclass(frozen=True, kw_only=True)
class EprLossTable:
    """Shared typed table for domain and surface EPR loss contributions.

    Args:
        dataframe: Existing dataframe returned by the Palace report loaders.
        channel: Loss channel represented by this table.
        index_column: Channel-specific source index column.
        participation_column: Channel-specific participation column.

    The input dataframe is copied so callers keep ownership of the original
    table. Missing optional metric columns remain absent from dataframe exports
    and appear as ``None`` in row records.
    """

    dataframe: pd.DataFrame
    channel: LossChannel
    index_column: str
    participation_column: str

    def __post_init__(self) -> None:
        """Copy the incoming frame to keep wrapper methods non-mutating."""
        object.__setattr__(self, "dataframe", self.dataframe.copy())

    @property
    def empty(self) -> bool:
        """Return whether this loss table has no rows."""
        return bool(self.dataframe.empty)

    @property
    def metric_columns(self) -> tuple[str, ...]:
        """Return common EPR/loss metric columns present in this table."""
        return tuple(
            column for column in _COMMON_METRIC_COLUMNS if column in self.dataframe
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the original report dataframe."""
        return self.dataframe.copy()

    def save_csv(self, path: str | Path) -> Path:
        """Persist the original report table as CSV.

        Args:
            path: Destination CSV path.

        Returns:
            Written path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(output_path, index=False)
        return output_path

    def metrics_dataframe(self) -> pd.DataFrame:
        """Return this loss table normalized to common EPR/loss column names."""
        import pandas as pd

        columns = ["channel", *_COMMON_CONTEXT_COLUMNS, *_COMMON_METRIC_COLUMNS]
        if self.dataframe.empty:
            return pd.DataFrame(columns=columns)

        rows = [record.to_dict() for record in self.to_records()]
        return cast("pd.DataFrame", pd.DataFrame.from_records(rows, columns=columns))

    def to_records(self) -> tuple[EprLossRecord, ...]:
        """Return row records with channel-specific columns normalized."""
        return tuple(
            self._record_from_row(cast("Mapping[str, Any]", row))
            for _, row in self.dataframe.iterrows()
        )

    def plot_inverse_q(
        self,
        *,
        label_column: str | None = None,
        title: str | None = None,
    ) -> PlotlyFigure | None:
        """Plot inverse-Q contributions as a Plotly bar chart.

        Args:
            label_column: Optional column from :meth:`metrics_dataframe` to use
                as the x-axis label.
            title: Optional figure title.

        Returns:
            Plotly figure when inverse-Q data exists, otherwise ``None``.
        """
        if self.empty or "inverse_q" not in self.dataframe:
            return None

        from gsim.palace.display import make_bar_figure

        frame = self.metrics_dataframe()
        labels = label_column or _first_existing_column(
            frame,
            (
                "source_name",
                "mode_index",
                "frequency_ghz",
                "source_index",
                "sample_value",
            ),
        )
        if labels is None:
            labels = "channel"
        return make_bar_figure(
            (
                {
                    "x": frame[labels],
                    "y": frame["inverse_q"],
                    "name": f"{self.channel} inverse Q",
                },
            ),
            title=title or f"{self.channel.title()} inverse-Q contributions",
            x_title=labels,
            y_title="inverse Q",
        )

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return original and normalized loss tables for this channel."""
        return {
            f"{self.channel}_loss_table": self.to_dataframe(),
            f"{self.channel}_epr_loss_table": self.metrics_dataframe(),
        }

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return figures owned by this loss channel."""
        figure = self.plot_inverse_q()
        if figure is None:
            return {}
        return {f"{self.channel}_inverse_q_bar_plot": figure}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default loss-channel tables and figures."""
        return {**self.tables(), **self.figures()}

    def _record_from_row(self, row: Mapping[str, Any]) -> EprLossRecord:
        index_value = row.get("source_index")
        if is_missing_value(index_value):
            index_value = row.get(self.index_column)
        return EprLossRecord(
            channel=self.channel,
            source_index=optional_int(index_value),
            source_name=optional_str(row.get("source_name")),
            mode_index=optional_int(row.get("mode_index")),
            frequency_ghz=optional_float(row.get("frequency_ghz")),
            participation=optional_float(row.get(self.participation_column)),
            loss_tangent=optional_float(row.get("loss_tangent")),
            inverse_q=optional_float(row.get("inverse_q")),
            q_equivalent=optional_float(row.get("q_equivalent")),
            gamma_rad_per_s=optional_float(row.get("gamma_rad_per_s")),
            gamma_hz=optional_float(row.get("gamma_hz")),
            gamma_mhz=optional_float(row.get("gamma_mhz")),
            gamma_per_us=optional_float(row.get("gamma_per_us")),
            t1_us=optional_float(row.get("t1_us")),
            raw=dict(row),
        )


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

    @classmethod
    def from_csv(cls, path: str | Path) -> DomainLoss:
        """Load a domain-loss table from a CSV file."""
        import pandas as pd

        return cls(cast("pd.DataFrame", pd.read_csv(path)))


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

    @classmethod
    def from_csv(cls, path: str | Path) -> SurfaceLoss:
        """Load a surface-loss table from a CSV file."""
        import pandas as pd

        return cls(cast("pd.DataFrame", pd.read_csv(path)))


@dataclass(frozen=True, kw_only=True)
class LossBudget:
    """Typed aggregate inverse-Q budget for a Palace report.

    The budget table sums domain and surface inverse-Q contributions using the
    grouping chosen by the owning Problem Type Report. It keeps the
    loader-produced dataframe intact while adding storage and plotting helpers
    used by report views.
    """

    dataframe: pd.DataFrame

    def __post_init__(self) -> None:
        """Copy the incoming frame to keep wrapper methods non-mutating."""
        object.__setattr__(self, "dataframe", self.dataframe.copy())

    @property
    def empty(self) -> bool:
        """Return whether this budget has no rows."""
        return bool(self.dataframe.empty)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the original budget dataframe."""
        return self.dataframe.copy()

    def save_csv(self, path: str | Path) -> Path:
        """Persist the budget table as CSV.

        Args:
            path: Destination CSV path.

        Returns:
            Written path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(output_path, index=False)
        return output_path

    @classmethod
    def from_csv(cls, path: str | Path) -> LossBudget:
        """Load a loss-budget table from a CSV file."""
        import pandas as pd

        return cls(dataframe=cast("pd.DataFrame", pd.read_csv(path)))

    def plot_inverse_q(self, *, title: str | None = None) -> PlotlyFigure | None:
        """Plot domain and surface inverse-Q budget contributions.

        Args:
            title: Optional figure title.

        Returns:
            Plotly figure when budget data exists, otherwise ``None``.
        """
        if self.empty or "total_inverse_q_sum" not in self.dataframe:
            return None

        from gsim.palace.display import make_bar_figure

        frame = self.dataframe
        label_column = _first_existing_column(
            frame,
            ("mode_index", "frequency_ghz", "source_index", "sample_value"),
        )
        if label_column is None:
            label_column = frame.index.name or "row"
            x_values = frame.index
        else:
            x_values = frame[label_column]

        bars: list[dict[str, Any]] = []
        if "domain_inverse_q_sum" in frame.columns:
            bars.append(
                {"x": x_values, "y": frame["domain_inverse_q_sum"], "name": "domain"}
            )
        if "surface_inverse_q_sum" in frame.columns:
            bars.append(
                {
                    "x": x_values,
                    "y": frame["surface_inverse_q_sum"],
                    "name": "surface",
                }
            )
        return make_bar_figure(
            bars,
            title=title or "Loss budget",
            x_title=label_column,
            y_title="inverse Q",
            barmode="stack",
        )

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return visual figures owned by this loss budget."""
        figure = self.plot_inverse_q()
        return {} if figure is None else {"loss_budget_bar_plot": figure}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default loss-budget tables and figures."""
        return {"loss_budget_table": self.to_dataframe(), **self.figures()}


@dataclass(frozen=True, kw_only=True)
class ReportLoss:
    """Typed aggregate loss surface attached to a Problem Type Report."""

    domain: DomainLoss
    surface: SurfaceLoss
    budget: LossBudget

    @property
    def empty(self) -> bool:
        """Return whether all loss tables are empty."""
        return self.domain.empty and self.surface.empty and self.budget.empty

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return report-loss tables by stable view name."""
        return {
            "domain_loss_table": self.domain.to_dataframe(),
            "surface_loss_table": self.surface.to_dataframe(),
            "loss_budget_table": self.budget.to_dataframe(),
        }

    def metrics_tables(self) -> dict[str, pd.DataFrame]:
        """Return normalized EPR/loss metric tables by stable view name."""
        return {
            "domain_epr_loss_table": self.domain.metrics_dataframe(),
            "surface_epr_loss_table": self.surface.metrics_dataframe(),
        }

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return loss figures owned by this aggregate loss object."""
        figures: dict[str, PlotlyFigure] = {}
        domain_figure = self.domain.plot_inverse_q()
        surface_figure = self.surface.plot_inverse_q()
        budget_figure = self.budget.plot_inverse_q()
        if domain_figure is not None:
            figures["domain_inverse_q_bar_plot"] = domain_figure
        if surface_figure is not None:
            figures["surface_inverse_q_bar_plot"] = surface_figure
        if budget_figure is not None:
            figures["loss_budget_bar_plot"] = budget_figure
        return figures

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default loss tables, normalized metrics, and figures."""
        return {**self.tables(), **self.metrics_tables(), **self.figures()}


def _first_existing_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


__all__ = [
    "DOMAIN_LOSS_COLUMNS",
    "LOSS_BUDGET_COLUMNS",
    "SURFACE_LOSS_COLUMNS",
    "DomainLoss",
    "EprLossRecord",
    "EprLossTable",
    "LossBudget",
    "ReportLoss",
    "SurfaceLoss",
]
