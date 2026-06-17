"""Electrostatic typed result data for Palace reports.

This module owns terminal-matrix data produced by Palace Electrostatic solves.
The object stores SI-unit matrix values, terminal labels, display scaling, and
table-oriented exports used by reports and notebook views.

It does not discover files, generate capacitance postprocessing config, or
compose problem reports. Those layers live in ``gsim.palace.resolve``,
``gsim.palace.mesh``, and ``gsim.palace.results`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsim.palace.display import DisplayValue, PlotlyFigure

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class TerminalMatrix:
    """Palace electrostatic terminal matrix with named terminals."""

    source_path: Path
    matrix_kind: str
    dataframe: pd.DataFrame
    terminal_names: tuple[str, ...]
    source_unit: str
    display_scale: float
    display_unit: str

    @property
    def display_dataframe(self) -> pd.DataFrame:
        """Return a copy scaled for notebook display."""
        frame = self.dataframe * self.display_scale
        frame.attrs.update(
            {
                "matrix_kind": self.matrix_kind,
                "source_unit": self.source_unit,
                "display_scale": self.display_scale,
                "display_unit": self.display_unit,
                "csv_path": str(self.source_path),
            }
        )
        return frame

    def to_long_dataframe(self) -> pd.DataFrame:
        """Return a long-form terminal-pair table."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for row_offset, row_name in enumerate(self.terminal_names, start=1):
            for col_offset, col_name in enumerate(self.terminal_names, start=1):
                value = float(self.dataframe.loc[row_name, col_name])
                rows.append(
                    {
                        "matrix_kind": self.matrix_kind,
                        "matrix_csv_path": str(self.source_path),
                        "row_index": row_offset,
                        "column_index": col_offset,
                        "row_terminal": row_name,
                        "column_terminal": col_name,
                        "element": f"{row_name} -> {col_name}",
                        "is_diagonal": row_offset == col_offset,
                        "value_si": value,
                        "source_unit": self.source_unit,
                        "display_value": value * self.display_scale,
                        "display_unit": self.display_unit,
                        "display_scale": self.display_scale,
                    }
                )
        return pd.DataFrame.from_records(rows)

    def plot_heatmap(self) -> PlotlyFigure:
        """Plot the terminal matrix as a generic heatmap."""
        from gsim.palace.display import make_heatmap_figure

        frame = self.display_dataframe
        return make_heatmap_figure(
            frame.to_numpy(),
            x=tuple(frame.columns),
            y=tuple(frame.index),
            title=f"{self.matrix_kind} terminal matrix",
            x_title="column terminal",
            y_title="row terminal",
            colorbar_title=self.display_unit,
        )

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return visual figures owned by this terminal matrix."""
        return {f"terminal_{self.matrix_kind}_heatmap": self.plot_heatmap()}

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return tabular views owned by this terminal matrix."""
        return {
            f"terminal_{self.matrix_kind}_table": self.display_dataframe,
            f"terminal_{self.matrix_kind}_long_table": self.to_long_dataframe(),
        }

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default terminal-matrix tables and figures."""
        return {**self.tables(), **self.figures()}


@dataclass(frozen=True)
class TerminalMatrixConvergence:
    """Electrostatic terminal-matrix AMR history and pass summary."""

    matrix_kind: str
    history: pd.DataFrame
    pass_summary: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return terminal-matrix convergence tables owned by this object."""
        return {
            f"terminal_{self.matrix_kind}_history_table": self.history,
            f"terminal_{self.matrix_kind}_pass_summary_table": self.pass_summary,
        }

    def plot_pass_delta(self) -> PlotlyFigure | None:
        """Plot max terminal-matrix pass deltas when summary rows exist."""
        if self.pass_summary.empty or "pass_index" not in self.pass_summary.columns:
            return None

        y_columns = (
            "max_abs_display_delta_to_previous",
            "max_abs_display_delta_to_final",
            "max_abs_relative_delta_to_previous_percent",
            "max_abs_relative_delta_to_final_percent",
        )
        traces: list[dict[str, Any]] = []
        for column in y_columns:
            if column not in self.pass_summary.columns:
                continue
            series = self.pass_summary[column]
            if not series.notna().any():
                continue
            traces.append(
                {
                    "x": self.pass_summary["pass_index"],
                    "y": series,
                    "name": column,
                    "mode": "markers+lines",
                }
            )
        if not traces:
            return None

        from gsim.palace.display import make_trace_figure

        return make_trace_figure(
            traces,
            title=f"{self.matrix_kind} terminal-matrix convergence",
            x_title="AMR pass",
            y_title="delta",
        )

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return terminal-matrix convergence figures owned by this object."""
        figure = self.plot_pass_delta()
        if figure is None:
            return {}
        return {f"terminal_{self.matrix_kind}_convergence_trace_plot": figure}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default terminal-matrix convergence tables and figures."""
        return {**self.tables(), **self.figures()}


__all__ = ["TerminalMatrix", "TerminalMatrixConvergence"]
