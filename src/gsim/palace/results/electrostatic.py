"""Electrostatic typed result data for Palace reports.

This module owns electrostatic terminal-matrix values produced by Palace. A
terminal matrix records how conductor charges respond when each terminal is
excited in turn; convergence views show whether those extracted capacitances
stabilize across adaptive mesh passes.

It does not discover files, generate capacitance postprocessing config, or
compose problem reports. Those layers live in ``gsim.palace.resolve``,
``gsim.palace.mesh``, and ``gsim.palace.results`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gsim.palace.display import DisplayValue, PlotlyFigure, make_trace_figure

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class TerminalMatrix:
    """One Palace electrostatic terminal matrix with named conductor labels."""

    source_path: Path
    matrix_kind: str
    dataframe: pd.DataFrame
    terminal_names: tuple[str, ...]
    source_unit: str
    display_scale: float
    display_unit: str

    @property
    def display_dataframe(self) -> pd.DataFrame:
        """Return matrix values in engineering units for compact inspection."""
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
        """Return one row per terminal pair for AMR convergence alignment.

        Palace writes each pass as a square matrix. Convergence plots need the
        same ``row terminal -> column terminal`` element lined up across passes,
        so the matrix is flattened into stable element records.
        """
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

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the compact matrix table a notebook user should inspect."""
        return {f"terminal_{self.matrix_kind}_table": self.display_dataframe}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return the default terminal-matrix display surface."""
        return cast("dict[str, DisplayValue]", self.tables())


@dataclass(frozen=True)
class TerminalMatrixConvergence:
    """Per-element AMR convergence for matrices derived from several passes.

    The loader starts from multiple Palace terminal matrices, then stores a
    flattened history because every useful convergence question is about one
    matrix element tracked across adaptive passes.
    """

    matrix_kind: str
    history: pd.DataFrame
    pass_summary: pd.DataFrame

    def plot_matrix_convergence(self) -> PlotlyFigure | None:
        """Plot capacitance value convergence for every terminal pair.

        Each trace is one matrix element. A stable trace means the extracted
        capacitance or mutual capacitance is no longer moving materially as the
        adaptive mesh refines.
        """
        required = {"pass_index", "element", "display_value"}
        if self.history.empty or not required.issubset(self.history.columns):
            return None

        traces: list[dict[str, Any]] = []
        for element, frame in self.history.groupby("element", sort=True):
            traces.append(
                {
                    "x": frame["pass_index"],
                    "y": frame["display_value"],
                    "name": str(element),
                    "mode": "markers+lines",
                }
            )
        if not traces:
            return None

        display_unit = _first_nonempty_value(self.history, "display_unit")
        y_title = (
            f"{self.matrix_kind} ({display_unit})"
            if display_unit is not None
            else self.matrix_kind
        )

        return make_trace_figure(
            traces,
            title=f"{self.matrix_kind} matrix convergence",
            x_title="Adaptive pass",
            y_title=y_title,
        )

    def plot_pass_delta(self) -> PlotlyFigure | None:
        """Plot per-element relative change between consecutive AMR passes."""
        required = {
            "pass_index",
            "element",
            "abs_relative_delta_to_previous_percent",
        }
        if self.history.empty or not required.issubset(self.history.columns):
            return None

        traces: list[dict[str, Any]] = []
        import pandas as pd

        for element, frame in self.history.groupby("element", sort=True):
            series = cast(
                "pd.Series",
                pd.to_numeric(
                    frame["abs_relative_delta_to_previous_percent"],
                    errors="coerce",
                ),
            )
            finite = cast(
                "pd.Series",
                series.notna() & (series != float("inf")) & (series != float("-inf")),
            )
            if not finite.any():
                continue
            ordered = frame.assign(_delta=series).loc[finite].sort_values("pass_index")
            traces.append(
                {
                    "x": ordered["pass_index"],
                    "y": ordered["_delta"],
                    "name": str(element),
                    "mode": "markers+lines",
                }
            )
        if not traces:
            return None

        return make_trace_figure(
            traces,
            title=f"{self.matrix_kind} convergence delta",
            x_title="Adaptive pass",
            y_title="%",
        )

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return plots that make AMR convergence inspectable by element."""
        figures: dict[str, PlotlyFigure] = {}
        matrix_figure = self.plot_matrix_convergence()
        delta_figure = self.plot_pass_delta()
        if matrix_figure is not None:
            figures[f"terminal_{self.matrix_kind}_convergence_trace_plot"] = (
                matrix_figure
            )
        if delta_figure is not None:
            figures[f"terminal_{self.matrix_kind}_delta_trace_plot"] = delta_figure
        return figures

    def visualize(self) -> dict[str, DisplayValue]:
        """Return default convergence plots for notebook inspection."""
        return cast("dict[str, DisplayValue]", self.figures())


def _first_nonempty_value(frame: pd.DataFrame, column: str) -> str | None:
    """Return the first non-empty string-like value from a dataframe column."""
    if column not in frame.columns:
        return None
    for value in frame[column]:
        text = str(value)
        if value is not None and text and text.lower() != "nan":
            return text
    return None


__all__ = ["TerminalMatrix", "TerminalMatrixConvergence"]
