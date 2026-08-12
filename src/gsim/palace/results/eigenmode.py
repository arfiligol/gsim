"""Eigenmode typed result data for Palace reports.

This module owns normalized Eigenmode CSV data after Palace has produced
solver output. It stores frequencies, quality factors, and solver error
columns in a typed object that reports and notebook views can reuse.

File discovery and CSV parsing stay in ``gsim.palace.resolve``. Problem-level
report composition stays in ``gsim.palace.results``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gsim.palace.display import DisplayValue, PlotlyFigure, make_trace_figure

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Eigenmodes:
    """Palace Eigenmode result table with normalized report columns."""

    source_path: Path
    dataframe: pd.DataFrame

    @property
    def n_modes(self) -> int:
        """Number of loaded eigenmodes."""
        return len(self.dataframe)

    @property
    def mode_indices(self) -> NDArray:
        """Palace mode indices as integers."""
        return self.dataframe["mode_index"].to_numpy(dtype=int)

    @property
    def freq_real_ghz(self) -> NDArray:
        """Real part of the complex eigenfrequency in GHz."""
        return self.dataframe["freq_real_ghz"].to_numpy(dtype=float)

    @property
    def freq_imag_ghz(self) -> NDArray:
        """Imaginary part of the complex eigenfrequency in GHz."""
        return self.dataframe["freq_imag_ghz"].to_numpy(dtype=float)

    @property
    def q(self) -> NDArray:
        """Palace-reported quality factor values."""
        return self.dataframe["q"].to_numpy(dtype=float)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the normalized eigenmode table."""
        frame = self.dataframe.copy()
        frame.attrs.update({"csv_path": str(self.source_path)})
        return frame

    def to_report_dataframe(self) -> pd.DataFrame:
        """Return notebook-facing column aliases for eigenmode reports."""
        frame = self.to_dataframe().rename(
            columns={
                "freq_real_ghz": "frequency_ghz",
                "freq_imag_ghz": "imaginary_frequency_ghz",
                "q": "q_factor",
                "error_backward": "backward_error",
                "error_absolute": "absolute_error",
            }
        )
        return frame

    def plot_frequencies(self) -> PlotlyFigure | None:
        """Plot final eigenmode frequencies by mode index."""
        if self.dataframe.empty:
            return None

        return make_trace_figure(
            (
                {
                    "x": self.mode_indices,
                    "y": self.freq_real_ghz,
                    "name": "frequency",
                    "mode": "markers+lines",
                },
            ),
            title="Eigenmode frequencies",
            x_title="mode index",
            y_title="frequency (GHz)",
        )

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return visual figures owned by this eigenmode data object."""
        figure = self.plot_frequencies()
        return {} if figure is None else {"eigenmode_frequencies_trace_plot": figure}

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return tabular views owned by this eigenmode data object."""
        return {"eigenmodes_table": self.to_report_dataframe()}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default Eigenmode tables and figures."""
        return {**self.tables(), **self.figures()}


@dataclass(frozen=True)
class EigenmodeConvergence:
    """Eigenmode AMR pass history and convergence summary tables."""

    mode_history: pd.DataFrame
    pass_summary: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return Eigenmode convergence tables owned by this object."""
        return {
            "eigenmode_mode_history_table": self.mode_history,
            "eigenmode_pass_summary_table": self.pass_summary,
        }

    def plot_frequency_convergence(self) -> PlotlyFigure | None:
        """Plot frequency convergence by mode when AMR history is available."""
        required = {"iteration_index", "frequency_ghz", "mode_index"}
        if self.mode_history.empty or not required.issubset(self.mode_history.columns):
            return None

        traces = []
        for mode_index, frame in self.mode_history.groupby("mode_index", sort=True):
            traces.append(
                {
                    "x": frame["iteration_index"],
                    "y": frame["frequency_ghz"],
                    "name": f"mode {mode_index}",
                    "mode": "markers+lines",
                }
            )
        return make_trace_figure(
            traces,
            title="Eigenmode frequency convergence",
            x_title="AMR pass",
            y_title="frequency (GHz)",
        )

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return Eigenmode convergence figures owned by this object."""
        figure = self.plot_frequency_convergence()
        if figure is None:
            return {}
        return {"eigenmode_frequency_convergence_trace_plot": figure}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default Eigenmode convergence tables and figures."""
        return {**self.tables(), **self.figures()}


__all__ = ["EigenmodeConvergence", "Eigenmodes"]
