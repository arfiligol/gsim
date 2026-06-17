"""Typed Palace postprocessing report tables.

This module owns already-loaded Palace postprocessing CSV reports that remain
table-only Typed Data. ``domain-E.csv`` and ``surface-Q.csv`` are shared
postprocessing tables; ``port-EPR.csv`` is Eigenmode lumped-port EPR data.
These objects preserve the source table semantics and expose the table through
``visualize()`` without deriving loss metrics.

Derived EPR/loss language such as inverse-Q budgets, equivalent Q, decay
rates, and T1 belongs to ``gsim.palace.results.loss``. This separation keeps
raw Palace postprocessing tables distinct from loss-oriented typed data.

The classes here do not discover files, read CSVs, build index maps, compose
reports, or choose notebook display policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from gsim.palace.display import DisplayValue, PlotlyFigure

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class PostprocessingTable:
    """Base typed wrapper for one Palace postprocessing table.

    Subclasses provide a stable ``table_name``. The wrapper prevents problem
    reports from directly publishing raw DataFrames as result items while
    keeping table-only data lightweight and explicit.
    """

    dataframe: pd.DataFrame

    table_name: ClassVar[str]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the table owned by this typed result object."""
        return {f"{self.table_name}_table": self.dataframe}

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return figures owned by this table result.

        Table-only postprocessing data has no default problem-neutral plot.
        Loss, convergence, and matrix objects provide plots where the
        semantics are clear.
        """
        return {}

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default table and figure views for this typed result."""
        return {**self.tables(), **self.figures()}


class DomainEnergy(PostprocessingTable):
    """Typed table for Palace ``domain-E.csv`` energy participation rows."""

    table_name: ClassVar[str] = "domain_energy"


class SurfaceQ(PostprocessingTable):
    """Typed table for Palace ``surface-Q.csv`` participation/Q rows."""

    table_name: ClassVar[str] = "surface_q"


class PortEpr(PostprocessingTable):
    """Typed table for Eigenmode ``port-EPR.csv`` lumped-port rows."""

    table_name: ClassVar[str] = "port_epr"


__all__ = [
    "DomainEnergy",
    "PortEpr",
    "PostprocessingTable",
    "SurfaceQ",
]
