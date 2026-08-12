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

from typing import ClassVar

from gsim.palace.results.base import NamedTableResult


class PostprocessingTable(NamedTableResult):
    """Base typed wrapper for one Palace postprocessing table.

    Subclasses provide a stable ``table_name``. The wrapper prevents problem
    reports from directly publishing raw DataFrames as result items while
    keeping table-only data lightweight and explicit.
    """

    table_name: ClassVar[str]


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
