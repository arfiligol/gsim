"""Eigenmode Problem Type Report model.

This module owns the semantic aggregate for a completed Palace Eigenmode run:
final mode rows, adaptive-pass convergence, optional indexed postprocessing
tables, and mode-indexed loss summaries. It does not parse raw solver files,
discover run folders, or define plotting primitives. Resolve assembly
constructs this report, Typed Data objects own their table/figure semantics,
and Display owns generic rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gsim.palace._shared import finite_min
from gsim.palace.display import VisualizationProvider
from gsim.palace.results.eigenmode import EigenmodeConvergence, Eigenmodes
from gsim.palace.results.loss import DomainLoss, LossBudget, ReportLoss, SurfaceLoss
from gsim.palace.results.postprocessing import (
    DomainEnergy,
    PortEpr,
    SurfaceQ,
)
from gsim.palace.results.reports.base import BasePalaceReport


@dataclass(frozen=True, kw_only=True)
class EigenmodeReport(BasePalaceReport):
    """Composed Palace Eigenmode report tables for notebook workflows.

    Eigenmode loss is mode-indexed: domain and surface loss rows are tied to
    eigenmodes, and the budget summarizes inverse-Q contributions per mode.
    Required eigenmode data is supplied by Resolve assembly; optional EPR/loss
    tables remain empty when Palace did not produce their source reports.
    """

    eigenmodes: Eigenmodes
    convergence: EigenmodeConvergence
    domain_energy_result: DomainEnergy
    surface_q_result: SurfaceQ
    surface_interface_summary: pd.DataFrame
    port_epr_result: PortEpr
    domain_epr_loss: DomainLoss
    surface_epr_loss: SurfaceLoss
    loss_budget_result: LossBudget

    @property
    def problem_type(self) -> str:
        """Return the Palace problem type represented by this report."""
        return "Eigenmode"

    @property
    def modes(self) -> pd.DataFrame:
        """Return final mode rows with notebook-facing column names."""
        return self.eigenmodes.to_report_dataframe()

    @property
    def mode_history(self) -> pd.DataFrame:
        """Return Eigenmode AMR mode history as a dataframe view."""
        return self.convergence.mode_history

    @property
    def pass_summary(self) -> pd.DataFrame:
        """Return Eigenmode AMR pass summary as a dataframe view."""
        return self.convergence.pass_summary

    @property
    def domain_energy(self) -> pd.DataFrame:
        """Return domain energy rows as a dataframe view."""
        return self.domain_energy_result.dataframe

    @property
    def surface_q(self) -> pd.DataFrame:
        """Return surface participation rows as a dataframe view."""
        return self.surface_q_result.dataframe

    @property
    def port_epr(self) -> pd.DataFrame:
        """Return port EPR rows as a dataframe view."""
        return self.port_epr_result.dataframe

    @property
    def domain_loss(self) -> pd.DataFrame:
        """Return Eigenmode domain/material EPR loss rows."""
        return self.domain_epr_loss.to_dataframe()

    @property
    def surface_loss(self) -> pd.DataFrame:
        """Return Eigenmode surface/interface EPR loss rows."""
        return self.surface_epr_loss.to_dataframe()

    @property
    def loss_budget(self) -> pd.DataFrame:
        """Return per-mode Eigenmode aggregate loss-budget rows."""
        return self.loss_budget_result.to_dataframe()

    @property
    def epr_losses(self) -> tuple[DomainLoss, SurfaceLoss]:
        """Return typed Eigenmode domain and surface EPR loss tables."""
        return (self.domain_epr_loss, self.surface_epr_loss)

    @property
    def loss(self) -> ReportLoss:
        """Return typed Eigenmode domain, surface, and budget loss data."""
        return ReportLoss(
            domain=self.domain_epr_loss,
            surface=self.surface_epr_loss,
            budget=self.loss_budget_result,
        )

    def loss_budget_column_min(self, column: str) -> float | None:
        """Return a finite minimum from the Eigenmode loss budget."""
        loss_budget = self.loss_budget
        if loss_budget.empty or column not in loss_budget.columns:
            return None
        return finite_min(loss_budget[column].to_numpy())

    def typed_results(self) -> tuple[VisualizationProvider, ...]:
        """Return Typed Data owned by this Eigenmode report."""
        typed_data: list[VisualizationProvider] = [
            *super().typed_results(),
            self.eigenmodes,
        ]
        if (
            not self.convergence.mode_history.empty
            or not self.convergence.pass_summary.empty
        ):
            typed_data.append(self.convergence)
        if not self.domain_energy.empty:
            typed_data.append(self.domain_energy_result)
        if not self.surface_q.empty:
            typed_data.append(self.surface_q_result)
        if not self.port_epr.empty:
            typed_data.append(self.port_epr_result)
        if not self.loss.empty:
            typed_data.append(self.loss)
        return tuple(typed_data)

    def to_sweep_metrics(self) -> dict[str, Any]:
        """Return compact Eigenmode metrics for sweep-summary tables."""
        return {
            **super().to_sweep_metrics(),
            "mode_count": int(self.eigenmodes.n_modes),
            "pass_count": len(self.pass_summary),
            "min_frequency_ghz": finite_min(self.eigenmodes.freq_real_ghz),
            "min_q": finite_min(self.eigenmodes.q),
            "domain_energy_rows": len(self.domain_energy),
            "surface_q_rows": len(self.surface_q),
            "port_epr_rows": len(self.port_epr),
            "loss_budget_rows": len(self.loss_budget),
            "min_q_total": self.loss_budget_column_min("q_total"),
        }

    @property
    def missing_reports(self) -> tuple[str, ...]:
        """Optional report names that were expected but absent."""
        if self.sources.empty:
            return ()
        missing = self.sources.loc[
            (~self.sources["required"])
            & (~self.sources["present"])
            & (self.sources["name"] != "iteration*/eig.csv"),
            "name",
        ]
        return tuple(str(name) for name in missing)


__all__ = ["EigenmodeReport"]
