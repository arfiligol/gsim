"""Electrostatic Problem Type Report model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gsim.palace._shared import finite_min
from gsim.palace.display import VisualizationProvider
from gsim.palace.results.electrostatic import TerminalMatrix, TerminalMatrixConvergence
from gsim.palace.results.loss import DomainLoss, LossBudget, ReportLoss, SurfaceLoss
from gsim.palace.results.postprocessing import DomainEnergy, SurfaceQ
from gsim.palace.results.reports.base import BasePalaceReport


@dataclass(frozen=True, kw_only=True)
class ElectrostaticReport(BasePalaceReport):
    """Composed Palace Electrostatic report tables for notebook workflows.

    Electrostatic loss is source/sample grouped: it derives dielectric and
    material participation from electrostatic postprocessing tables, optionally
    using an external frequency to express inverse-Q and equivalent loss rates.
    """

    terminal_c: TerminalMatrix
    terminal_cm: TerminalMatrix | None
    terminal_cinv: TerminalMatrix | None
    terminal_c_convergence: TerminalMatrixConvergence
    terminal_cm_convergence: TerminalMatrixConvergence
    terminal_cinv_convergence: TerminalMatrixConvergence
    domain_energy_result: DomainEnergy
    surface_q_result: SurfaceQ
    surface_interface_summary: pd.DataFrame
    domain_epr_loss: DomainLoss
    surface_epr_loss: SurfaceLoss
    surface_epr_convergence: pd.DataFrame
    loss_budget_result: LossBudget

    @property
    def problem_type(self) -> str:
        """Return the Palace problem type represented by this report."""
        return "Electrostatic"

    @property
    def capacitance(self) -> TerminalMatrix:
        """Return the capacitance matrix report."""
        return self.terminal_c

    @property
    def mutual_capacitance(self) -> TerminalMatrix | None:
        """Return the mutual capacitance matrix report when present."""
        return self.terminal_cm

    @property
    def inverse_capacitance(self) -> TerminalMatrix | None:
        """Return the inverse capacitance matrix report when present."""
        return self.terminal_cinv

    @property
    def matrix_history(self) -> dict[str, pd.DataFrame]:
        """Return terminal matrix convergence tables by matrix kind."""
        return {
            "C": self.terminal_c_history,
            "Cm": self.terminal_cm_history,
            "Cinv": self.terminal_cinv_history,
        }

    @property
    def matrix_pass_summary(self) -> dict[str, pd.DataFrame]:
        """Return terminal matrix pass summaries by matrix kind."""
        return {
            "C": self.terminal_c_pass_summary,
            "Cm": self.terminal_cm_pass_summary,
            "Cinv": self.terminal_cinv_pass_summary,
        }

    @property
    def terminal_c_history(self) -> pd.DataFrame:
        """Return capacitance convergence history as a dataframe view."""
        return self.terminal_c_convergence.history

    @property
    def terminal_cm_history(self) -> pd.DataFrame:
        """Return mutual-capacitance convergence history as a dataframe view."""
        return self.terminal_cm_convergence.history

    @property
    def terminal_cinv_history(self) -> pd.DataFrame:
        """Return inverse-capacitance convergence history as a dataframe view."""
        return self.terminal_cinv_convergence.history

    @property
    def terminal_c_pass_summary(self) -> pd.DataFrame:
        """Return capacitance pass summary as a dataframe view."""
        return self.terminal_c_convergence.pass_summary

    @property
    def terminal_cm_pass_summary(self) -> pd.DataFrame:
        """Return mutual-capacitance pass summary as a dataframe view."""
        return self.terminal_cm_convergence.pass_summary

    @property
    def terminal_cinv_pass_summary(self) -> pd.DataFrame:
        """Return inverse-capacitance pass summary as a dataframe view."""
        return self.terminal_cinv_convergence.pass_summary

    @property
    def domain_energy(self) -> pd.DataFrame:
        """Return domain energy rows as a dataframe view."""
        return self.domain_energy_result.dataframe

    @property
    def surface_q(self) -> pd.DataFrame:
        """Return surface participation rows as a dataframe view."""
        return self.surface_q_result.dataframe

    @property
    def domain_loss(self) -> pd.DataFrame:
        """Return Electrostatic domain/material EPR loss rows."""
        return self.domain_epr_loss.to_dataframe()

    @property
    def surface_loss(self) -> pd.DataFrame:
        """Return Electrostatic surface/interface EPR loss rows."""
        return self.surface_epr_loss.to_dataframe()

    @property
    def loss_budget(self) -> pd.DataFrame:
        """Return Electrostatic source/sample aggregate loss-budget rows."""
        return self.loss_budget_result.to_dataframe()

    @property
    def epr_losses(self) -> tuple[DomainLoss, SurfaceLoss]:
        """Return typed Electrostatic domain and surface EPR loss tables."""
        return (self.domain_epr_loss, self.surface_epr_loss)

    @property
    def loss(self) -> ReportLoss:
        """Return typed Electrostatic domain, surface, and budget loss data."""
        return ReportLoss(
            domain=self.domain_epr_loss,
            surface=self.surface_epr_loss,
            surface_convergence=self.surface_epr_convergence,
            budget=self.loss_budget_result,
        )

    def loss_budget_column_min(self, column: str) -> float | None:
        """Return a finite minimum from the Electrostatic loss budget."""
        loss_budget = self.loss_budget
        if loss_budget.empty or column not in loss_budget.columns:
            return None
        return finite_min(loss_budget[column].to_numpy())

    def typed_results(self) -> tuple[VisualizationProvider, ...]:
        """Return Typed Data owned by this Electrostatic report."""
        typed_data: list[VisualizationProvider] = [
            *super().typed_results(),
            self.terminal_c,
        ]
        if self.terminal_cm is not None:
            typed_data.append(self.terminal_cm)
        if self.terminal_cinv is not None:
            typed_data.append(self.terminal_cinv)
        typed_data.extend(
            convergence
            for convergence in (
                self.terminal_c_convergence,
                self.terminal_cm_convergence,
                self.terminal_cinv_convergence,
            )
            if not convergence.history.empty
        )
        if not self.domain_energy.empty:
            typed_data.append(self.domain_energy_result)
        if not self.surface_q.empty:
            typed_data.append(self.surface_q_result)
        if not self.loss.empty:
            typed_data.append(self.loss)
        return tuple(typed_data)

    def to_sweep_metrics(self) -> dict[str, Any]:
        """Return compact Electrostatic metrics for sweep-summary tables."""
        rows, columns = self.capacitance.dataframe.shape
        return {
            **super().to_sweep_metrics(),
            "terminal_count": len(self.capacitance.terminal_names),
            "capacitance_row_count": int(rows),
            "capacitance_column_count": int(columns),
            "has_mutual_capacitance": self.mutual_capacitance is not None,
            "has_inverse_capacitance": self.inverse_capacitance is not None,
            "domain_energy_rows": len(self.domain_energy),
            "surface_q_rows": len(self.surface_q),
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
            & (~self.sources["name"].str.startswith("iteration*/")),
            "name",
        ]
        return tuple(str(name) for name in missing)


__all__ = ["ElectrostaticReport"]
