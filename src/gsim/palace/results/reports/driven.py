"""Driven Problem Type Report model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gsim.palace._shared import finite_min
from gsim.palace.display import VisualizationProvider
from gsim.palace.results.driven import SParams
from gsim.palace.results.loss import DomainLoss, LossBudget, ReportLoss, SurfaceLoss
from gsim.palace.results.postprocessing import DomainEnergy, SurfaceQ
from gsim.palace.results.reports.base import BasePalaceReport


@dataclass(frozen=True, kw_only=True)
class DrivenReport(BasePalaceReport):
    """Composed Palace Driven report tables for notebook workflows.

    Driven loss is frequency-sample grouped: domain and surface postprocessing
    rows are interpreted against the sweep sample they came from. It is not a
    per-mode Eigenmode budget and it does not rely on Electrostatic's external
    frequency convention.
    """

    sparams: SParams
    domain_energy_result: DomainEnergy
    surface_q_result: SurfaceQ
    surface_interface_summary: pd.DataFrame
    domain_epr_loss: DomainLoss
    surface_epr_loss: SurfaceLoss
    loss_budget_result: LossBudget

    @property
    def problem_type(self) -> str:
        """Return the Palace problem type represented by this report."""
        return "Driven"

    @property
    def network(self) -> SParams:
        """Return the driven S-parameter network."""
        return self.sparams

    @property
    def domain_energy(self) -> pd.DataFrame:
        """Return domain energy rows as a dataframe view."""
        return self.domain_energy_result.dataframe

    @property
    def surface_q(self) -> pd.DataFrame:
        """Return surface-Q rows as a dataframe view."""
        return self.surface_q_result.dataframe

    @property
    def domain_loss(self) -> pd.DataFrame:
        """Return Driven frequency-sample domain/material EPR loss rows."""
        return self.domain_epr_loss.to_dataframe()

    @property
    def surface_loss(self) -> pd.DataFrame:
        """Return Driven frequency-sample surface/interface EPR loss rows."""
        return self.surface_epr_loss.to_dataframe()

    @property
    def loss_budget(self) -> pd.DataFrame:
        """Return Driven frequency-sample aggregate loss-budget rows."""
        return self.loss_budget_result.to_dataframe()

    @property
    def epr_losses(self) -> tuple[DomainLoss, SurfaceLoss]:
        """Return typed Driven domain and surface EPR loss tables."""
        return (self.domain_epr_loss, self.surface_epr_loss)

    @property
    def loss(self) -> ReportLoss:
        """Return typed Driven domain, surface, and budget loss data."""
        return ReportLoss(
            domain=self.domain_epr_loss,
            surface=self.surface_epr_loss,
            budget=self.loss_budget_result,
        )

    def loss_budget_column_min(self, column: str) -> float | None:
        """Return a finite minimum from the Driven loss budget."""
        loss_budget = self.loss_budget
        if loss_budget.empty or column not in loss_budget.columns:
            return None
        return finite_min(loss_budget[column].to_numpy())

    def typed_results(self) -> tuple[VisualizationProvider, ...]:
        """Return Typed Data owned by this Driven report."""
        typed_data: list[VisualizationProvider] = [
            *super().typed_results(),
            self.sparams,
        ]
        if not self.domain_energy.empty:
            typed_data.append(self.domain_energy_result)
        if not self.surface_q.empty:
            typed_data.append(self.surface_q_result)
        if not self.loss.empty:
            typed_data.append(self.loss)
        return tuple(typed_data)

    def to_sweep_metrics(self) -> dict[str, Any]:
        """Return compact Driven metrics for sweep-summary tables."""
        return {
            **super().to_sweep_metrics(),
            "frequency_point_count": len(self.sparams.freq),
            "port_count": len(self.sparams.port_names),
            "s_parameter_count": len(self.sparams.keys()),
            "domain_energy_rows": len(self.domain_energy),
            "surface_q_rows": len(self.surface_q),
            "loss_budget_rows": len(self.loss_budget),
            "min_q_total": self.loss_budget_column_min("q_total"),
            "domain_material_rows": len(self.domain_materials),
            "dielectric_interface_rows": len(self.dielectric_interfaces),
        }


__all__ = ["DrivenReport"]
