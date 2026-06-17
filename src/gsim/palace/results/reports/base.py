"""Base report contract shared by Palace Problem Type Reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from gsim.palace.display import DisplayValue, VisualizationProvider
from gsim.palace.results.benchmarks import SimulationBenchmark, SimulationPerformance


class ReportRunSummaryLike(Protocol):
    """Structural run metadata carried by reports as provenance."""

    @property
    def problem_type(self) -> str | None:
        """Palace problem type recorded by resolve."""
        ...

    @property
    def runtime(self) -> Mapping[str, Any]:
        """Runtime sidecar summary recorded by resolve."""
        ...

    @property
    def resource(self) -> Mapping[str, Any]:
        """Resource sidecar summary recorded by resolve."""
        ...

    @property
    def missing_artifacts(self) -> tuple[str, ...]:
        """Core handoff artifacts missing from the resolved run."""
        ...


@dataclass(frozen=True, kw_only=True)
class BasePalaceReport:
    """Common report contract shared by every Palace problem type.

    The base report carries source-status and lookup tables for audit while
    exposing simulation performance and benchmark Typed Data as shared report
    results.

    Problem-specific report classes add Typed Data objects such as S-parameters,
    Eigenmodes, or TerminalMatrix. The base class does not parse files, render
    notebook views, or define problem physics.
    """

    domain_materials: pd.DataFrame
    dielectric_interfaces: pd.DataFrame
    index_map: pd.DataFrame
    sources: pd.DataFrame
    run_summary: ReportRunSummaryLike | None = None

    @property
    def problem_type(self) -> str:
        """Return the Palace problem type represented by this report."""
        return type(self).__name__.removesuffix("Report")

    @property
    def missing_reports(self) -> tuple[str, ...]:
        """Optional report names that were expected but absent."""
        if self.sources.empty:
            return ()
        missing = self.sources.loc[
            (~self.sources["required"]) & (~self.sources["present"]),
            "name",
        ]
        return tuple(str(name) for name in missing)

    @property
    def missing_artifacts(self) -> tuple[str, ...]:
        """Core handoff artifacts missing from the associated run summary."""
        if self.run_summary is None:
            return ()
        return self.run_summary.missing_artifacts

    @property
    def performance(self) -> SimulationPerformance | None:
        """Return typed simulation performance data when run metadata exists."""
        if self.run_summary is None:
            return None
        return SimulationPerformance.from_run_summary(self.run_summary)

    @property
    def benchmark(self) -> SimulationBenchmark | None:
        """Return typed simulation benchmark data when run metadata exists."""
        if self.run_summary is None:
            return None
        return SimulationBenchmark.from_run_summary(self.run_summary)

    def typed_results(self) -> tuple[VisualizationProvider, ...]:
        """Return common Typed Data owned by every problem report.

        Source status and missing-artifact metadata remain available as
        attributes, but they are not semantic result data and are not displayed
        by ``show_all_results()``.
        """
        typed_data: list[VisualizationProvider] = []
        if self.performance is not None:
            typed_data.append(self.performance)
        if self.benchmark is not None:
            typed_data.append(self.benchmark)
        return tuple(typed_data)

    def result_items(self) -> dict[str, DisplayValue]:
        """Return named visualizer outputs from Typed Data owned by this report."""
        from gsim.palace.display import collect_visualizations

        return collect_visualizations(self.typed_results())

    def show_all_results(self) -> dict[str, DisplayValue]:
        """Display and return all result items owned by this report.

        Problem reports are aggregation points. They do not construct new plot
        semantics; they call visualizers on the Typed Data objects they own and
        pass the resulting named items to the generic display helper.
        """
        from gsim.palace.display import display_items

        items = self.result_items()
        display_items(items)
        return items

    def to_sweep_metrics(self) -> dict[str, Any]:
        """Return compact metrics shared by sweep-summary tables."""
        return {
            "status": "loaded",
            "problem_type": self.problem_type,
            "index_map_rows": len(self.index_map),
            "missing_reports": ",".join(self.missing_reports),
        }
