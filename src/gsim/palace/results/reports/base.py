"""Base report contract shared by Palace Problem Type Reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from gsim.palace.display import (
    DisplayValue,
    VisualizationProvider,
    collect_visualizations,
    display_items,
)
from gsim.palace.results.benchmarks import SimulationBenchmark


@dataclass(frozen=True, kw_only=True)
class BasePalaceReport:
    """Common report contract shared by every Palace problem type.

    The base report carries source-status and lookup tables for audit while
    exposing simulation benchmark metadata through an explicit helper.

    Problem-specific report classes add Typed Data objects such as S-parameters,
    Eigenmodes, or TerminalMatrix. The base class does not parse files, render
    notebook views, or define problem physics.
    """

    domain_materials: pd.DataFrame
    dielectric_interfaces: pd.DataFrame
    index_map: pd.DataFrame
    sources: pd.DataFrame
    run_summary: Any | None = None

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
    def benchmark(self) -> SimulationBenchmark | None:
        """Return typed simulation benchmark data when run metadata exists."""
        if self.run_summary is None:
            return None
        return SimulationBenchmark.from_run_summary(self.run_summary)

    def typed_results(self) -> tuple[VisualizationProvider, ...]:
        """Return common Typed Data owned by every problem report.

        Source status, missing artifacts, and run benchmark metadata remain
        available through explicit accessors, but they are not semantic physics
        results and are not displayed by ``show_all_results()``.
        """
        return ()

    def benchmark_items(self) -> dict[str, DisplayValue]:
        """Return display items for explicit run-cost benchmark inspection."""
        benchmark = self.benchmark
        if benchmark is None:
            return {}
        return dict(benchmark.visualize())

    def show_simulation_benchmark(
        self,
        *,
        figure_layout: Mapping[str, Any] | None = None,
        plotly_config: Mapping[str, Any] | None = None,
    ) -> dict[str, DisplayValue]:
        """Display and return benchmark items for this run.

        Benchmark data answers run-cost questions such as wall time, core hours,
        model size, and peak memory. It is intentionally separate from
        ``show_all_results()`` so the default report view stays focused on the
        problem's physical result data.
        """
        items = self.benchmark_items()
        display_items(
            items,
            figure_layout=figure_layout,
            plotly_config=plotly_config,
        )
        return items

    def result_items(self) -> dict[str, DisplayValue]:
        """Return named visualizer outputs from Typed Data owned by this report."""
        return collect_visualizations(self.typed_results())

    def show_all_results(
        self,
        *,
        figure_layout: Mapping[str, Any] | None = None,
        plotly_config: Mapping[str, Any] | None = None,
    ) -> dict[str, DisplayValue]:
        """Display and return all result items owned by this report.

        Problem reports are aggregation points. They do not construct new plot
        semantics; they call visualizers on the Typed Data objects they own and
        pass the resulting named items to the generic display helper.

        Figure builders already decide plot semantics: traces, titles, axes,
        and units. ``figure_layout`` is a cosmetic Plotly layout override
        applied before display. ``plotly_config`` is render-time configuration
        passed to Plotly when showing the figure.
        """
        items = self.result_items()
        display_items(
            items,
            figure_layout=figure_layout,
            plotly_config=plotly_config,
        )
        return items

    def to_sweep_metrics(self) -> dict[str, Any]:
        """Return compact metrics shared by sweep-summary tables."""
        return {
            "status": "loaded",
            "problem_type": self.problem_type,
            "index_map_rows": len(self.index_map),
            "missing_reports": ",".join(self.missing_reports),
        }
