"""Simulation performance and benchmark typed data for Palace runs.

This module owns semantic typed data that users inspect as result tables or
figures: runtime/resource performance and flat benchmark metrics. Artifact
presence, missing-file checks, handoff sidecars, and sweep point metadata are
resolve-layer metadata and live under ``gsim.palace.resolve``.

The classes here do not discover files, parse JSON/CSV sidecars, compose
problem reports, or decide which notebooks display them. Resolve supplies
already-loaded run metadata; reports aggregate these typed-data visualizers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from gsim.palace._shared import as_mapping
from gsim.palace.display import DisplayValue, PlotlyFigure

if TYPE_CHECKING:
    import pandas as pd


class RunSummaryLike(Protocol):
    """Structural run-summary data consumed by performance typed results."""

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


@dataclass(frozen=True)
class SimulationPerformance:
    """Typed performance data for one Palace run.

    The object turns raw runtime/resource dictionaries from a resolved
    run-summary object into table and figure surfaces. It intentionally
    excludes source-status and missing-artifact checks; those describe folder
    structure rather than simulation performance.
    """

    problem_type: str | None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    resource: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_summary(cls, summary: RunSummaryLike) -> SimulationPerformance:
        """Build typed performance data from resolved raw run metadata."""
        return cls(
            problem_type=summary.problem_type,
            runtime=dict(summary.runtime),
            resource=dict(summary.resource),
        )

    def summary_table(self) -> pd.DataFrame:
        """Return a compact table of runtime/resource performance fields."""
        import pandas as pd

        resource_runtime = as_mapping(self.resource.get("runtime"))
        allocation = as_mapping(self.resource.get("allocation"))
        model_size = as_mapping(self.resource.get("model_size"))
        memory = as_mapping(self.resource.get("memory"))
        rows = [
            ("problem_type", self.problem_type),
            ("runtime_present", self.runtime.get("present") is True),
            ("runtime_status", self.runtime.get("status")),
            ("runtime_elapsed_seconds", self.runtime.get("elapsed_seconds")),
            ("resource_present", self.resource.get("present") is True),
            ("resource_status", self.resource.get("status")),
            ("resource_wall_time_seconds", resource_runtime.get("wall_time_seconds")),
            ("resource_core_hours", resource_runtime.get("core_hours")),
            ("resource_nodes", allocation.get("nodes")),
            ("resource_num_processes", allocation.get("num_processes")),
            ("resource_num_threads", allocation.get("num_threads")),
            ("resource_global_unknowns", model_size.get("global_unknowns")),
            ("resource_peak_total_hwm_gib", memory.get("peak_total_hwm_gib")),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return notebook-ready performance tables."""
        return {"simulation_performance_table": self.summary_table()}

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return general performance figures when numeric metrics exist."""
        numeric_rows = []
        for metric, value in self.summary_table().itertuples(index=False):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if metric.endswith("_present"):
                continue
            numeric_rows.append({"x": [metric], "y": [numeric], "name": metric})
        if not numeric_rows:
            return {}

        from gsim.palace.display import make_bar_figure

        return {
            "simulation_performance_metrics_bar_plot": make_bar_figure(
                numeric_rows,
                title="Simulation performance metrics",
                x_title="Metric",
                y_title="Value",
            )
        }

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default performance tables and figures."""
        return {**self.tables(), **self.figures()}


@dataclass(frozen=True)
class SimulationBenchmark:
    """Typed benchmark data for one Palace run.

    Benchmark data is a flat subset of simulation performance intended for
    comparison across runs or sweep points. It is a report result because users
    inspect it through tables and plots; raw artifact presence remains in
    resolve metadata.
    """

    problem_type: str | None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_summary(cls, summary: RunSummaryLike) -> SimulationBenchmark:
        """Build flat benchmark data from resolved raw run metadata."""
        runtime = summary.runtime
        resource = summary.resource
        resource_runtime = as_mapping(resource.get("runtime"))
        model_size = as_mapping(resource.get("model_size"))
        memory = as_mapping(resource.get("memory"))
        return cls(
            problem_type=summary.problem_type,
            metrics={
                "runtime_present": runtime.get("present") is True,
                "runtime_status": runtime.get("status"),
                "runtime_elapsed_seconds": runtime.get("elapsed_seconds"),
                "resource_present": resource.get("present") is True,
                "resource_status": resource.get("status"),
                "resource_wall_time_seconds": resource_runtime.get("wall_time_seconds"),
                "resource_core_hours": resource_runtime.get("core_hours"),
                "resource_global_unknowns": model_size.get("global_unknowns"),
                "resource_peak_total_hwm_gib": memory.get("peak_total_hwm_gib"),
            },
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return one table row of benchmark metrics."""
        import pandas as pd

        return pd.DataFrame.from_records(
            [{"problem_type": self.problem_type, **dict(self.metrics)}]
        )

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return notebook-ready benchmark tables."""
        return {"simulation_benchmark_table": self.to_dataframe()}

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return general benchmark figures when numeric metrics exist."""
        bars = []
        for metric, value in self.metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if metric.endswith("_present"):
                continue
            bars.append({"x": [metric], "y": [numeric], "name": metric})
        if not bars:
            return {}

        from gsim.palace.display import make_bar_figure

        return {
            "simulation_benchmark_metrics_bar_plot": make_bar_figure(
                bars,
                title="Simulation benchmark metrics",
                x_title="Metric",
                y_title="Value",
            )
        }

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default benchmark tables and figures."""
        return {**self.tables(), **self.figures()}


__all__ = [
    "SimulationBenchmark",
    "SimulationPerformance",
]
