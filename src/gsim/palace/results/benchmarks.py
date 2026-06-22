"""Simulation benchmark typed data for Palace runs.

This module owns semantic typed data that users inspect as result tables or
figures: runtime/resource fields and flat benchmark metrics. Artifact presence,
missing-file checks, handoff sidecars, and sweep point metadata are resolve-layer
metadata and live under ``gsim.palace.resolve``.

The classes here do not discover files, parse JSON/CSV sidecars, compose
problem reports, or decide which notebooks display them. Resolve supplies
already-loaded run metadata; reports aggregate these typed-data visualizers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gsim.palace._shared import as_mapping, optional_float, optional_int
from gsim.palace.display import DisplayValue, PlotlyFigure, make_trace_subplot_figure

if TYPE_CHECKING:
    import pandas as pd

_GIB = 1024.0**3
_ADAPTIVE_PASS_COLUMNS = (
    "adaptive_pass",
    "cumulative_wall_time_seconds",
    "cumulative_core_hours",
    "peak_total_hwm_gib",
    "global_unknowns",
)


@dataclass(frozen=True)
class SimulationBenchmark:
    """Typed benchmark data for one Palace run.

    Benchmark data describes run cost and solver-resource scale: elapsed time,
    allocated compute, model size, and peak memory. Reports keep it out of
    ``show_all_results()`` so physics results stay first; notebooks display it
    explicitly with ``show_simulation_benchmark()`` when run-cost inspection is
    relevant.
    """

    problem_type: str | None
    runtime: Mapping[str, Any] = field(default_factory=dict)
    resource: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_summary(cls, summary: Any) -> SimulationBenchmark:
        """Build benchmark data from resolved raw run metadata."""
        return cls(
            problem_type=summary.problem_type,
            runtime=dict(summary.runtime),
            resource=dict(summary.resource),
        )

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return flat benchmark metrics for sweep/run comparison rows."""
        runtime = self.runtime
        resource = self.resource
        resource_runtime = as_mapping(resource.get("runtime"))
        model_size = as_mapping(resource.get("model_size"))
        memory = as_mapping(resource.get("memory"))
        return {
            "runtime_present": runtime.get("present") is True,
            "runtime_status": runtime.get("status"),
            "runtime_elapsed_seconds": runtime.get("elapsed_seconds"),
            "resource_present": resource.get("present") is True,
            "resource_status": resource.get("status"),
            "resource_wall_time_seconds": resource_runtime.get("wall_time_seconds"),
            "resource_core_hours": resource_runtime.get("core_hours"),
            "resource_cores": self.resource_cores,
            "resource_global_unknowns": model_size.get("global_unknowns"),
            "resource_peak_total_hwm_gib": memory.get("peak_total_hwm_gib"),
        }

    @property
    def resource_cores(self) -> float | None:
        """Return allocated CPU cores used to convert wall time into core-hours."""
        allocation = as_mapping(self.resource.get("allocation"))
        cores = optional_float(allocation.get("cores"))
        if cores is not None:
            return cores
        cores = optional_float(allocation.get("num_cpus"))
        if cores is not None:
            return cores
        processes = optional_float(allocation.get("num_processes"))
        threads = optional_float(allocation.get("num_threads"))
        if processes is None or threads is None:
            return None
        return processes * threads

    @property
    def has_summary_metrics(self) -> bool:
        """Return whether the flat summary has at least one numeric benchmark."""
        ignored = {
            "runtime_present",
            "runtime_status",
            "resource_present",
            "resource_status",
        }
        return any(
            optional_float(value) is not None
            for metric, value in self.metrics.items()
            if metric not in ignored
        )

    def summary_table(self) -> pd.DataFrame:
        """Return the compact run-cost summary for human notebook scanning.

        The rows answer the first engineering questions after a solve finishes:
        whether runtime/resource sidecars were available, elapsed wall time,
        HPC billing scale in core-hours, allocated cores, model size in global
        unknowns, and peak high-water memory in GiB.
        """
        import pandas as pd

        allocation = as_mapping(self.resource.get("allocation"))
        rows = [
            ("problem_type", self.problem_type),
            *self.metrics.items(),
            ("resource_nodes", allocation.get("nodes")),
            ("resource_num_processes", allocation.get("num_processes")),
            ("resource_num_threads", allocation.get("num_threads")),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def to_dataframe(self) -> pd.DataFrame:
        """Return one flat benchmark row for sweep-level comparison."""
        import pandas as pd

        return pd.DataFrame.from_records(
            [{"problem_type": self.problem_type, **dict(self.metrics)}]
        )

    def adaptive_pass_dataframe(self) -> pd.DataFrame:
        """Return adaptive-pass resource diagnostics from resolved sidecars.

        Each row is the resource state by the end of one Palace adaptive pass.
        Palace's timing table is cumulative, so ``cumulative_wall_time_seconds``
        uses the pass ``Total`` row directly. ``cumulative_core_hours`` applies
        the run allocation for HPC cost review, ``peak_total_hwm_gib`` reports
        the memory high-water mark needed to size jobs, and ``global_unknowns``
        records the FEM model size for cross-solver/resource comparison.
        """
        import pandas as pd

        rows: dict[int, dict[str, Any]] = {}

        for row in self._table_records("stage_timing"):
            if not _is_total_stage(row):
                continue
            pass_index = optional_int(row.get("pass_index"))
            if pass_index is None:
                continue
            cumulative_wall_time = optional_float(row.get("max_seconds"))
            if cumulative_wall_time is None:
                continue
            rows.setdefault(pass_index, {"adaptive_pass": pass_index})[
                "cumulative_wall_time_seconds"
            ] = cumulative_wall_time

        cores = self.resource_cores
        if cores is not None:
            for row in rows.values():
                wall_time = optional_float(row.get("cumulative_wall_time_seconds"))
                if wall_time is not None:
                    row["cumulative_core_hours"] = wall_time * cores / 3600.0

        for row in self._table_records("stage_memory"):
            if not _is_total_stage(row):
                continue
            pass_index = optional_int(row.get("pass_index"))
            peak_bytes = optional_float(row.get("total_hwm_bytes"))
            if pass_index is None or peak_bytes is None:
                continue
            rows.setdefault(pass_index, {"adaptive_pass": pass_index})[
                "peak_total_hwm_gib"
            ] = peak_bytes / _GIB

        for row in self._table_records("amr_passes"):
            pass_index = optional_int(row.get("pass_index"))
            if pass_index is None:
                continue
            unknowns = optional_int(row.get("global_unknowns"))
            if unknowns is not None:
                rows.setdefault(pass_index, {"adaptive_pass": pass_index})[
                    "global_unknowns"
                ] = unknowns

        self._fill_summary_metrics(rows)
        return pd.DataFrame.from_records(
            [rows[key] for key in sorted(rows)],
            columns=_ADAPTIVE_PASS_COLUMNS,
        )

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return notebook-ready benchmark tables with engineering meaning."""
        tables: dict[str, pd.DataFrame] = {}
        if self.has_summary_metrics:
            tables["simulation_benchmark_summary_table"] = self.summary_table()
        adaptive_passes = self.adaptive_pass_dataframe()
        if not adaptive_passes.empty:
            tables["simulation_benchmark_adaptive_pass_table"] = adaptive_passes
        return tables

    def figures(self) -> dict[str, PlotlyFigure]:
        """Return adaptive-pass resource traces when sidecar data exists."""
        adaptive_passes = self.adaptive_pass_dataframe()
        panels = [
            panel
            for panel in (
                _trace_panel(
                    adaptive_passes,
                    column="cumulative_wall_time_seconds",
                    title="Cumulative wall time",
                    y_title="Wall time (minutes)",
                    y_scale=1.0 / 60.0,
                ),
                _trace_panel(
                    adaptive_passes,
                    column="cumulative_core_hours",
                    title="Cumulative core-hours",
                    y_title="Core-hours (allocated cores x elapsed hours)",
                ),
                _trace_panel(
                    adaptive_passes,
                    column="global_unknowns",
                    title="FEM model size",
                    y_title="unknowns",
                ),
                _trace_panel(
                    adaptive_passes,
                    column="peak_total_hwm_gib",
                    title="Peak memory",
                    y_title="GiB",
                ),
            )
            if panel is not None
        ]
        if not panels:
            return {}

        return {
            "simulation_benchmark_adaptive_pass_trace_plot": (
                make_trace_subplot_figure(
                    panels,
                    title="Simulation benchmark by adaptive pass",
                    vertical_spacing=0.10,
                )
            )
        }

    def visualize(self) -> dict[str, DisplayValue]:
        """Return all default benchmark tables and figures."""
        return {**self.tables(), **self.figures()}

    def _table_path(self, name: str) -> Path | None:
        tables = as_mapping(self.resource.get("tables"))
        table = as_mapping(tables.get(name))
        path_value = table.get("path")
        if path_value is None:
            return None
        path = Path(str(path_value))
        if path.is_absolute():
            return path

        record_path_value = self.resource.get("path")
        if record_path_value is None:
            return None
        record_path = Path(str(record_path_value))
        run_root = (
            record_path.parent.parent
            if record_path.parent.name == "metadata"
            else record_path.parent
        )
        return run_root / path

    def _table_records(self, name: str) -> tuple[Mapping[str, Any], ...]:
        import pandas as pd

        path = self._table_path(name)
        if path is None or not path.is_file():
            return ()
        frame = pd.read_csv(path)
        return tuple(cast("Mapping[str, Any]", row) for row in frame.to_dict("records"))

    def _fill_summary_metrics(self, rows: dict[int, dict[str, Any]]) -> None:
        model_size = as_mapping(self.resource.get("model_size"))
        runtime = as_mapping(self.resource.get("runtime"))
        memory = as_mapping(self.resource.get("memory"))
        summary_pass = (
            max(rows)
            if rows
            else optional_int(model_size.get("completed_amr_iterations"))
        )
        if summary_pass is None:
            return

        row = rows.setdefault(summary_pass, {"adaptive_pass": summary_pass})
        row.setdefault(
            "cumulative_wall_time_seconds",
            runtime.get("wall_time_seconds"),
        )
        row.setdefault("cumulative_core_hours", runtime.get("core_hours"))
        row.setdefault("peak_total_hwm_gib", memory.get("peak_total_hwm_gib"))
        row.setdefault("global_unknowns", model_size.get("global_unknowns"))


def _is_total_stage(row: Mapping[str, Any]) -> bool:
    return str(row.get("stage", "")).casefold() == "total"


def _trace_panel(
    frame: pd.DataFrame,
    *,
    column: str,
    title: str,
    y_title: str,
    y_scale: float = 1.0,
) -> Mapping[str, Any] | None:
    if frame.empty or column not in frame:
        return None
    series = frame[["adaptive_pass", column]].dropna()
    if series.empty:
        return None
    return {
        "title": title,
        "x_title": "Adaptive pass",
        "y_title": y_title,
        "traces": (
            {
                "x": series["adaptive_pass"],
                "y": series[column] * y_scale,
                "name": title,
                "mode": "markers+lines",
            },
        ),
    }


__all__ = [
    "SimulationBenchmark",
]
