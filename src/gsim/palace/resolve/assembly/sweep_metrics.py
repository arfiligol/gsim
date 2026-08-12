"""Extract compact report-derived metrics for Palace sweep points.

Sweep report metrics are assembly behavior because they instantiate
problem-type reports and ask those reports for compact semantic metrics. Source
discovery stays in ``gsim.palace.resolve.sources``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsim.palace.resolve.assembly.driven import load_driven_report
from gsim.palace.resolve.assembly.eigenmode import load_eigenmode_report
from gsim.palace.resolve.assembly.electrostatic import load_electrostatic_report
from gsim.palace.resolve.problem_types import normalize_problem_type


def load_sweep_point_report_metrics(
    source: str | Path | dict,
    problem_type: str | None,
) -> dict[str, Any]:
    """Load compact problem-report metrics for one resolved sweep point."""
    normalized = normalize_problem_type(problem_type)
    if problem_type is None:
        return {
            "status": "skipped",
            "problem_type": None,
            "message": "problem type unavailable",
        }
    if not normalized:
        return {
            "status": "skipped",
            "problem_type": problem_type,
            "message": f"unsupported Palace problem type {problem_type!r}",
        }

    try:
        if normalized == "driven":
            return load_driven_report(source).to_sweep_metrics()
        if normalized == "eigenmode":
            return load_eigenmode_report(source).to_sweep_metrics()
        if normalized == "electrostatic":
            return load_electrostatic_report(source).to_sweep_metrics()
    except FileNotFoundError as exc:
        return {
            "status": "missing",
            "problem_type": problem_type,
            "message": str(exc),
        }
    except ValueError as exc:
        return {
            "status": "error",
            "problem_type": problem_type,
            "message": str(exc),
        }

    return {
        "status": "skipped",
        "problem_type": problem_type,
        "message": f"unsupported Palace problem type {problem_type!r}",
    }
