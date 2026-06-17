"""Problem-report assembly inside the Palace Resolve pipeline.

Assembly is the Resolve layer that constructs ``gsim.palace.results`` report
instances from source summaries, primitive loaders, and derived semantic
tables. It is an implementation boundary, not the owner of report semantics.

This package may choose which optional artifact families feed a report, record
source-status rows, and pass resolved run metadata into the report model. It
does not define Typed Data classes, Problem Type Report classes, display
helpers, or notebook-facing public shortcuts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from gsim.palace.resolve.assembly.driven import load_driven_report
from gsim.palace.resolve.assembly.eigenmode import load_eigenmode_report
from gsim.palace.resolve.assembly.electrostatic import load_electrostatic_report
from gsim.palace.resolve.problem_types import (
    canonical_problem_type,
    normalize_problem_type,
)


class ResolvedReportSource(Protocol):
    """Structural input needed to assemble a report bundle.

    Assembly deliberately depends on this structural contract instead of
    importing Resolve model classes. That keeps the dependency direction
    ``models -> assembly`` one-way for ``PalaceResolvedResult.load_report()``.
    """

    source: Any
    problem_type: str | None
    artifacts: Any

    @property
    def run_summary(self) -> Any:
        """Return the resolved Palace run summary."""
        ...


def load_report_for_resolved_result[BundleT](
    resolved: ResolvedReportSource,
    *,
    frequency_ghz: float | None,
    require_report: bool,
    bundle_type: Callable[..., BundleT],
) -> BundleT:
    """Load a problem report for an already resolved Palace artifact set.

    The resolved object owns artifact identity. Assembly owns the
    problem-type dispatch that turns those artifacts into a report bundle.
    ``bundle_type`` is injected by the model method so this module does not
    import Resolve models at runtime and cannot form an import cycle.
    """
    source = resolved.source
    loader_source = dict(source) if isinstance(source, Mapping) else source
    summary = resolved.run_summary
    resolved_problem = resolved.problem_type
    report: object | None = None
    report_status = "skipped"
    report_message: str | None = None

    normalized = normalize_problem_type(resolved_problem)
    if normalized is None and resolved_problem is None:
        report_message = "problem type unavailable"
    elif normalized is None:
        report_message = f"unsupported Palace problem type {resolved_problem!r}"
    else:
        try:
            if normalized == "driven":
                report = load_driven_report(loader_source, run_summary=summary)
            elif normalized == "eigenmode":
                report = load_eigenmode_report(loader_source, run_summary=summary)
            elif normalized == "electrostatic":
                report = load_electrostatic_report(
                    loader_source,
                    frequency_ghz=frequency_ghz,
                    run_summary=summary,
                )
            else:
                report_message = f"unsupported Palace problem type {resolved_problem!r}"
        except FileNotFoundError as exc:
            report_status = "missing"
            report_message = str(exc)
            if require_report:
                raise
        except ValueError as exc:
            report_status = "error"
            report_message = str(exc)
            if require_report:
                raise
        else:
            if report is not None:
                report_status = "loaded"

    if require_report and report is None:
        raise FileNotFoundError(report_message or "Palace report could not be loaded")

    return bundle_type(
        source=source,
        problem_type=canonical_problem_type(resolved_problem),
        artifacts=resolved.artifacts,
        report=report,
        report_status=report_status,
        report_message=report_message,
    )


__all__ = [
    "load_driven_report",
    "load_eigenmode_report",
    "load_electrostatic_report",
    "load_report_for_resolved_result",
]
