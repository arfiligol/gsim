"""Resolve-layer bundle models for Palace artifacts and reports.

These models describe the Resolve-stage boundary after a run folder or cloud
result mapping is available. They record where artifacts live and attach the
problem report produced by the resolve pipeline. They do not define typed
physics data or parse files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gsim.palace.resolve.sources.run_models import PalaceRunSummary
    from gsim.palace.results.reports.base import BasePalaceReport


@dataclass(frozen=True)
class PalaceRunArtifacts:
    """Typed access to Palace handoff, runtime, and result artifact status.

    The object is a small wrapper around :class:`PalaceRunSummary`. It exists so
    notebook and display code can pass a typed artifact surface instead of a raw
    ``dict[str, Path]`` from ``run()`` or ``run_local()``.
    """

    source: str | Path | Mapping[str, str | Path]
    summary: PalaceRunSummary

    @property
    def missing_artifacts(self) -> tuple[str, ...]:
        """Return core handoff artifacts that were not found."""
        return self.summary.missing_artifacts

    @property
    def result_names(self) -> tuple[str, ...]:
        """Return present Palace result file names."""
        return tuple(
            name for name, artifact in self.summary.results.items() if artifact.present
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly artifact bundle."""
        source: str | dict[str, str]
        if isinstance(self.source, Mapping):
            source = {str(name): str(path) for name, path in self.source.items()}
        else:
            source = str(self.source)
        return {
            "source": source,
            "summary": self.summary.to_dict(),
            "missing_artifacts": list(self.missing_artifacts),
            "result_names": list(self.result_names),
        }


@dataclass(frozen=True)
class PalaceResolvedResult[ReportT: BasePalaceReport]:
    """Resolved Palace artifact identity before report composition.

    This object is the boundary between simulation execution or handoff
    generation and typed report loading. It records where Palace artifacts
    live, which problem type they declare, and which core files are present.
    It does not parse problem-specific CSV content into physics objects until
    callers explicitly request :meth:`load_report`.
    """

    source: str | Path | Mapping[str, str | Path]
    problem_type: str | None
    artifacts: PalaceRunArtifacts

    @property
    def run_summary(self) -> PalaceRunSummary:
        """Return the compact Palace run artifact summary."""
        return self.artifacts.summary

    def load_report(
        self,
        *,
        frequency_ghz: float | None = None,
        require_report: bool = False,
    ) -> PalaceResultBundle[ReportT]:
        """Load the problem-specific typed report for this resolved result.

        Args:
            frequency_ghz: Optional external frequency used by reports that
                cannot infer loss-rate frequency from solver rows.
            require_report: Raise when the problem-specific report cannot be
                loaded.

        Returns:
            :class:`PalaceResultBundle` carrying this resolved artifact state
            and the typed report when available. The report type follows the
            generic parameter resolved from the literal ``problem_type`` used
            with ``resolve_palace_result()``.
        """
        from gsim.palace.resolve.assembly import load_report_for_resolved_result

        return cast(
            "PalaceResultBundle[ReportT]",
            load_report_for_resolved_result(
                self,
                frequency_ghz=frequency_ghz,
                require_report=require_report,
                bundle_type=PalaceResultBundle,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly resolved-result summary."""
        return {
            "problem_type": self.problem_type,
            "artifacts": self.artifacts.to_dict(),
            "missing_artifacts": list(self.artifacts.missing_artifacts),
            "result_names": list(self.artifacts.result_names),
        }


@dataclass(frozen=True)
class PalaceResultBundle[ReportT: BasePalaceReport]:
    """Typed Palace result bundle for notebook display and handoff review.

    The bundle carries artifact status, runtime/resource sidecars, and the
    problem-specific report when it can be loaded from the same source. The
    report remains optional for non-strict artifact audits; ``require_report``
    gives reviewer code a non-optional concrete report.
    """

    source: str | Path | Mapping[str, str | Path]
    problem_type: str | None
    artifacts: PalaceRunArtifacts
    report: ReportT | None
    report_status: str
    report_message: str | None = None

    @property
    def run_summary(self) -> PalaceRunSummary:
        """Return the compact Palace run summary."""
        return self.artifacts.summary

    @property
    def missing_reports(self) -> tuple[str, ...]:
        """Return optional report names missing from the loaded report."""
        if self.report is None:
            return ()
        return self.report.missing_reports

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return display-ready warnings about missing report or artifacts."""
        messages: list[str] = []
        if self.report_status != "loaded":
            messages.append(
                self.report_message or f"report status: {self.report_status}"
            )
        if self.artifacts.missing_artifacts:
            messages.append(
                "missing core artifacts: " + ", ".join(self.artifacts.missing_artifacts)
            )
        if self.missing_reports:
            messages.append(
                "missing optional reports: " + ", ".join(self.missing_reports)
            )
        return tuple(messages)

    def require_report(self) -> ReportT:
        """Return the loaded concrete report or raise with resolve context.

        This is the reviewer-facing accessor for typed notebook workflows. The
        ``report`` attribute remains optional for non-strict audit paths, while
        this method provides a non-optional concrete report type when callers
        resolved with a literal Palace problem type.
        """
        if self.report is not None:
            return self.report
        raise FileNotFoundError(
            self.report_message or "Palace report could not be loaded"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly bundle summary without serializing dataframes."""
        return {
            "problem_type": self.problem_type,
            "artifacts": self.artifacts.to_dict(),
            "report_status": self.report_status,
            "report_message": self.report_message,
            "missing_reports": list(self.missing_reports),
            "warnings": list(self.warnings),
        }
