"""Resolve Palace artifact identity into typed boundary objects.

This module is the first Resolve entrypoint after ``run()``, ``run_local()``,
or handoff package generation. It audits source artifacts and returns a
``PalaceResolvedResult``. Loading problem reports from that resolved object is
owned by ``resolve.assembly`` through ``PalaceResolvedResult.load_report()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, overload

from gsim.palace.resolve.models import (
    PalaceResolvedResult,
    PalaceRunArtifacts,
)
from gsim.palace.resolve.problem_types import (
    canonical_problem_type,
)
from gsim.palace.resolve.sources.run_summary import load_palace_run_summary

if TYPE_CHECKING:
    from gsim.palace.resolve.problem_types import (
        DrivenProblemType,
        EigenmodeProblemType,
        ElectrostaticProblemType,
    )
    from gsim.palace.results.reports.base import BasePalaceReport
    from gsim.palace.results.reports.driven import DrivenReport
    from gsim.palace.results.reports.eigenmode import EigenmodeReport
    from gsim.palace.results.reports.electrostatic import ElectrostaticReport


@overload
def resolve_palace_result(
    source: str | Path | Mapping[str, str | Path],
    *,
    problem_type: DrivenProblemType,
    include_hashes: bool = False,
) -> PalaceResolvedResult[DrivenReport]: ...


@overload
def resolve_palace_result(
    source: str | Path | Mapping[str, str | Path],
    *,
    problem_type: EigenmodeProblemType,
    include_hashes: bool = False,
) -> PalaceResolvedResult[EigenmodeReport]: ...


@overload
def resolve_palace_result(
    source: str | Path | Mapping[str, str | Path],
    *,
    problem_type: ElectrostaticProblemType,
    include_hashes: bool = False,
) -> PalaceResolvedResult[ElectrostaticReport]: ...


@overload
def resolve_palace_result(
    source: str | Path | Mapping[str, str | Path],
    *,
    problem_type: None = None,
    include_hashes: bool = False,
) -> PalaceResolvedResult[BasePalaceReport]: ...


@overload
def resolve_palace_result(
    source: str | Path | Mapping[str, str | Path],
    *,
    problem_type: str | None = None,
    include_hashes: bool = False,
) -> PalaceResolvedResult[BasePalaceReport]: ...


def resolve_palace_result(
    source: str | Path | Mapping[str, str | Path],
    *,
    problem_type: str | None = None,
    include_hashes: bool = False,
) -> PalaceResolvedResult[BasePalaceReport]:
    """Resolve Palace artifact identity without loading physics reports.

    This is the explicit boundary between Run Stage artifacts and typed result
    parsing. It inspects generated artifacts, determines the Palace problem
    type, and reports core file presence. It does not parse S-parameter,
    eigenmode, terminal-matrix, EPR, or loss tables; callers opt into that work
    with ``resolved.load_report(...)`` after reviewing the source audit.

    Args:
        source: Simulation directory, Palace output directory, or mapping of
            already-located result artifact names to paths.
        problem_type: Optional explicit Palace problem type. When omitted, the
            value is read from ``config.json`` when that artifact is present.
        include_hashes: Include SHA-256 checksums in artifact status rows.

    Returns:
        ``PalaceResolvedResult`` with artifact status and problem identity.

    Raises:
        FileNotFoundError: If the source path itself cannot be resolved by the
            run-summary loader.
    """
    summary_source: str | Path | dict[str, str | Path] = (
        dict(source) if isinstance(source, Mapping) else source
    )
    summary = load_palace_run_summary(summary_source, include_hashes=include_hashes)
    resolved_problem = problem_type or summary.problem_type
    artifacts = PalaceRunArtifacts(source=source, summary=summary)
    return PalaceResolvedResult(
        source=source,
        problem_type=canonical_problem_type(resolved_problem),
        artifacts=artifacts,
    )
