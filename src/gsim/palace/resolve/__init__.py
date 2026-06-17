"""Palace artifact resolution public facade.

Resolve is the adapter boundary after a Run Stage has produced a canonical run
folder or cloud artifact mapping. It audits the available sources and returns
structured objects that can load semantic result reports.

Public review path:

``resolve_palace_result(...) -> PalaceResolvedResult.load_report()
-> PalaceResultBundle.require_report() -> DrivenReport/EigenmodeReport/...``

The implementation is split by responsibility:

* ``resolve.resolution`` audits artifact identity and returns
  ``PalaceResolvedResult`` without parsing problem-specific result tables.
* ``resolve.models`` defines the typed boundary objects shared by notebook
  review code and Resolve internals.
* ``resolve.sources`` discovers run-root inputs, ``metadata/`` sidecars,
  ``logs/`` runtime evidence, raw ``results/palace/`` files, and sweep metadata.
* ``resolve.loaders`` parses primitive Palace artifacts such as S-parameters,
  eigenmode CSVs, terminal matrices, fields, and indexed CSVs.
* ``resolve.derived`` builds semantic derived tables such as material,
  participation, and loss summaries.
* ``resolve.assembly`` dispatches from a resolved artifact set to
  problem-type report instances using loaders and derived tables.

Resolve does not own Typed Data or Problem Type Report semantics; those live in
``gsim.palace.results``. Resolve also does not own plotting primitives; those
live in ``gsim.palace.display``.

This package facade intentionally exposes the high-level review path only.
Primitive loaders and derived table builders remain importable from their
owner subpackages when implementation code needs them, but notebook/review code
should not treat them as the Resolve public API.
"""

from __future__ import annotations

from gsim.palace.resolve.models import (
    PalaceResolvedResult,
    PalaceResultBundle,
    PalaceRunArtifacts,
)
from gsim.palace.resolve.problem_types import (
    canonical_problem_type,
    normalize_problem_type,
)
from gsim.palace.resolve.resolution import resolve_palace_result
from gsim.palace.resolve.sources.run_models import (
    PalaceArtifactStatus,
    PalaceRunSummary,
)
from gsim.palace.resolve.sources.run_summary import load_palace_run_summary
from gsim.palace.resolve.sources.sweep_models import (
    PalaceSweepPointSpec,
    PalaceSweepPointSummary,
    PalaceSweepResourceIndexResult,
    PalaceSweepSummary,
)
from gsim.palace.resolve.sweeps import load_palace_sweep_summary

__all__ = [
    "PalaceArtifactStatus",
    "PalaceResolvedResult",
    "PalaceResultBundle",
    "PalaceRunArtifacts",
    "PalaceRunSummary",
    "PalaceSweepPointSpec",
    "PalaceSweepPointSummary",
    "PalaceSweepResourceIndexResult",
    "PalaceSweepSummary",
    "canonical_problem_type",
    "load_palace_run_summary",
    "load_palace_sweep_summary",
    "normalize_problem_type",
    "resolve_palace_result",
]
