"""Static typing contract for Palace resolve report review paths.

Run with:

    uv run pyright scripts/typechecks/palace_resolve_typing_contract.py
    uv run mypy --strict scripts/typechecks/palace_resolve_typing_contract.py

The file is intentionally not a runtime test. It records the reviewer-facing
typing contract: a literal Palace problem type should lead to the matching
concrete report type after ``bundle.require_report()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_type

from gsim.palace import (
    DrivenReport,
    EigenmodeReport,
    ElectrostaticReport,
    PalaceResolvedResult,
    PalaceResultBundle,
    resolve_palace_result,
)
from gsim.palace.results.reports.base import BasePalaceReport

source = Path("build/my_run")

driven_resolved = resolve_palace_result(source, problem_type="Driven")
assert_type(driven_resolved, PalaceResolvedResult[DrivenReport])
driven_bundle = driven_resolved.load_report(require_report=True)
assert_type(driven_bundle, PalaceResultBundle[DrivenReport])
driven_report = driven_bundle.require_report()
assert_type(driven_report, DrivenReport)
driven_report.sparams.visualize()

eigenmode_resolved = resolve_palace_result(source, problem_type="eigenmode")
assert_type(eigenmode_resolved, PalaceResolvedResult[EigenmodeReport])
eigenmode_report = eigenmode_resolved.load_report(require_report=True).require_report()
assert_type(eigenmode_report, EigenmodeReport)
eigenmode_report.eigenmodes.visualize()

electrostatic_resolved = resolve_palace_result(source, problem_type="Electrostatic")
assert_type(electrostatic_resolved, PalaceResolvedResult[ElectrostaticReport])
electrostatic_bundle = electrostatic_resolved.load_report(require_report=True)
assert_type(electrostatic_bundle, PalaceResultBundle[ElectrostaticReport])
electrostatic_report = electrostatic_bundle.require_report()
assert_type(electrostatic_report, ElectrostaticReport)
electrostatic_report.capacitance.visualize()


def check_dynamic_problem_type(problem_type: str | None) -> None:
    """Dynamic problem labels keep only the common report contract."""
    dynamic_resolved = resolve_palace_result(source, problem_type=problem_type)
    assert_type(dynamic_resolved, PalaceResolvedResult[BasePalaceReport])
    dynamic_report = dynamic_resolved.load_report().require_report()
    assert_type(dynamic_report, BasePalaceReport)


inferred_resolved = resolve_palace_result(source)
assert_type(inferred_resolved, PalaceResolvedResult[BasePalaceReport])
inferred_report = inferred_resolved.load_report().require_report()
assert_type(inferred_report, BasePalaceReport)
