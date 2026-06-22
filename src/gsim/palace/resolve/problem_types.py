"""Problem-type normalization for Palace artifact resolution.

Resolve code uses this module when it needs to route a resolved artifact set to
the matching problem report. The helpers only normalize Palace problem labels;
they do not load reports, inspect files, or define simulation classes.
"""

from __future__ import annotations

from typing import Literal

type DrivenProblemType = Literal["Driven", "driven"]
type EigenmodeProblemType = Literal["Eigenmode", "eigenmode"]
type ElectrostaticProblemType = Literal["Electrostatic", "electrostatic"]
type MagnetostaticProblemType = Literal["Magnetostatic", "magnetostatic"]
type KnownProblemType = (
    DrivenProblemType
    | EigenmodeProblemType
    | ElectrostaticProblemType
    | MagnetostaticProblemType
)


def normalize_problem_type(problem_type: str | None) -> str | None:
    """Return the lowercase routing key for a supported Palace problem type."""
    if problem_type is None:
        return None
    normalized = str(problem_type).strip().lower()
    aliases = {
        "driven": "driven",
        "eigenmode": "eigenmode",
        "electrostatic": "electrostatic",
        "magnetostatic": "magnetostatic",
    }
    return aliases.get(normalized)


def canonical_problem_type(problem_type: str | None) -> str | None:
    """Return the notebook-facing canonical Palace problem type name."""
    normalized = normalize_problem_type(problem_type)
    if normalized == "driven":
        return "Driven"
    if normalized == "eigenmode":
        return "Eigenmode"
    if normalized == "electrostatic":
        return "Electrostatic"
    if normalized == "magnetostatic":
        return "Magnetostatic"
    return problem_type


__all__ = [
    "DrivenProblemType",
    "EigenmodeProblemType",
    "ElectrostaticProblemType",
    "KnownProblemType",
    "MagnetostaticProblemType",
    "canonical_problem_type",
    "normalize_problem_type",
]
