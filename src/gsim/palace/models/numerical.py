"""Numerical solver configuration models for Palace simulations.

This module owns the small ``gsim`` numerical API and serializes it to Palace
JSON fragments. Full Palace key coverage stays with the vendored JSON schemas.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gsim.palace.models.versions import PalaceConfigVersion

PalaceLinearSolverType = Literal[
    "AMS",
    "BoomerAMG",
    "SuperLU",
    "MUMPS",
    "STRUMPACK",
    "Jacobi",
    "Default",
]
PalaceDevice = Literal["CPU", "GPU", "Debug"]


class NumericalConfig(BaseModel):
    """Numerical solver configuration for Palace FEM simulations.

    Attributes:
        order: Finite element polynomial order (1-4). Higher order means more
            accurate field approximation per mesh element at higher cost.
            order=1: fast, low accuracy. order=2: good balance (default).
            order=3-4: high accuracy, significantly more DOFs. Increasing order
            can reduce lumped port reflection artifacts and is often more
            cost-effective than mesh refinement for smooth geometries.
        tolerance: Linear solver relative residual convergence tolerance.
            Tighter tolerance (e.g. 1e-8) gives more accurate solves at the
            cost of more iterations. Default 1e-6 is suitable for most cases.
        max_iterations: Maximum Krylov solver iterations before giving up.
            Increase if you see "solver did not converge" warnings.
        solver_type: Linear solver / preconditioner backend.
            "Default" auto-selects (AMS for curl-curl, sparse direct for
            frequency domain). "SuperLU", "STRUMPACK", "MUMPS" are sparse
            direct solvers — more robust but use more memory.
        preconditioner: Preconditioner for iterative solves.
            "AMS" (Auxiliary-space Maxwell Solver) is best for EM problems.
            "BoomerAMG" is an algebraic multigrid alternative.
        device: Compute device. "GPU" enables GPU-accelerated assembly and
            solves if Palace was built with GPU support.
    """

    model_config = ConfigDict(validate_assignment=True)

    order: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Finite element polynomial order. Higher order = more accurate "
        "fields per element but more expensive. order=1: fast/low accuracy, "
        "order=2: good balance (default), order=3-4: high accuracy. "
        "Increasing order can reduce lumped port reflection artifacts.",
    )

    tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Linear solver relative residual convergence tolerance. "
        "Tighter (e.g. 1e-8) gives more accurate solves at higher cost.",
    )
    max_iterations: int = Field(
        default=400,
        ge=1,
        description="Maximum Krylov solver iterations. Increase if solver "
        "does not converge.",
    )
    solver_type: PalaceLinearSolverType = Field(
        default="Default",
        description="Linear solver backend. 'Default' auto-selects. "
        "Direct solvers (SuperLU, STRUMPACK, MUMPS) are more robust "
        "but use more memory. AMS, BoomerAMG, and Jacobi select iterative "
        "preconditioners directly.",
    )

    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = Field(
        default="Default",
        description="Convenience alias for Solver.Linear.Type when solver_type "
        "is 'Default'. 'AMS' is best for EM curl-curl problems. "
        "'BoomerAMG' is an algebraic multigrid alternative.",
    )

    device: PalaceDevice = Field(
        default="CPU",
        description="Compute device. 'GPU' enables GPU-accelerated assembly "
        "and solves if Palace was built with GPU support.",
    )

    linear_solver: dict[str, Any] | None = Field(
        default=None,
        description="Advanced Solver.Linear Palace JSON fragment. When set, it "
        "takes precedence over the simple tolerance/max_iterations/"
        "solver_type/preconditioner convenience fields.",
    )

    def to_linear_solver_config(
        self,
        *,
        palace_version: PalaceConfigVersion = "0.16.0",
    ) -> dict[str, object]:
        """Convert to Palace ``Solver.Linear`` config.

        Notes:
            - For ``solver_type='MUMPS'``, direct-solver defaults follow the
              existing Palace TODO template in this codebase.
            - ``preconditioner`` is applied as ``Solver.Linear.Type`` when
              ``solver_type='Default'``.
            - Rare Palace-native keys should be added through ``set_linear_solver``
              or config hints and checked by final JSON schema validation.
        """
        _ = palace_version
        if self.linear_solver is not None:
            return deepcopy(self.linear_solver)

        solver_type = self.solver_type
        if solver_type == "Default" and self.preconditioner != "Default":
            solver_type = self.preconditioner

        linear: dict[str, object] = {
            "Type": solver_type,
            "KSPType": "GMRES",
            "Tol": self.tolerance,
            "MaxIts": self.max_iterations,
        }
        if solver_type == "MUMPS":
            linear.update(
                {
                    "MaxIts": 1,
                    "MGMaxLevels": 1,
                    "EstimatorMaxIts": 0,
                    "EstimatorTol": 1e-6,
                    "DivFreeTol": 1e-6,
                    "DivFreeMaxIts": 0,
                    "PCMatReal": False,
                    "ComplexCoarseSolve": True,
                }
            )
        return linear

    def to_solver_config(
        self,
        *,
        palace_version: PalaceConfigVersion = "0.16.0",
    ) -> dict[str, object]:
        """Convert to Palace ``Solver`` section config."""
        return {
            "Order": self.order,
            "Device": self.device,
            "Linear": self.to_linear_solver_config(palace_version=palace_version),
        }

    def to_palace_config(
        self,
        *,
        palace_version: PalaceConfigVersion = "0.16.0",
    ) -> dict[str, object]:
        """Convert to the official Palace ``Solver`` JSON shape."""
        return self.to_solver_config(palace_version=palace_version)


__all__ = [
    "NumericalConfig",
]
