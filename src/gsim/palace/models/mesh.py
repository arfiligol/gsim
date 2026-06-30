"""Mesh configuration models for Palace simulations.

This module contains Pydantic models for mesh generation configuration.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeshConfig(BaseModel):
    """Configuration for mesh generation with quality presets.

    Attributes:
        refined_mesh_size: Mesh size near conductors (um)
        max_mesh_size: Maximum mesh size in air/dielectric (um)
        cells_per_wavelength: Number of mesh cells per wavelength
        margin: XY margin around design (um)
        margin_x: X-axis margin override (um). Falls back to margin.
        margin_y: Y-axis margin override (um). Falls back to margin.
        airbox_margin: Extra airbox around stack (um); 0 = disabled
        fmax: Maximum frequency for mesh sizing (Hz)
        boundary_conditions: List of boundary conditions for each face
        planar_conductors: Treat conductors as 2D PEC surfaces instead of volumes
        show_gui: Show gmsh GUI during meshing
        preview_only: Generate preview only, don't save mesh
    """

    model_config = ConfigDict(validate_assignment=True)

    refined_mesh_size: float = Field(default=5.0, gt=0)
    max_mesh_size: float = Field(default=300.0, gt=0)
    cells_per_wavelength: int = Field(default=10, ge=1)
    margin: float = Field(default=0.0, ge=0)
    margin_x: float | None = Field(
        default=None,
        ge=0,
        description="X margin override (um). Falls back to margin.",
    )
    margin_y: float | None = Field(
        default=None,
        ge=0,
        description="Y margin override (um). Falls back to margin.",
    )
    airbox_margin: float = Field(default=0.0, ge=0)
    fmax: float = Field(default=100e9, gt=0)
    boundary_conditions: list[str] | None = None
    planar_conductors: bool = False
    surface_epr_enabled: bool = False
    surface_epr_representation: Literal["A", "B", "C"] = Field(
        default="B",
        description=(
            "Surface EPR route representation: A/B/C share Full-3D "
            "interface discovery; A/B remove conductor volumes during final "
            "topology materialization, C keeps them."
        ),
    )
    surface_epr_inset_margins_um: tuple[float, ...] = Field(
        default=(0.0, 0.05),
        description=(
            "Surface EPR inset margins in um. 0 means total; positive values "
            "define generated interface inset partitions."
        ),
    )
    merge_via_distance: float = Field(default=2.0, ge=0)
    curve_fit_mode: Literal["line", "spline", "bspline"] = "line"
    curve_fit_layers: list[str] = Field(default_factory=lambda: ["core", "core2"])
    curve_fit_tolerance_um: float = Field(default=0.0, ge=0)
    curve_fit_min_points: int = Field(default=8, ge=3)
    curve_fit_corner_angle_deg: float = Field(default=45.0, gt=0, lt=180)
    high_order_elements: bool = False
    high_order_order: int = Field(default=2, ge=2, le=6)
    high_order_optimize: bool = True
    show_gui: bool = False
    preview_only: bool = False

    @property
    def effective_margin_x(self) -> float:
        """Resolved X margin (margin_x if set, else margin)."""
        return self.margin_x if self.margin_x is not None else self.margin

    @property
    def effective_margin_y(self) -> float:
        """Resolved Y margin (margin_y if set, else margin)."""
        return self.margin_y if self.margin_y is not None else self.margin

    @model_validator(mode="after")
    def set_default_boundary_conditions(self) -> Self:
        """Set default boundary conditions if not provided."""
        if self.boundary_conditions is None:
            object.__setattr__(
                self,
                "boundary_conditions",
                ["ABC", "ABC", "ABC", "ABC", "ABC", "ABC"],
            )
        margins = tuple(
            sorted({float(value) for value in self.surface_epr_inset_margins_um})
        )
        if any(value < 0.0 or not math.isfinite(value) for value in margins):
            raise ValueError("surface_epr_inset_margins_um values must be finite >= 0")
        if 0.0 not in margins:
            margins = (0.0, *margins)
        object.__setattr__(self, "surface_epr_inset_margins_um", margins)
        object.__setattr__(
            self,
            "surface_epr_representation",
            str(self.surface_epr_representation).upper(),
        )
        return self

    @classmethod
    def coarse(cls, **kwargs: Any) -> Self:
        """Fast mesh for quick iteration (~2.5 elements per wavelength).

        This preset is suitable for initial debugging and quick checks.
        Not recommended for accurate results.
        """
        defaults: dict[str, Any] = {
            "refined_mesh_size": 10.0,
            "max_mesh_size": 600.0,
            "cells_per_wavelength": 5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def default(cls, **kwargs: Any) -> Self:
        """Balanced mesh (~5 elements per wavelength).

        This preset provides a good balance between accuracy and computation time.
        Suitable for most simulations.
        """
        defaults: dict[str, Any] = {
            "refined_mesh_size": 5.0,
            "max_mesh_size": 300.0,
            "cells_per_wavelength": 10,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def fine(cls, **kwargs: Any) -> Self:
        """High accuracy mesh (~10 elements per wavelength).

        This preset provides higher accuracy at the cost of increased
        computation time. Use for final production simulations.
        """
        defaults: dict[str, Any] = {
            "refined_mesh_size": 2.0,
            "max_mesh_size": 70.0,
            "cells_per_wavelength": 20,
        }
        defaults.update(kwargs)
        return cls(**defaults)


__all__ = [
    "MeshConfig",
]
