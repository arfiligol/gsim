"""Typed activation intent for Palace simulation regions.

This module owns the public model used by simulation classes to declare which
stack layers should become explicit 3D dielectric regions in the Palace mesh.
It records region intent only; mesh geometry creation, physical groups, Palace
Domains, and Palace Boundaries stay in their mesh-owner modules.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ActivatedRegionRole = Literal["substrate", "inter_die_vacuum", "outer_vacuum"]


class ActivatedRegion(BaseModel):
    """A stack layer selected for explicit Palace mesh activation."""

    model_config = ConfigDict(validate_assignment=True)

    layer: str = Field(min_length=1)
    role: ActivatedRegionRole
    die: str | None = None
    lower_die: str | None = None
    upper_die: str | None = None
    margin_x: float = Field(default=0.0, ge=0)
    margin_y: float = Field(default=0.0, ge=0)
    z_above: float = Field(default=0.0, ge=0)
    z_below: float = Field(default=0.0, ge=0)
    material: str | None = None

    @field_validator("layer", "die", "lower_die", "upper_die", "material")
    @classmethod
    def _strip_non_empty(cls, value: str | None) -> str | None:
        """Strip optional region names and reject blank values."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Activated region names must be non-empty.")
        return stripped

    @model_validator(mode="after")
    def _validate_role_fields(self) -> Self:
        """Require the role-specific region fields."""
        if self.role == "inter_die_vacuum" and (
            self.lower_die is None or self.upper_die is None
        ):
            raise ValueError(
                "Inter-die vacuum activation requires lower_die and upper_die."
            )
        return self


__all__ = [
    "ActivatedRegion",
    "ActivatedRegionRole",
]
