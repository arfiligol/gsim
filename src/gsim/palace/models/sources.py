"""Boundary source selector models for Palace simulations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class CurrentSourceConfig(BaseModel):
    """Configuration for a magnetostatic surface-current source.

    Surface-current sources select conductor surfaces by layer and, optionally,
    by an XY center point when multiple conductor islands share the same layer.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str
    center: tuple[float, float] | None = None
    direction: str = "+X"

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, value: str) -> str:
        direction = value.strip().upper()
        if direction in {"X", "Y", "Z"}:
            direction = f"+{direction}"
        if direction not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise ValueError(
                "Current source direction must be one of +X, -X, +Y, -Y, +Z, -Z."
            )
        return direction


__all__ = ["CurrentSourceConfig"]
