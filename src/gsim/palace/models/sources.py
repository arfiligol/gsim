"""Boundary source selector models for Palace simulations.

These models describe user-facing source intent. The Palace config generator
is responsible for lowering that intent into mesh physical-group attributes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CurrentDirection = str | tuple[float, float, float] | list[float]
CoordinateSystem = Literal["Cartesian", "Cylindrical"]


def _normalize_direction(value: CurrentDirection) -> CurrentDirection:
    if isinstance(value, str):
        direction = value.strip().upper()
        if direction in {"X", "Y", "Z", "R"}:
            direction = f"+{direction}"
        if direction not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z", "+R", "-R"}:
            raise ValueError(
                "Current source direction must be one of +X, -X, +Y, -Y, "
                "+Z, -Z, +R, -R, or a 3-vector."
            )
        return direction

    try:
        direction_vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Current source direction must be one of +X, -X, +Y, -Y, "
            "+Z, -Z, +R, -R, or a 3-vector."
        ) from exc
    if len(direction_vector) != 3:
        raise ValueError("Current source vector direction must have 3 values.")
    if all(component == 0.0 for component in direction_vector):
        raise ValueError("Current source vector direction must be nonzero.")
    return direction_vector


def palace_direction(value: CurrentDirection) -> str | list[float]:
    """Return a JSON-safe Palace direction value."""
    if isinstance(value, str):
        return value
    return [float(item) for item in value]


class CurrentSourceElementConfig(BaseModel):
    """Selector for one element of a multielement surface-current source.

    The element stores the same layer/center selection intent as a single
    current source. It does not store raw Palace boundary attributes.
    """

    model_config = ConfigDict(validate_assignment=True)

    layer: str
    center: tuple[float, float] | None = None
    direction: CurrentDirection = "+X"
    coordinate_system: CoordinateSystem | None = None

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: CurrentDirection) -> CurrentDirection:
        return _normalize_direction(value)


class CurrentSourceConfig(BaseModel):
    """Configuration for a magnetostatic surface-current source.

    Surface-current sources select conductor surfaces by layer and, optionally,
    by an XY center point when multiple conductor islands share the same layer.
    Multielement sources keep the same selector contract on each element and
    let the config generator emit Palace ``Elements`` entries.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str | None = None
    center: tuple[float, float] | None = None
    direction: CurrentDirection = "+X"
    coordinate_system: CoordinateSystem | None = None
    elements: tuple[CurrentSourceElementConfig, ...] = Field(default_factory=tuple)

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(cls, value: CurrentDirection) -> CurrentDirection:
        return _normalize_direction(value)

    @model_validator(mode="after")
    def _validate_selector_shape(self) -> CurrentSourceConfig:
        if self.elements:
            if (
                self.layer is not None
                or self.center is not None
                or self.coordinate_system is not None
            ):
                raise ValueError(
                    "Multielement current sources must put layer, center, and "
                    "coordinate system on each element, not on the parent source."
                )
            return self
        if not self.layer:
            raise ValueError(
                "Current source layer is required unless elements are provided."
            )
        return self


__all__ = [
    "CoordinateSystem",
    "CurrentDirection",
    "CurrentSourceConfig",
    "CurrentSourceElementConfig",
    "palace_direction",
]
