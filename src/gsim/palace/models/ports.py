"""Port model contracts and LumpedPort direction normalization.

Owns the Pydantic port configuration models and the internal Cartesian
normalization boundary for Palace LumpedPort solver directions. Mesh geometry
placement belongs to ``gsim.palace.mesh``/port lowering, and radial current
source direction semantics remain in the source models.

``generate_sheet`` is stored here because it is port intent. Selecting or
creating the resulting solver boundary surface remains a mesh responsibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PalaceDirection = tuple[float, float, float]
PalaceDirectionInput = str | tuple[float, float, float] | list[float]

_CARTESIAN_DIRECTIONS: dict[str, PalaceDirection] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}


def normalize_palace_direction(value: PalaceDirectionInput) -> PalaceDirection:
    """Normalize a LumpedPort direction to a unit Cartesian vector.

    This primitive is intentionally Cartesian-only for LumpedPort v1. Radial
    ``+R``/``-R`` labels remain valid for magnetostatic current sources through
    their separate source model, but not for lumped-port solver direction.
    """
    if isinstance(value, str):
        direction = value.strip().upper()
        if direction in {"X", "Y", "Z"}:
            direction = f"+{direction}"
        if direction not in _CARTESIAN_DIRECTIONS:
            raise ValueError(
                "LumpedPort Direction must be one of +X, -X, +Y, -Y, +Z, -Z, "
                "or a finite nonzero 3-vector."
            )
        return _CARTESIAN_DIRECTIONS[direction]

    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LumpedPort Direction must be one of +X, -X, +Y, -Y, +Z, -Z, "
            "or a finite nonzero 3-vector."
        ) from exc

    if len(vector) != 3:
        raise ValueError("LumpedPort vector Direction must have exactly 3 values.")
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("LumpedPort vector Direction values must be finite.")

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise ValueError("LumpedPort vector Direction must be nonzero.")

    return tuple(
        0.0 if abs(component / norm) < 1e-15 else component / norm
        for component in vector
    )


def palace_direction_from_orientation(orientation: float | None) -> PalaceDirection:
    """Return the in-plane unit vector implied by a gdsfactory port orientation."""
    angle = math.radians(float(orientation) if orientation is not None else 0.0)
    return normalize_palace_direction((math.cos(angle), math.sin(angle), 0.0))


class PortType(Enum):
    """Palace port types used by mesh/config lowering."""

    LUMPED = "lumped"
    WAVEPORT = "waveport"


class PortGeometry(Enum):
    """Internal port geometry categories consumed by mesh generation."""

    INPLANE = "inplane"
    VIA = "via"


@dataclass
class PalacePort:
    """Mesh-ready Palace port record shared by mesh and config generation."""

    name: str
    port_type: PortType = PortType.LUMPED
    geometry: PortGeometry = PortGeometry.INPLANE
    center: tuple[float, float] = (0.0, 0.0)
    width: float = 0.0
    orientation: float = 0.0

    zmin: float = 0.0
    zmax: float = 0.0

    layer: str | None = None
    from_layer: str | None = None
    to_layer: str | None = None

    length: float | None = None
    generate_sheet: bool = True
    sheet_gds_layer: tuple[int, int] | None = None

    multi_element: bool = False
    centers: list[tuple[float, float]] | None = None
    direction: PalaceDirection | None = None
    directions: list[PalaceDirection] | None = None

    impedance: float = 50.0
    resistance: float | None = None
    inductance: float | None = None
    capacitance: float | None = None
    excited: bool = True

    z_margin: float = 0.0
    lateral_margin: float = 0.0
    max_size: bool = False
    mode: int = 1
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Normalize solver-direction fields to unit Cartesian vectors."""
        if self.direction is None:
            self.direction = (
                (0.0, 0.0, 1.0)
                if self.geometry == PortGeometry.VIA
                else palace_direction_from_orientation(self.orientation)
            )
        else:
            self.direction = normalize_palace_direction(self.direction)

        if self.directions is not None:
            self.directions = [
                normalize_palace_direction(direction) for direction in self.directions
            ]


class PortConfig(BaseModel):
    """Configuration for a single-element lumped port.

    Lumped ports can be inplane (horizontal, on single layer) or
    via (vertical, between two layers).

    Attributes:
        name: Port name (must match component port name)
        layer: Target layer for inplane ports
        from_layer: Bottom layer for via ports
        to_layer: Top layer for via ports
        length: Port extent along direction (um)
    offset: Shift port inward along the waveguide (um).
        Positive = away from boundary, into conductor.
    impedance: Port impedance (Ohms)
    excited: Whether this port is excited
    geometry: Port geometry type ("inplane" or "via")
    generate_sheet: If True, mesh creates the port sheet. If False, mesh
        selects a layout-authored horizontal solver sheet from the simulation
        layer catalog.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str | None = None
    from_layer: str | None = None
    to_layer: str | None = None
    length: float | None = Field(default=None, gt=0)
    impedance: float = Field(default=50.0, gt=0)
    resistance: float | None = Field(
        default=None, ge=0, description="Resistance in Ohms"
    )
    inductance: float | None = Field(default=None, ge=0, description="Inductance in H")
    capacitance: float | None = Field(
        default=None, ge=0, description="Capacitance in F"
    )
    direction: PalaceDirectionInput | None = Field(
        default=None,
        description=(
            "Solver field/polarization direction for Palace LumpedPort.Direction. "
            "Geometry still comes from the gdsfactory port center, width, "
            "orientation, and layer."
        ),
    )
    excited: bool = True
    geometry: Literal["inplane", "via"] = "inplane"
    generate_sheet: bool = True
    offset: float = Field(
        default=0.0,
        description="Shift port inward along the waveguide (um). "
        "Positive = away from boundary, into conductor.",
    )

    @model_validator(mode="after")
    def validate_layer_config(self) -> Self:
        """Validate layer configuration based on geometry type."""
        if self.geometry == "inplane" and self.layer is None:
            raise ValueError("Inplane ports require 'layer' to be specified")
        if self.geometry == "via" and (
            self.from_layer is None or self.to_layer is None
        ):
            raise ValueError("Via ports require both 'from_layer' and 'to_layer'")
        if self.geometry == "via" and not self.generate_sheet:
            raise ValueError("Authored sheets are only supported for inplane ports.")
        return self

    @field_validator("direction", mode="before")
    @classmethod
    def _validate_direction(
        cls, value: PalaceDirectionInput | None
    ) -> PalaceDirection | None:
        if value is None:
            return None
        return normalize_palace_direction(value)


class CPWPortConfig(BaseModel):
    """Configuration for a coplanar waveguide (CPW) port.

    CPW ports consist of two elements (upper and lower gaps) that are
    excited with opposite E-field directions to create the CPW mode.

    The port is placed at the center of the signal conductor. The two
    gap element surfaces are computed from s_width and gap_width.

    Attributes:
        name: Port name (must match a single component port at the signal center)
        layer: Target conductor layer
        s_width: Width of the signal (center) conductor (um)
        gap_width: Width of each gap between signal and ground (um)
        length: Port extent along direction (um)
        offset: Shift the port along the waveguide direction (um).
            Positive moves in the port orientation direction.
        impedance: Port impedance (Ohms)
        excited: Whether this port is excited
        generate_sheet: If True, mesh creates CPW element sheets. If False,
            mesh selects layout-authored horizontal solver sheets from the
            simulation layer catalog.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description="Port name matching component port")
    layer: str = Field(description="Target conductor layer")
    s_width: float = Field(gt=0, description="Signal conductor width (um)")
    gap_width: float = Field(
        gt=0, description="Gap width between signal and ground (um)"
    )
    length: float = Field(default=2.0, gt=0, description="Port extent in um")
    offset: float | None = Field(
        default=None,
        description="Shift port inward along the waveguide (um). "
        "Positive = away from boundary, into conductor. "
        "Defaults to length/2 (port flush with conductor edge).",
    )

    @model_validator(mode="after")
    def _default_offset(self) -> Self:
        """Default offset to length/2 so the port is flush with the conductor edge."""
        if self.offset is None:
            self.offset = self.length / 2
        return self

    impedance: float = Field(default=50.0, gt=0)
    excited: bool = True
    generate_sheet: bool = True


class TerminalConfig(BaseModel):
    """Configuration for a terminal (for electrostatic capacitance extraction).

    Terminals define conductor surfaces for capacitance matrix extraction
    in electrostatic simulations.

    Attributes:
        name: Terminal name
        layer: Target conductor layer
        center: Optional XY point used to select one conductor island on layer
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str
    center: tuple[float, float] | None = None


class WavePortConfig(BaseModel):
    """Configuration for a wave port (domain boundary with mode solving).

    Wave ports are used for domain-boundary ports where mode solving
    is needed. This is an alternative to lumped ports for more accurate
    S-parameter extraction.

    Attributes:
        name: Port name (must match component port name)
        layer: Target conductor layer
        z_margin: Margin to extend port geometry in z-direction (um)
        max_size: If True, set z_margin and lateral_margin to
        fill the full simulation domain
        mode: Mode number to excite
        offset: De-embedding distance in um
        excited: Whether this port is excited
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str
    layer: str | None = None
    z_margin: float = Field(default=0, ge=0)
    lateral_margin: float = Field(default=0.0, ge=0)
    max_size: bool = Field(
        default=False,
        description=(
            "When True, set z_margin and lateral_margin"
            " to fill the full simulation domain"
        ),
    )
    mode: int = Field(default=1, ge=1, description="Mode number to excite")
    offset: float = Field(default=0.0, ge=0, description="De-embedding distance in um")
    excited: bool = True


__all__ = [
    "CPWPortConfig",
    "PalaceDirection",
    "PalaceDirectionInput",
    "PalacePort",
    "PortConfig",
    "PortGeometry",
    "PortType",
    "TerminalConfig",
    "WavePortConfig",
    "normalize_palace_direction",
    "palace_direction_from_orientation",
]
