"""Magnetostatic simulation class for inductance-oriented workflows."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import Field, PrivateAttr

from gsim.palace.base import PalaceSimBase
from gsim.palace.models import (
    CurrentSourceConfig,
    CurrentSourceElementConfig,
    MagnetostaticConfig,
    WavePortConfig,
)

logger = logging.getLogger(__name__)


class MagnetostaticSim(PalaceSimBase):
    """Magnetostatic simulation for surface-current source problems.

    This class configures Palace magnetostatic runs without introducing a
    separate meshing path. Sources are selected from the generated conductor
    physical groups by layer and optional XY center point, matching the public
    electrostatic terminal-selection pattern.
    """

    simulation_type: Literal["magnetostatic"] = "magnetostatic"

    driven: None = None
    ports: None = None
    cpw_ports: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    terminals: None = None
    electrostatic: None = None
    eigenmode: None = None
    absorbing_boundary: bool = False

    current_sources: list[CurrentSourceConfig] = Field(default_factory=list)
    magnetostatic: MagnetostaticConfig = Field(default_factory=MagnetostaticConfig)

    _configured_sources: bool = PrivateAttr(default=False)

    def add_current_source(
        self,
        name: str,
        *,
        layer: str | None = None,
        center: tuple[float, float] | None = None,
        direction: str | tuple[float, float, float] | list[float] | None = None,
        coordinate_system: Literal["Cartesian", "Cylindrical"] | None = None,
        elements: tuple[CurrentSourceElementConfig | dict[str, Any], ...] = (),
    ) -> None:
        """Add or replace a magnetostatic surface-current source.

        Args:
            name: Source name used in the generated index map.
            layer: Target conductor layer.
            center: Optional XY point used to select one conductor island on
                ``layer``. When omitted, all conductor surfaces on the layer
                are assigned to the source.
            direction: Palace current direction. Axis keywords, radial
                keywords, and 3-vectors are supported.
            coordinate_system: Coordinate system for vector directions.
            elements: Optional selector-based elements for a multielement
                Palace surface-current source. When provided, put ``layer`` and
                ``center`` on each element instead of on the parent source.
        """
        self.current_sources = [
            source for source in self.current_sources if source.name != name
        ]
        source_kwargs: dict[str, Any] = {
            "name": name,
            "layer": layer,
            "center": center,
            "coordinate_system": coordinate_system,
            "elements": elements,
        }
        if direction is not None:
            source_kwargs["direction"] = direction
        self.current_sources.append(CurrentSourceConfig(**source_kwargs))
        self._configured_sources = True

    def set_magnetostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> None:
        """Configure magnetostatic solver output.

        Args:
            save_fields: Number of field solutions to save.
        """
        self.magnetostatic = MagnetostaticConfig(save_fields=save_fields)


__all__ = ["MagnetostaticSim"]
