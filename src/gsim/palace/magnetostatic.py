"""Magnetostatic simulation class for inductance-oriented workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    CurrentSourceConfig,
    MagnetostaticConfig,
    MaterialConfig,
    NumericalConfig,
    WavePortConfig,
)

logger = logging.getLogger(__name__)


class MagnetostaticSim(PalaceSimMixin, BaseModel):
    """Magnetostatic simulation for surface-current source problems.

    This class configures Palace magnetostatic runs without introducing a
    separate meshing path. Sources are selected from the generated conductor
    physical groups by layer and optional XY center point, matching the public
    electrostatic terminal-selection pattern.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    simulation_type: Literal["magnetostatic"] = "magnetostatic"

    driven: None = None
    ports: None = None
    cpw_ports: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    terminals: None = None
    electrostatic: None = None
    eigenmode: None = None
    absorbing_boundary: bool = False

    geometry: Geometry | None = None
    stack: LayerStack | None = None

    current_sources: list[CurrentSourceConfig] = Field(default_factory=list)
    magnetostatic: MagnetostaticConfig = Field(default_factory=MagnetostaticConfig)

    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, float] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)

    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_sources: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)
    _last_postprocessing_config: Any = PrivateAttr(default=None)

    def add_current_source(
        self,
        name: str,
        *,
        layer: str,
        center: tuple[float, float] | None = None,
        direction: str = "+X",
    ) -> None:
        """Add or replace a magnetostatic surface-current source.

        Args:
            name: Source name used in the generated index map.
            layer: Target conductor layer.
            center: Optional XY point used to select one conductor island on
                ``layer``. When omitted, all conductor surfaces on the layer
                are assigned to the source.
            direction: Palace current direction, one of ``+X``, ``-X``,
                ``+Y``, ``-Y``, ``+Z``, or ``-Z``.
        """
        self.current_sources = [
            source for source in self.current_sources if source.name != name
        ]
        self.current_sources.append(
            CurrentSourceConfig(
                name=name,
                layer=layer,
                center=center,
                direction=direction,
            )
        )
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
