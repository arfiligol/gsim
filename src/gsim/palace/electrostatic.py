"""Electrostatic simulation class for capacitance extraction.

This module provides the ElectrostaticSim class for extracting
capacitance matrices between terminals.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field, PrivateAttr

from gsim.palace.base import PalaceSimBase
from gsim.palace.models import (
    ElectrostaticConfig,
    TerminalConfig,
    WavePortConfig,
)

logger = logging.getLogger(__name__)


class ElectrostaticSim(PalaceSimBase):
    """Electrostatic simulation for capacitance matrix extraction.

    This class configures and runs electrostatic simulations to extract
    the capacitance matrix between conductor terminals. Unlike driven
    and eigenmode simulations, this does not use ports.

    Example:
        >>> from gsim.palace import ElectrostaticSim
        >>>
        >>> sim = ElectrostaticSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack()
        >>> sim.set_airbox(margin_x=120.0, margin_above=120.0, margin_below=20.0)
        >>> sim.add_terminal("T1", layer="topmetal2")
        >>> sim.add_terminal("T2", layer="topmetal2")
        >>> sim.set_electrostatic()
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()  # dict[str, Path]
        >>> print(results["terminal-C.csv"])

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        terminals: List of terminal configurations
        electrostatic: Electrostatic simulation configuration
        materials: Material property overrides
        numerical: Numerical solver configuration
    """

    simulation_type: Literal["electrostatic"] = "electrostatic"

    driven: None = None
    ports: None = None
    cpw_ports: None = None
    wave_ports: list[WavePortConfig] = Field(default_factory=list)

    # Terminal configurations (no ports in electrostatic)
    terminals: list[TerminalConfig] = Field(default_factory=list)

    # Electrostatic simulation config
    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)
    eigenmode: None = None
    absorbing_boundary: bool = False

    _configured_terminals: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Terminal methods
    # -------------------------------------------------------------------------

    def add_terminal(
        self,
        name: str,
        *,
        layer: str,
        center: tuple[float, float] | None = None,
        port_name: str | None = None,
        physical_label: str | None = None,
    ) -> None:
        """Add a terminal for capacitance extraction.

        Terminals define conductor surfaces for capacitance matrix extraction.

        Args:
            name: Terminal name
            layer: Target conductor layer
            center: Optional XY point used to select one conductor island on
                ``layer``. When omitted, all conductor surfaces on the layer
                are assigned to the terminal.
            port_name: Optional component port name used to derive ``center``.
            physical_label: Optional short label for generated physical names.

        Example:
            >>> sim.add_terminal("T1", layer="topmetal2")
            >>> sim.add_terminal("T2", layer="topmetal2")
        """
        if center is None and port_name is not None:
            port = self._find_gf_port(port_name)
            center = (float(port.center[0]), float(port.center[1]))

        # Remove existing terminal with same name
        self.terminals = [t for t in self.terminals if t.name != name]
        self.terminals.append(
            TerminalConfig(
                name=name,
                layer=layer,
                center=center,
                port_name=port_name,
                physical_label=physical_label,
            )
        )

    # -------------------------------------------------------------------------
    # Electrostatic configuration
    # -------------------------------------------------------------------------

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> None:
        """Configure electrostatic simulation.

        Args:
            save_fields: Number of field solutions to save

        Example:
            >>> sim.set_electrostatic(save_fields=1)
        """
        self.electrostatic = ElectrostaticConfig(
            save_fields=save_fields,
        )


__all__ = ["ElectrostaticSim"]
