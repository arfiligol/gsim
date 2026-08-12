"""gdsfactory port lowering helpers for Palace simulations.

This package provides the notebook-facing helpers that attach Palace intent to
live gdsfactory ports and later lower that metadata into
``gsim.palace.models.PalacePort`` records.

Stable port models live in ``gsim.palace.models``, while mesh sheet creation,
physical groups, Palace JSON generation, and result resolution stay in their
own packages. The port pipeline is ``PalaceSimBase.add_port()`` to these
helpers, then to mesh generation through the model records.
"""

from __future__ import annotations

from gsim.palace.ports.lowering import (
    configure_cpw_port,
    configure_inplane_port,
    configure_via_port,
    configure_wave_port,
    extract_ports,
)

__all__ = [
    "configure_cpw_port",
    "configure_inplane_port",
    "configure_via_port",
    "configure_wave_port",
    "extract_ports",
]
