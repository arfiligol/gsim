"""Port lowering between notebook declarations and mesh-ready Palace ports.

This module writes Palace metadata onto gdsfactory ports and extracts that
metadata into ``PalacePort`` records for mesh generation. It owns the metadata
keys, live port lookup details, and CPW element intent derived from a single
signal-center gdsfactory port.

PDK simulation-layer catalogs, authored polygon selection, Gmsh surface
creation, physical groups, and Palace JSON generation are handled by the
surrounding model and mesh layers. The flow is notebook declaration to
gdsfactory port metadata, then ``extract_ports()``, then ``gsim.palace.mesh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gsim.palace.models.ports import (
    PalaceDirectionInput,
    PalacePort,
    PortGeometry,
    PortType,
    normalize_palace_direction,
    palace_direction_from_orientation,
)
from gsim.palace.models.simulation_layers import normalize_gds_layer

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack


def port_gds_layer(port) -> tuple[int, int]:
    """Return the full GDS layer/datatype tuple for a gdsfactory port."""
    layer_info = getattr(port, "layer_info", None)
    if layer_info is not None:
        layer = getattr(layer_info, "layer", None)
        datatype = getattr(layer_info, "datatype", None)
        if layer is not None and datatype is not None:
            return int(layer), int(datatype)

    try:
        return normalize_gds_layer(getattr(port, "layer", None))
    except ValueError as exc:
        raise ValueError(
            "Authored Palace solver sheets require the gdsfactory port to expose "
            "a full GDS layer/datatype tuple."
        ) from exc


def _shift_port_center(port, offset: float) -> None:
    """Shift a port's center inward along its orientation by `offset` um.

    Positive offset moves the port away from the domain boundary and into
    the conductor (same convention as configure_cpw_port).

    Args:
        port: gdsfactory Port whose center will be mutated in-place.
        offset: Distance in um. Positive = inward (against the outward normal).
    """
    import numpy as np

    orientation_rad = np.deg2rad(
        float(port.orientation) if port.orientation is not None else 0.0
    )
    # Port orientation points *outward*; negate so positive offset goes inward.
    longitudinal = np.array([np.cos(orientation_rad), np.sin(orientation_rad)])
    shifted = (
        np.array([float(port.center[0]), float(port.center[1])]) - longitudinal * offset
    )
    port.center = (float(shifted[0]), float(shifted[1]))


def configure_inplane_port(
    ports,
    layer: str,
    length: float,
    impedance: float = 50.0,
    excited: bool = True,
    offset: float = 0.0,
    direction: PalaceDirectionInput | None = None,
    generate_sheet: bool = True,
):
    """Configure gdsfactory port(s) as inplane (lumped) ports for Palace simulation.

    Inplane ports are horizontal ports on a single metal layer, used for CPW gaps
    or similar structures where excitation occurs in the XY plane.

    Args:
        ports: Single gdsfactory Port or iterable of Ports (e.g., c.ports)
        layer: Target conductor layer name (e.g., 'topmetal2')
        length: Port extent along direction in um (perpendicular to port width)
        direction: Optional solver field/polarization direction. Port-sheet
            geometry still follows the gdsfactory port orientation.
        impedance: Port impedance in Ohms (default: 50)
        excited: Whether port is excited vs just measured (default: True)
        offset: Shift port inward along the waveguide (um).
            Positive moves away from the boundary, into the conductor.
        generate_sheet: If True, mesh creates a sheet from port metadata. If
            False, mesh selects a layout-authored solver sheet from the
            simulation layer catalog using this gdsfactory port's layer.

    Examples:
        ```python
        configure_inplane_port(c.ports["o1"], layer="topmetal2", length=5.0)
        configure_inplane_port(c.ports, layer="topmetal2", length=5.0)  # all ports
        configure_inplane_port(c.ports["o1"], layer="topmetal2", length=5.0, offset=2.0)
        ```
    """
    # Handle single port or iterable
    port_list = [ports] if hasattr(ports, "info") else ports

    for port in port_list:
        if offset != 0.0:
            _shift_port_center(port, offset)

        port.info["palace_type"] = "lumped"
        port.info["layer"] = layer
        port.info["length"] = length
        port.info["direction"] = (
            normalize_palace_direction(direction) if direction is not None else None
        )
        port.info["impedance"] = impedance
        port.info["excited"] = excited
        port.info["generate_sheet"] = generate_sheet
        port.info["sheet_gds_layer"] = None if generate_sheet else port_gds_layer(port)


def configure_via_port(
    ports,
    from_layer: str,
    to_layer: str,
    impedance: float = 50.0,
    excited: bool = True,
    offset: float = 0.0,
    direction: PalaceDirectionInput | None = None,
):
    """Configure gdsfactory port(s) as via (vertical) lumped ports.

    Via ports are vertical lumped ports between two metal layers, used for microstrip
    feed structures where excitation occurs in the Z direction.

    Args:
        ports: Single gdsfactory Port or iterable of Ports (e.g., c.ports)
        from_layer: Bottom conductor layer name (e.g., 'metal1')
        to_layer: Top conductor layer name (e.g., 'topmetal2')
        direction: Optional solver field/polarization direction. Port-sheet
            geometry still follows the gdsfactory port orientation.
        resistance: Series resistance in Ohms (default: 50)
        inductance: Series inductance in Henries (default: 0)
        capacitance: Shunt capacitance in Farads (default: 0)
        excited: Whether port is excited vs just measured (default: True)
        offset: Shift port inward along the waveguide in XY (um).
            Positive moves away from the boundary, into the conductor.

    Examples:
        ```python
        configure_via_port(c.ports["o1"], from_layer="metal1", to_layer="topmetal2")
        configure_via_port(
            c.ports, from_layer="metal1", to_layer="topmetal2"
        )  # all ports
        configure_via_port(
            c.ports["o1"], from_layer="metal1", to_layer="topmetal2", offset=2.0
        )
        ```
    """
    # Handle single port or iterable
    port_list = [ports] if hasattr(ports, "info") else ports

    for port in port_list:
        if offset != 0.0:
            _shift_port_center(port, offset)

        port.info["palace_type"] = "lumped"
        port.info["from_layer"] = from_layer
        port.info["to_layer"] = to_layer
        port.info["direction"] = (
            normalize_palace_direction(direction) if direction is not None else None
        )
        port.info["impedance"] = impedance
        port.info["excited"] = excited


def configure_cpw_port(
    port,
    layer: str,
    s_width: float,
    gap_width: float,
    length: float,
    impedance: float = 50.0,
    excited: bool = True,
    offset: float | None = None,
    generate_sheet: bool = True,
):
    """Configure a gdsfactory port as a CPW (multi-element) lumped port.

    In CPW (Ground-Signal-Ground), E-fields are opposite in the two gaps.
    The port should be placed at the signal center. The upper and lower gap
    centers are computed from the signal width and gap width.

    Args:
        port: gdsfactory Port at the signal center
        layer: Target conductor layer name (e.g., 'topmetal2')
        s_width: Signal conductor width in um
        gap_width: Gap width between signal and ground in um
        length: Port extent along direction (um)
        impedance: Port impedance in Ohms (default: 50)
        excited: Whether port is excited (default: True)
        offset: Shift port inward along the waveguide (um).
            Positive moves away from the boundary, into the conductor.
            Defaults to length/2 (port flush with conductor edge).
        generate_sheet: If True, mesh creates CPW element sheets. If False,
            mesh selects layout-authored solver sheets from the simulation layer
            catalog using this gdsfactory port's layer.

    Examples:
        ```python
        configure_cpw_port(
            c.ports["o1"],
            layer="topmetal2",
            s_width=10.0,
            gap_width=6.0,
            length=5.0,
        )
        ```
    """
    import numpy as np

    if offset is None:
        offset = length / 2

    center = np.array([float(port.center[0]), float(port.center[1])])
    orientation_rad = np.deg2rad(
        float(port.orientation) if port.orientation is not None else 0.0
    )

    # Longitudinal direction (along waveguide / port orientation)
    # Port orientation points *outward* from the component, so we
    # negate it: positive offset moves the port *inward* along the
    # waveguide (away from the boundary, into the conductor).
    longitudinal = np.array([np.cos(orientation_rad), np.sin(orientation_rad)])

    # Apply longitudinal offset
    if offset != 0.0:
        center = center - longitudinal * offset

    # Transverse direction (perpendicular to port orientation, in-plane)
    # Port orientation points along the waveguide; transverse is 90° CCW
    transverse = np.array([-np.sin(orientation_rad), np.cos(orientation_rad)])

    # Gap center offset from signal center
    gap_offset = (s_width + gap_width) / 2.0

    upper_center = center + transverse * gap_offset
    lower_center = center - transverse * gap_offset

    # Store computed CPW element info on the single port
    port.info["palace_type"] = "cpw"
    port.info["layer"] = layer
    port.info["length"] = length
    port.info["impedance"] = impedance
    port.info["excited"] = excited
    port.info["generate_sheet"] = generate_sheet
    port.info["sheet_gds_layer"] = None if generate_sheet else port_gds_layer(port)
    port.info["cpw_upper_center"] = (float(upper_center[0]), float(upper_center[1]))
    port.info["cpw_lower_center"] = (float(lower_center[0]), float(lower_center[1]))
    port.info["cpw_gap_width"] = gap_width


def configure_wave_port(
    ports,
    layer: str,
    z_margin: float = 0.0,
    lateral_margin: float = 0.0,
    max_size: bool = False,
    mode: int = 1,
    excited: bool = True,
    offset: float = 0.0,
):
    """Configure gdsfactory port(s) as wave ports for Palace simulation.

    Wave ports are domain boundary ports where mode solving is needed.

    Args:
        ports: Single gdsfactory Port or iterable of Ports (e.g., c.ports)
        layer: Target conductor layer name (e.g., 'topmetal2')
        z_margin: Margin in the z-direction for the wave port
        lateral_margin: Margin in the x/y direction
        max_size: When True, automatically set z_margin and lateral_margin
            to fill the full simulation domain boundary on that side.
        mode: Mode number to excite.
        offset: Offset distance used for scattering parameter de-embedding.
        excited: Whether port is excited vs just measured (default: True)

    Examples:
        ```python
        configure_wave_port(
            c.ports["o1"], name="o1", layer="topmetal2", z_margin=5.0, mode=1
        )
        configure_wave_port(
            c.ports, name="all_ports", layer="topmetal2", z_margin=5.0, mode=1
        )  # all ports
        ```
    """
    # Handle single port or iterable
    port_list = [ports] if hasattr(ports, "info") else ports

    for port in port_list:
        port.info["palace_type"] = "waveport"
        port.info["layer"] = layer
        port.info["z_margin"] = z_margin
        port.info["lateral_margin"] = lateral_margin
        port.info["max_size"] = max_size
        port.info["mode"] = mode
        port.info["offset"] = offset
        port.info["excited"] = excited


def extract_ports(component, stack: LayerStack) -> list[PalacePort]:
    """Extract Palace ports from a gdsfactory component.

    Handles all port types: inplane, via, and CPW (multi-element).

    Args:
        component: gdsfactory Component with configured ports
        stack: LayerStack from stack module

    Returns:
        List of PalacePort objects ready for simulation
    """
    palace_ports = []

    for port in component.ports:
        info = port.info
        palace_type = info.get("palace_type")

        if palace_type is None:
            continue

        if palace_type == "cpw":
            # Single-port CPW: gap centers were pre-computed by configure_cpw_port
            layer_name = info.get("layer")
            zmin, zmax = 0.0, 0.0
            if layer_name and layer_name in stack.layers:
                layer = stack.layers[layer_name]
                zmin = layer.zmin
                zmax = layer.zmax

            upper_center = info["cpw_upper_center"]
            lower_center = info["cpw_lower_center"]
            gap_width = info["cpw_gap_width"]

            centers = [
                (float(upper_center[0]), float(upper_center[1])),
                (float(lower_center[0]), float(lower_center[1])),
            ]

            # Compute E-field directions from port orientation.
            # transverse = 90° CCW from the longitudinal direction = [-sin, cos]
            # upper_center = signal_center + transverse * gap_offset
            #   -> E-field in upper gap points toward signal: -transverse direction
            # lower_center = signal_center - transverse * gap_offset
            #   -> E-field in lower gap points toward signal: +transverse direction
            import numpy as np

            orientation_rad = np.deg2rad(
                float(port.orientation) if port.orientation is not None else 0.0
            )
            transverse = np.array([-np.sin(orientation_rad), np.cos(orientation_rad)])

            directions = [
                normalize_palace_direction(
                    (float(-transverse[0]), float(-transverse[1]), 0.0)
                ),
                normalize_palace_direction(
                    (float(transverse[0]), float(transverse[1]), 0.0)
                ),
            ]

            cpw_port = PalacePort(
                name=port.name,
                port_type=PortType.LUMPED,
                geometry=PortGeometry.INPLANE,
                center=(float(port.center[0]), float(port.center[1])),
                width=gap_width,
                orientation=float(port.orientation)
                if port.orientation is not None
                else 0.0,
                zmin=zmin,
                zmax=zmax,
                layer=layer_name,
                length=info.get("length"),
                generate_sheet=info.get("generate_sheet", True),
                sheet_gds_layer=info.get("sheet_gds_layer"),
                multi_element=True,
                centers=centers,
                directions=directions,
                impedance=info.get("impedance", 50.0),
                excited=info.get("excited", True),
            )
            palace_ports.append(cpw_port)
            continue

        # Handle single-element ports (lumped, waveport)
        center = (float(port.center[0]), float(port.center[1]))
        width = float(port.width)
        orientation = float(port.orientation) if port.orientation is not None else 0.0

        zmin, zmax = 0.0, 0.0
        from_layer = info.get("from_layer")
        to_layer = info.get("to_layer")
        layer_name = info.get("layer")

        if palace_type == "lumped":
            port_type = PortType.LUMPED
            if from_layer and to_layer:
                geometry = PortGeometry.VIA
                if from_layer in stack.layers:
                    zmin = stack.layers[from_layer].zmin
                if to_layer in stack.layers:
                    zmax = stack.layers[to_layer].zmax
            elif layer_name:
                geometry = PortGeometry.INPLANE
                if layer_name in stack.layers:
                    layer = stack.layers[layer_name]
                    zmin = layer.zmin
                    zmax = layer.zmax
            else:
                raise ValueError(f"Lumped port '{port.name}' missing layer info")
            direction = info.get("direction")
            if direction is None:
                direction = (
                    (0.0, 0.0, 1.0)
                    if geometry == PortGeometry.VIA
                    else palace_direction_from_orientation(orientation)
                )

        elif palace_type == "waveport":
            port_type = PortType.WAVEPORT
            geometry = PortGeometry.INPLANE  # Waveport geometry TBD
            direction = palace_direction_from_orientation(orientation)
            if layer_name in stack.layers:
                layer = stack.layers[layer_name]
                zmin = layer.zmin
                zmax = layer.zmax
        else:
            raise ValueError(f"Unknown port type: {palace_type}")

        palace_port = PalacePort(
            name=port.name,
            port_type=port_type,
            geometry=geometry,
            center=center,
            width=width,
            orientation=orientation,
            zmin=zmin,
            zmax=zmax,
            layer=layer_name,
            from_layer=from_layer,
            to_layer=to_layer,
            length=info.get("length"),
            generate_sheet=info.get("generate_sheet", True),
            sheet_gds_layer=info.get("sheet_gds_layer"),
            direction=direction,
            impedance=info.get("impedance", 50.0),
            resistance=info.get("resistance"),
            inductance=info.get("inductance"),
            capacitance=info.get("capacitance"),
            z_margin=info.get("z_margin", 0.0),
            lateral_margin=info.get("lateral_margin", 0.0),
            max_size=info.get("max_size", False),
            excited=info.get("excited", True),
            mode=info.get("mode", 1),
            offset=info.get("offset", 0.0),
        )
        palace_ports.append(palace_port)

    return palace_ports
