"""Palace configuration file generation.

This module handles generating Palace config.json and collecting mesh statistics.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gmsh

from gsim.palace.ports.config import PortType

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import (
        BoundaryModeConfig,
        DrivenConfig,
        EigenmodeConfig,
        ElectrostaticConfig,
        MagnetostaticConfig,
        NumericalConfig,
    )
    from gsim.palace.models.ports import TerminalConfig
    from gsim.palace.ports.config import PalacePort


def generate_palace_config(
    groups: dict,
    ports: list[PalacePort],
    port_info: list,
    stack: LayerStack,
    output_path: Path,
    model_name: str,
    fmax: float,
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    boundary_mode_config: BoundaryModeConfig | None = None,
    absorbing_boundary: bool = True,
    periodic_axis: str | None = None,
    hints: dict[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: list[TerminalConfig] | None = None,
    magnetostatic_config: MagnetostaticConfig | None = None,
    current_sources: list[Any] | None = None,
    palace_version: str = "0.16.0",
    validate_schema: bool = False,
    material_overlay: Any | None = None,
    postprocessing_config: Mapping[str, list[dict[str, Any]]] | None = None,
    boundary_postprocessing_config: Mapping[str, list[dict[str, Any]]] | None = None,
) -> Path:
    """Generate Palace config.json file.

    Args:
        groups: Physical group information from mesh generation
        ports: List of PalacePort objects
        port_info: Port metadata list
        stack: Layer stack for material properties
        output_path: Output directory path
        model_name: Base name for output files
        fmax: Maximum frequency (Hz) - used as fallback if driven_config not provided
        driven_config: Optional DrivenConfig for frequency sweep settings.
            When provided, material dispersion is evaluated at the center
            frequency ``(fmin + fmax) / 2`` of the sweep band.
        eigenmode_config: Optional EigenmodeConfig for eigenproblems settings
        absorbing_boundary: Whether to add absorbing (PML) boundary
        periodic_axis: Optional periodic axis identifier
        hints: Additional config hints merged into the JSON

    Returns:
        Path to the generated config.json
    """
    from gsim.palace.ports.config import PortGeometry

    if simulation_type not in (
        "driven",
        "eigenmode",
        "boundarymode",
        "electrostatic",
        "electrostatics",
        "magnetostatic",
    ):
        raise ValueError(f"Unsupported simulation type: {simulation_type}")

    # Use driven_config if provided, otherwise fall back to legacy parameters
    if driven_config is not None:
        solver_driven = driven_config.to_palace_config()
    else:
        # Legacy behavior - compute from fmax
        freq_step = fmax / 40e9
        solver_driven = {
            "Samples": [
                {
                    "Type": "Driven",
                    "MinFreq": 1.0,  # 1 GHz
                    "MaxFreq": fmax / 1e9,
                    "FreqStep": freq_step,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": 0.02,
        }

    if eigenmode_config is not None:
        solver_eigenmode = eigenmode_config.to_palace_config()
    else:
        # Legacy behavior - compute from fmax
        solver_eigenmode = (
            {
                "N": 10,
                "Tol": 1.0e-6,
                "Target": fmax,
            },
        )

    if boundary_mode_config is not None:
        solver_boundarymode = boundary_mode_config.to_palace_config()
    else:
        solver_boundarymode = {
            "Freq": fmax / 1e9,
            "N": 1,
            "Save": 0,
            "Tol": 1.0e-6,
            "Type": "Default",
        }

    solver_conf: dict[str, object]
    if numerical_config is not None:
        solver_conf = dict(numerical_config.to_solver_config())
    else:
        # Backward-compatible defaults for direct generate_palace_config() calls
        # that do not provide a NumericalConfig.
        solver_conf = {
            "Linear": {
                "Type": "Default",
                "KSPType": "GMRES",
                "Tol": 1e-6,
                "MaxIts": 1000,
            },
            "Order": 2,
            "Device": "CPU",
        }

    if simulation_type == "driven":
        solver_conf["Driven"] = solver_driven
    elif simulation_type == "eigenmode":
        solver_conf["Eigenmode"] = solver_eigenmode
    elif simulation_type in ("electrostatic", "electrostatics"):
        if electrostatic_config is not None:
            solver_conf["Electrostatic"] = electrostatic_config.to_palace_config()
        else:
            solver_conf["Electrostatic"] = {"Save": 0}
    elif simulation_type == "magnetostatic":
        solver_conf["Magnetostatic"] = (
            magnetostatic_config.to_palace_config()
            if magnetostatic_config is not None
            else {"Save": 0}
        )
    elif simulation_type == "boundarymode":
        solver_conf["BoundaryMode"] = solver_boundarymode
    else:
        raise NotImplementedError

    problem_type_map = {
        "driven": "Driven",
        "eigenmode": "Eigenmode",
        "boundarymode": "BoundaryMode",
        "electrostatic": "Electrostatic",
        "electrostatics": "Electrostatic",
        "magnetostatic": "Magnetostatic",
    }

    model_l0 = 1e-6

    problem_config: dict[str, object] = {
        "Type": problem_type_map[simulation_type],
        "Verbose": 3,
        "Output": "results/palace",
    }
    refinement: dict[str, Any] = {
        "UniformLevels": 0,
        "Tol": 1e-2,
        "MaxIts": 0,
    }
    config: dict[str, object] = {
        "Problem": problem_config,
        "Model": {
            "Mesh": f"{model_name}.msh",
            "L0": model_l0,  # um
            "Refinement": refinement,
        },
        "Solver": solver_conf,
    }

    # Build domains section
    # Evaluate dispersion models at the center frequency of the sweep band
    stack_materials = stack.materials
    material_resolution: dict[str, Any] | None = None
    material_frequency = (
        driven_config.center_frequency if driven_config is not None else fmax
    )
    if driven_config is not None or material_overlay is not None:
        from gsim.palace.materials import resolve_palace_materials_with_report

        stack_materials, material_resolution = resolve_palace_materials_with_report(
            stack.materials,
            material_frequency,
            material_overlay=material_overlay,
        )

    # Support material keys with different capitalization conventions
    # between stack layers and material dictionaries.
    _materials_by_lower = {
        str(name).lower(): props for name, props in stack_materials.items()
    }

    def _lookup_material(name: str) -> dict[str, object]:
        props = stack_materials.get(name)
        if isinstance(props, dict):
            return props
        fallback = _materials_by_lower.get(str(name).lower())
        return fallback if isinstance(fallback, dict) else {}

    is_electrostatic = simulation_type in ("electrostatic", "electrostatics")
    materials: list[dict[str, object]] = []
    for volume_name, info in groups["volumes"].items():
        material_name = volume_name
        stack_material_name = material_name
        is_via = info.get("is_via", False)
        is_shaped_dielectric = info.get("is_shaped_dielectric", False)

        if is_via or is_shaped_dielectric:
            layer = stack.layers.get(material_name)
            if layer is None:
                # Native 2D material domains (e.g. sio2/sin) may not map to
                # a stack layer name; resolve them directly as material names.
                mat_props = _lookup_material(material_name)
            else:
                stack_material_name = layer.material
                mat_props = _lookup_material(stack_material_name)
        elif info.get("stack_layer") is not None:
            stack_layer_name = str(info["stack_layer"])
            layer = stack.layers.get(stack_layer_name)
            if layer is None:
                raise ValueError(
                    f"Mesh volume {volume_name!r} references missing stack layer "
                    f"{stack_layer_name!r}."
                )
            stack_material_name = str(info.get("material") or layer.material)
            mat_props = _lookup_material(stack_material_name)
            if not mat_props:
                raise ValueError(
                    f"Activated region {volume_name!r} uses material "
                    f"{stack_material_name!r}, which is not in the layer stack "
                    "or material overlay."
                )
        else:
            mat_props = _lookup_material(material_name)

        mat_entry: dict[str, object] = {"Attributes": [info["phys_group"]]}

        if volume_name in {"airbox", "air"}:
            mat_entry["Permittivity"] = 1.0
            mat_entry["LossTan"] = 0.0
        elif is_via:
            sigma = mat_props.get("conductivity", 0.0)
            mat_entry["Permittivity"] = 1.0
            if not is_electrostatic and isinstance(sigma, (int, float)) and sigma > 0:
                mat_entry["Conductivity"] = sigma
        elif is_shaped_dielectric:
            perm = mat_props.get("permittivity", 1.0)
            mat_entry["Permittivity"] = perm

            lt = mat_props.get("loss_tangent", 0.0)
            if isinstance(lt, list) or (isinstance(lt, (int, float)) and lt > 0):
                mat_entry["LossTan"] = lt
            else:
                mat_entry["LossTan"] = 0.0

            if "permeability" in mat_props:
                mat_entry["Permeability"] = mat_props["permeability"]

            if "material_axes" in mat_props:
                mat_entry["MaterialAxes"] = mat_props["material_axes"]
        else:
            perm = mat_props.get("permittivity", 1.0)
            mat_entry["Permittivity"] = perm

            if "permeability" in mat_props:
                mat_entry["Permeability"] = mat_props["permeability"]

            sigma = mat_props.get("conductivity", 0.0)
            lt = mat_props.get("loss_tangent", 0.0)
            if not is_electrostatic and (
                (isinstance(sigma, (int, float)) and sigma > 0)
                or isinstance(sigma, list)
            ):
                mat_entry["Conductivity"] = sigma
            elif isinstance(lt, list) or (isinstance(lt, (int, float)) and lt > 0):
                mat_entry["LossTan"] = lt
            else:
                mat_entry["LossTan"] = 0.0

            if "material_axes" in mat_props:
                mat_entry["MaterialAxes"] = mat_props["material_axes"]

        materials.append(mat_entry)

    config["Domains"] = {
        "Materials": materials,
        "Postprocessing": {"Energy": [], "Probe": []},
    }

    # Build boundaries section
    conductors: list[dict[str, object]] = []

    for name, info in groups["conductor_surfaces"].items():
        # Extract layer name from "layer_xy" / "layer_z" (3D) or use
        # the raw key directly for native 2D conductor boundaries.
        layer_name = name.rsplit("_", 1)[0] if name.endswith(("_xy", "_z")) else name
        layer = stack.layers.get(layer_name)
        if layer:
            mat_props = stack_materials.get(layer.material, {})
            conductors.append(
                {
                    "Attributes": [info["phys_group"]],
                    "Conductivity": mat_props.get("conductivity", 5.8e7),
                    "Thickness": layer.zmax - layer.zmin,
                }
            )

    # Handle PEC surfaces (planar conductors + PEC blocks)
    pec_attrs: list[int] = [
        info["phys_group"] for info in groups.get("pec_surfaces", {}).values()
    ]
    boundaries: dict[str, object]

    if (
        is_electrostatic
        and terminals
        and _has_sgb_structured_conductor_surfaces(groups)
    ):
        boundaries = _sgb_electrostatic_boundaries(
            groups,
            terminals,
            electrostatic_config=electrostatic_config,
        )
    elif is_electrostatic and terminals:
        if any(terminal.net_id is not None for terminal in terminals):
            raise ValueError(
                "Semantic-net terminals require structured SGB Route A/B mesh groups."
            )
        if (
            electrostatic_config is not None
            and electrostatic_config.exterior_boundary_policy != "none"
        ):
            raise ValueError(
                "Exterior-domain Ground requires structured SGB Route A/B mesh groups."
            )
        terminal_layer_names: set[str] = {
            terminal.layer for terminal in terminals if terminal.layer is not None
        }
        via_boundary = groups.get("via_boundary_surfaces", {})

        def _via_touches(via_name: str, conductor_layer_name: str) -> bool:
            """Z-range overlap (or touching) between a via and a conductor."""
            via = stack.layers.get(via_name)
            cond = stack.layers.get(conductor_layer_name)
            if via is None or cond is None:
                return False
            return via.zmin <= cond.zmax and via.zmax >= cond.zmin

        terminal_entries: list[dict[str, object]] = []
        assigned_pgs: set[int] = set()

        # Track which vias were attached to a terminal so we don't also
        # send their surfaces to ground.
        vias_on_terminal: set[str] = set()

        pec_surfaces = groups.get("pec_surfaces", {})

        for idx, terminal in enumerate(terminals, start=1):
            if terminal.layer is None:
                raise ValueError("Native electrostatic terminals require a layer.")
            attrs: list[int] = []
            # Thick conductor shells (named "<layer>_xy" / "<layer>_z")
            for surf_name, surf_info in groups["conductor_surfaces"].items():
                surf_layer = surf_name.rsplit("_", 1)[0]
                if surf_layer == terminal.layer:
                    attrs.append(surf_info["phys_group"])
            # Planar (thin) conductor surfaces (keyed by layer name)
            if terminal.layer in pec_surfaces:
                attrs.append(pec_surfaces[terminal.layer]["phys_group"])
            # Vias touching this terminal's layer
            for via_name, via_pgs in via_boundary.items():
                if _via_touches(via_name, terminal.layer):
                    attrs.extend(via_pgs)
                    vias_on_terminal.add(via_name)

            assigned_pgs.update(attrs)
            terminal_entries.append(
                {
                    "Index": idx,
                    "Attributes": sorted(attrs),
                }
            )

        # Anything that wasn't assigned to a terminal becomes Ground.
        ground_attrs: list[int] = []
        for surf_info in groups["conductor_surfaces"].values():
            pg = surf_info["phys_group"]
            if pg not in assigned_pgs:
                ground_attrs.append(pg)
        for surf_info in pec_surfaces.values():
            pg = surf_info["phys_group"]
            if pg not in assigned_pgs:
                ground_attrs.append(pg)

        # Vias that touch a non-terminal conductor -> tie to ground so they
        # don't float (Palace's solver has no current-flow through volumes).
        for via_name, via_pgs in via_boundary.items():
            if via_name in vias_on_terminal:
                continue
            for cond_name in stack.layers or {}:
                if cond_name in terminal_layer_names:
                    continue
                cond = stack.layers.get(cond_name)
                if cond is None or cond.layer_type != "conductor":
                    continue
                if _via_touches(via_name, cond_name):
                    ground_attrs.extend(via_pgs)
                    break

        boundaries: dict[str, object] = {
            "Terminal": terminal_entries,
        }
        if ground_attrs:
            boundaries["Ground"] = {"Attributes": sorted(set(ground_attrs))}

    elif simulation_type == "magnetostatic":
        if not current_sources:
            raise ValueError(
                "Magnetostatic config generation requires at least one current source."
            )
        source_entries: list[dict[str, object]] = []
        flux_entries: list[dict[str, object]] = []
        for index, source in enumerate(current_sources, start=1):
            source_entry, attrs = _magnetostatic_source_entry(
                groups=groups,
                source=source,
                index=index,
            )
            source_entries.append(source_entry)
            flux_entries.append(
                {
                    "Index": index,
                    "Attributes": attrs,
                    "Type": "Magnetic",
                    "TwoSided": False,
                }
            )
        boundaries = {"SurfaceCurrent": source_entries}
        outer_attrs = _outer_boundary_attributes(groups)
        if outer_attrs:
            boundaries["PMC"] = {"Attributes": outer_attrs}
        if flux_entries:
            boundaries["Postprocessing"] = {"SurfaceFlux": flux_entries}

    elif simulation_type == "boundarymode":
        boundaries = {}
        if conductors:
            boundaries["Conductivity"] = conductors
        if pec_attrs:
            boundaries["PEC"] = {"Attributes": sorted(set(pec_attrs))}

    else:
        lumped_ports: list[dict[str, object]] = []
        wave_ports: list[dict[str, object]] = []
        port_idx = 1
        # Passive reactive ports are appended after all primary ports are assigned
        # indices so that their synthetic indices never clash.  We collect them here
        # and append them to lumped_ports once the primary loop finishes.
        passive_reactive_ports: list[dict[str, object]] = []

        for port in ports:
            if simulation_type != "driven":
                port.excited = False
            port_key = f"P{port_idx}"
            if port_key in groups["port_surfaces"]:
                port_group = groups["port_surfaces"][port_key]

                if port.sheet_layer is not None:
                    _validate_layout_sheet_port_group(port, port_group, port_key)
                    entry: dict[str, object] = {
                        "Index": port_idx,
                        "Direction": list(port_group["direction"]),
                        "Excitation": (
                            False
                            if simulation_type == "eigenmode"
                            else (port_idx if port.excited else False)
                        ),
                        "Attributes": [port_group["phys_group"]],
                    }
                    if port.resistance is not None:
                        entry["R"] = port.resistance
                    if port.inductance is not None and port.inductance > 0:
                        entry["L"] = port.inductance
                    if port.capacitance is not None and port.capacitance > 0:
                        entry["C"] = port.capacitance
                    lumped_ports.append(entry)
                elif port.multi_element:
                    # Multi-element port (CPW)
                    if port_group.get("type") == "cpw":
                        elements = [
                            {
                                "Attributes": [elem["phys_group"]],
                                "Direction": elem["direction"],
                            }
                            for elem in port_group["elements"]
                        ]
                        lumped_ports.append(
                            {
                                "Index": port_idx,
                                "R": port.impedance,
                                "Excitation": port_idx if port.excited else False,
                                "Elements": elements,
                            }
                        )
                else:
                    # Single-element port
                    if port.port_type == PortType.LUMPED:
                        direction = (
                            "Z"
                            if port.geometry == PortGeometry.VIA
                            else port.direction.upper()
                        )

                        has_reactive = (
                            port.resistance is not None
                            or (port.inductance is not None and port.inductance > 0)
                            or (port.capacitance is not None and port.capacitance > 0)
                        )

                        if simulation_type == "driven" and has_reactive:
                            # Driven simulations forbid L/C on excited ports.
                            # Emit the excited port with only R=impedance, then a
                            # separate passive (Active: false) lumped port carrying
                            # the reactive parameters on the same boundary surface.
                            # Palace allows shared attributes when Active is false.
                            port_entry: dict[str, object] = {
                                "Index": port_idx,
                                "R": port.impedance,
                                "Direction": direction,
                                "Excitation": port_idx if port.excited else False,
                                "Attributes": [port_group["phys_group"]],
                            }
                            lumped_ports.append(port_entry)

                            reactive_entry: dict[str, object] = {
                                # Placeholder index - will be replaced after the
                                # primary loop assigns all port_idx values.
                                "Index": None,
                                "Direction": direction,
                                "Attributes": [port_group["phys_group"]],
                                "Active": False,
                            }
                            if port.resistance is not None:
                                reactive_entry["R"] = port.resistance
                            if port.inductance is not None and port.inductance > 0:
                                reactive_entry["L"] = port.inductance
                            if port.capacitance is not None and port.capacitance > 0:
                                reactive_entry["C"] = port.capacitance
                            passive_reactive_ports.append(reactive_entry)
                        else:
                            # Eigenmode (or non-excited driven port): R/L/C on the
                            # same port entry is supported.
                            eigenmode_entry: dict[str, object] = {
                                "Index": port_idx,
                                "Direction": direction,
                                "Excitation": port_idx if port.excited else False,
                                "Attributes": [port_group["phys_group"]],
                            }
                            if port.impedance:
                                eigenmode_entry["R"] = port.impedance
                            if port.resistance is not None:
                                eigenmode_entry["R"] = port.resistance
                            if port.inductance is not None and port.inductance > 0:
                                eigenmode_entry["L"] = port.inductance
                            if port.capacitance is not None and port.capacitance > 0:
                                eigenmode_entry["C"] = port.capacitance
                            lumped_ports.append(eigenmode_entry)

                    elif port.port_type == PortType.WAVEPORT:
                        wave_ports.append(
                            {
                                "Index": port_idx,
                                "Mode": port.mode,
                                "Offset": port.offset,
                                "Excitation": port_idx if port.excited else False,
                                "Attributes": [port_group["phys_group"]],
                            }
                        )
            elif port.sheet_layer is not None:
                raise ValueError(
                    f"layout_sheet port '{port.name}' has no SGB port surface "
                    f"{port_key}."
                )
            port_idx += 1

        # Assign unique indices to passive reactive ports now that all primary
        # indices are consumed (port_idx is one past the last primary index).
        synthetic_idx = port_idx
        for entry in passive_reactive_ports:
            entry["Index"] = synthetic_idx
            synthetic_idx += 1
        lumped_ports.extend(passive_reactive_ports)

        boundaries: dict[str, object] = {
            "Conductivity": conductors,
            "LumpedPort": lumped_ports,
            "WavePort": wave_ports,
        }

        # Add PEC boundaries if any exist
        if pec_attrs:
            boundaries["PEC"] = {"Attributes": pec_attrs}

    if "absorbing" in groups["boundary_surfaces"] and absorbing_boundary:
        absorbing_pg = groups["boundary_surfaces"]["absorbing"]["phys_group"]
        # phys_group may be a list (multiple __None groups) or a single int
        attrs = absorbing_pg if isinstance(absorbing_pg, list) else [absorbing_pg]
        boundaries["Absorbing"] = {
            "Attributes": attrs,
            "Order": 2,
        }

    if (
        simulation_type == "eigenmode"
        and eigenmode_config is not None
        and eigenmode_config.floquet
    ):
        axis = (periodic_axis or "").lower()
        if axis not in {"x", "y"}:
            raise ValueError(
                "Floquet eigenmode requires a periodic axis set in mesh(). "
                "Use mesh(periodic_axis='x') or mesh(periodic_axis='y')."
            )

        donor_info = groups["boundary_surfaces"].get("periodic_donor")
        receiver_info = groups["boundary_surfaces"].get("periodic_receiver")
        if donor_info is None or receiver_info is None:
            raise ValueError(
                "Floquet enabled but periodic donor/receiver boundaries were not "
                "found in the generated mesh."
            )

        donor_pg = donor_info.get("phys_group")
        receiver_pg = receiver_info.get("phys_group")
        periodic_donor_attrs = donor_pg if isinstance(donor_pg, list) else [donor_pg]
        periodic_receiver_attrs = (
            receiver_pg if isinstance(receiver_pg, list) else [receiver_pg]
        )

        if not periodic_donor_attrs or not periodic_receiver_attrs:
            raise ValueError("Floquet periodic boundary attributes are empty.")

        axis_lit: Literal["x", "y"] = "x" if axis == "x" else "y"

        floquet_vector = eigenmode_config.compute_floquet_wave_vector(
            periodic_axis=axis_lit,
            l0=model_l0,
        )

        boundaries["Periodic"] = {
            "FloquetWaveVector": floquet_vector,
            "BoundaryPairs": [
                {
                    "DonorAttributes": sorted(periodic_donor_attrs),
                    "ReceiverAttributes": sorted(periodic_receiver_attrs),
                }
            ],
        }

    config["Boundaries"] = boundaries

    if postprocessing_config:
        domains = config["Domains"]
        if not isinstance(domains, dict):
            raise TypeError("Internal Palace domains configuration must be a mapping.")
        domain_postprocessing = domains["Postprocessing"]
        if not isinstance(domain_postprocessing, dict):
            raise TypeError("Internal Palace domain postprocessing must be a mapping.")
        domain_postprocessing.update(postprocessing_config)
    if boundary_postprocessing_config:
        boundary_postprocessing = boundaries.setdefault("Postprocessing", {})
        if not isinstance(boundary_postprocessing, dict):
            raise TypeError(
                "Internal Palace boundary postprocessing must be a mapping."
            )
        boundary_postprocessing.update(deepcopy(boundary_postprocessing_config))
        interface_resolution_rows = _resolve_boundary_dielectric_interfaces(
            boundary_postprocessing,
            material_definitions=stack.materials,
            material_frequency_hz=material_frequency,
            material_overlay=material_overlay,
        )
        if interface_resolution_rows:
            if material_resolution is None:
                material_resolution = {
                    "schema_version": 1,
                    "evaluation_frequency_hz": float(material_frequency),
                    "materials": [],
                }
            material_resolution["interfaces"] = interface_resolution_rows

    # Hints can tune solver details, but typed mesh/problem owners remain
    # authoritative: they must not be replaced through an unstructured mapping.
    if hints:
        _reject_protected_config_hints(hints)
        _deep_merge_config(config, hints)

    _reject_private_config_keys(config)

    # Write config file
    config_path = output_path / "config.json"
    if validate_schema:
        from gsim.palace.config_validation import validate_palace_config

        validate_palace_config(config, palace_version=palace_version)
    with config_path.open("w") as f:
        json.dump(config, f, indent=4)

    if material_resolution is not None:
        material_resolution["emitted_material_rows"] = [
            dict(entry) for entry in materials
        ]
        from gsim.palace.run_folder import prepare_palace_run_folder

        with prepare_palace_run_folder(output_path).material_resolution_path.open(
            "w"
        ) as f:
            json.dump(material_resolution, f, indent=4)

    # Port metadata is a gsim sidecar, not a Palace input at the run root.
    from gsim.palace.run_folder import prepare_palace_run_folder

    port_info_path = prepare_palace_run_folder(output_path).port_information_path
    port_info_struct = {"ports": port_info, "unit": 1e-6, "name": model_name}
    with port_info_path.open("w") as f:
        json.dump(port_info_struct, f, indent=4)

    return config_path


def _validate_layout_sheet_port_group(
    port: PalacePort, group: Mapping, key: str
) -> None:
    """Require an exact SGB sheet direction matching the gdsfactory port."""
    if group.get("type") != "lumped_sheet":
        raise ValueError(f"layout_sheet port '{port.name}' has invalid group {key}.")
    attribute = group.get("physical_attribute")
    expected_source_layer = (
        None
        if port.sheet_layer is None
        else f"{port.sheet_layer[0]}/{port.sheet_layer[1]}"
    )
    if (
        not isinstance(attribute, Mapping)
        or attribute.get("port_name") != port.name
        or attribute.get("port_index") != int(key.removeprefix("P"))
        or attribute.get("source_layer") != expected_source_layer
        or attribute.get("target_layer") != port.layer
    ):
        raise ValueError(
            f"layout_sheet port '{port.name}' does not match SGB source ownership."
        )
    direction = group.get("direction")
    if (
        isinstance(direction, (str, bytes))
        or not isinstance(direction, (tuple, list))
        or len(direction) != 3
    ):
        raise ValueError(f"layout_sheet port '{port.name}' has no numeric direction.")
    vector = tuple(float(value) for value in direction)
    length = math.hypot(vector[0], vector[1])
    if (
        not all(math.isfinite(value) for value in vector)
        or not math.isclose(vector[2], 0.0, abs_tol=1e-12)
        or not math.isclose(length, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(
            f"layout_sheet port '{port.name}' direction must be finite normalized XY."
        )
    angle = math.radians(port.orientation)
    expected = (math.cos(angle), math.sin(angle))
    if not math.isclose(vector[0], expected[0], abs_tol=1e-12) or not math.isclose(
        vector[1], expected[1], abs_tol=1e-12
    ):
        raise ValueError(
            f"layout_sheet port '{port.name}' direction does not match orientation."
        )


def _resolve_boundary_dielectric_interfaces(
    boundary_postprocessing: dict[str, Any],
    *,
    material_definitions: Mapping[str, Any],
    material_frequency_hz: float,
    material_overlay: Any | None,
) -> list[dict[str, Any]]:
    """Resolve internal dielectric material tokens into Palace solver rows."""
    dielectric_rows = boundary_postprocessing.get("Dielectric")
    if not isinstance(dielectric_rows, list):
        return []

    resolution_rows: list[dict[str, Any]] = []
    for row_index, interface in enumerate(dielectric_rows, start=1):
        if not isinstance(interface, dict):
            raise TypeError("Boundary Postprocessing.Dielectric rows must be mappings.")
        material_name = interface.pop("_MaterialName", None)
        if material_name is None:
            continue
        if not isinstance(material_name, str) or not material_name:
            raise TypeError(
                "Dielectric interface _MaterialName must be a nonempty string."
            )

        resolved_material, resolution = _resolve_single_interface_material(
            material_name,
            material_definitions=material_definitions,
            material_frequency_hz=material_frequency_hz,
            material_overlay=material_overlay,
        )
        permittivity = resolved_material.get("permittivity")
        if permittivity is None:
            raise ValueError(
                "Dielectric interface material resolution did not provide "
                f"Permittivity for {material_name!r}."
            )
        interface["Permittivity"] = permittivity
        if "loss_tangent" in resolved_material:
            interface["LossTan"] = resolved_material["loss_tangent"]
        elif "LossTan" not in interface:
            interface["LossTan"] = 0.0

        resolution_rows.append(
            {
                "interface_row_index": row_index,
                "surface_index": interface.get("Index"),
                "surface_attributes": list(interface.get("Attributes", ())),
                "interface_type": interface.get("Type"),
                "interface_material_name": material_name,
                "palace_interface": dict(interface),
                **resolution,
            }
        )
    return resolution_rows


def _resolve_single_interface_material(
    material_name: str,
    *,
    material_definitions: Mapping[str, Any],
    material_frequency_hz: float,
    material_overlay: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one interface material through the Palace material authority."""
    from gsim.palace.materials import resolve_palace_materials_with_report

    source = material_definitions.get(material_name, {})
    if not isinstance(source, Mapping):
        raise TypeError(f"Material {material_name!r} must resolve from a mapping.")
    resolved, report = resolve_palace_materials_with_report(
        {material_name: dict(source)},
        material_frequency_hz,
        material_overlay=material_overlay,
    )
    rows = report.get("materials", [])
    resolution = dict(rows[0]) if isinstance(rows, list) and rows else {}
    return dict(resolved[material_name]), resolution


def _reject_private_config_keys(value: Any, *, path: tuple[str, ...] = ()) -> None:
    """Fail before serialization if an internal-only key reaches Palace JSON."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            current_path = (*path, key_text)
            if key_text.startswith("_"):
                raise ValueError(
                    "Internal Palace configuration key reached solver JSON: "
                    + ".".join(current_path)
                )
            _reject_private_config_keys(child, path=current_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_config_keys(child, path=(*path, str(index)))


def _deep_merge_config(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Recursively merge non-owned caller configuration hints."""
    for key, value in updates.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _deep_merge_config(existing, value)
        else:
            target[key] = deepcopy(value)


_PROTECTED_HINT_PATHS: tuple[tuple[str, ...], ...] = (
    ("Domains",),
    ("Boundaries",),
    ("Problem", "Type"),
    ("Problem", "Output"),
    ("Model", "Mesh"),
)


def _reject_protected_config_hints(
    hints: Mapping[str, Any], *, path: tuple[str, ...] = ()
) -> None:
    """Fail closed when hints attempt to replace typed Palace owners."""
    for key, value in hints.items():
        current_path = (*path, str(key))
        protected_owner = current_path in _PROTECTED_HINT_PATHS
        protected_descendant = any(
            protected_path[: len(current_path)] == current_path
            for protected_path in _PROTECTED_HINT_PATHS
        )
        if protected_owner or (protected_descendant and not isinstance(value, Mapping)):
            raise ValueError(
                "Palace config hints cannot overwrite "
                f"{'.'.join(current_path)}. Use the typed owner API instead."
            )
        if isinstance(value, Mapping):
            _reject_protected_config_hints(value, path=current_path)


def _has_sgb_structured_conductor_surfaces(groups: Mapping[str, Any]) -> bool:
    """Return whether groups carry the optional structured SGB Route A/B contract."""
    return any(
        isinstance(info, Mapping)
        and info.get("sgb_record") == "final_physical_group"
        and info.get("source") == "volume_interface"
        and info.get("solver_use") == "solver_active"
        and info.get("interface_type") in {"MA", "MS"}
        for info in groups.get("boundary_surfaces", {}).values()
    )


def _matches_layer_selector(name: str, info: Mapping[str, Any], selector: Any) -> bool:
    """Match a native conductor island to an explicit typed selector."""
    layer = info.get("layer", name.rsplit("_", 1)[0])
    if layer != selector.layer and name != selector.layer:
        return False
    center = getattr(selector, "center", None)
    return center is None or _bbox_contains_center(info.get("bbox"), center)


def _bbox_contains_center(bbox: Any, center: tuple[float, float]) -> bool:
    """Return whether a selector point lies in a valid generated XY bounding box."""
    if (
        not isinstance(bbox, list | tuple)
        or len(bbox) != 6
        or not all(isinstance(value, int | float) for value in bbox)
    ):
        return False
    x, y = center
    tolerance = 1e-6
    return (
        float(bbox[0]) - tolerance <= x <= float(bbox[3]) + tolerance
        and float(bbox[1]) - tolerance <= y <= float(bbox[4]) + tolerance
    )


def _magnetostatic_selector_attributes(
    groups: Mapping[str, Any],
    selector: Any,
) -> list[int]:
    """Resolve one current-source selector to complete matching island attributes."""
    if not selector.layer:
        raise ValueError("Current source selector requires a layer.")
    attributes: set[int] = set()
    for group_name in ("conductor_surfaces", "pec_surfaces"):
        for name, info in groups.get(group_name, {}).items():
            if (
                isinstance(info, Mapping)
                and not info.get("postprocessing_only")
                and _matches_layer_selector(name, info, selector)
            ):
                attributes.update(_physical_group_values(info.get("phys_group")))
    return sorted(attributes)


def _current_source_direction(value: Any) -> str | list[float]:
    """Serialize a pre-validated current direction for Palace JSON."""
    return value if isinstance(value, str) else [float(item) for item in value]


def _magnetostatic_source_entry(
    *,
    groups: Mapping[str, Any],
    source: Any,
    index: int,
) -> tuple[dict[str, object], list[int]]:
    """Lower one typed source, including its optional disjoint elements."""
    if source.elements:
        elements: list[dict[str, object]] = []
        selected: set[int] = set()
        for element_index, element in enumerate(source.elements, start=1):
            attributes = _magnetostatic_selector_attributes(groups, element)
            if not attributes:
                raise ValueError(
                    f"Current source {source.name!r} element {element_index} "
                    f"on layer {element.layer!r} did not match any conductor."
                )
            overlap = selected.intersection(attributes)
            if overlap:
                raise ValueError(
                    f"Current source {source.name!r} element {element_index} "
                    f"matched attributes already selected by an earlier element: "
                    f"{sorted(overlap)}."
                )
            selected.update(attributes)
            entry: dict[str, object] = {
                "Attributes": attributes,
                "Direction": _current_source_direction(element.direction),
            }
            if element.coordinate_system is not None:
                entry["CoordinateSystem"] = element.coordinate_system
            elements.append(entry)
        return {"Index": index, "Elements": elements}, sorted(selected)

    attributes = _magnetostatic_selector_attributes(groups, source)
    if not attributes:
        raise ValueError(
            f"Current source {source.name!r} on layer {source.layer!r} "
            "did not match any conductor."
        )
    entry = {
        "Index": index,
        "Attributes": attributes,
        "Direction": _current_source_direction(source.direction),
    }
    if source.coordinate_system is not None:
        entry["CoordinateSystem"] = source.coordinate_system
    return entry, attributes


def _outer_boundary_attributes(groups: Mapping[str, Any]) -> list[int]:
    """Return generated exterior attributes for magnetostatic PMC boundaries."""
    absorbing = groups.get("boundary_surfaces", {}).get("absorbing")
    if not isinstance(absorbing, Mapping):
        return []
    return list(_physical_group_values(absorbing.get("phys_group")))


def _physical_group_values(value: Any) -> tuple[int, ...]:
    """Return one or more integer Palace attributes from a group field."""
    raw = value if isinstance(value, list | tuple | set) else (value,)
    return tuple(
        int(item)
        for item in raw
        if isinstance(item, int) and not isinstance(item, bool)
    )


def _selector_terminal_net_id(terminal: Any) -> str:
    """Match the deterministic selector net emitted by SGB lowering."""
    label = terminal.physical_label or terminal.name
    if not terminal.layer or not isinstance(label, str) or not label.strip():
        raise ValueError("SGB selector terminal needs a non-empty layer and label.")
    return re.sub(r"[^A-Za-z0-9_@]+", "_", f"{terminal.layer}@{label}".strip()).strip(
        "_"
    )


def _sgb_electrostatic_boundaries(
    groups: Mapping[str, Any],
    terminals: list[Any],
    *,
    electrostatic_config: Any | None,
) -> dict[str, object]:
    """Assign complete structured conductor components atomically to terminals."""
    component_attrs: dict[str, set[int]] = {}
    component_nets: dict[str, set[str | None]] = {}
    attribute_components: dict[int, str] = {}
    for info in groups.get("boundary_surfaces", {}).values():
        if (
            not isinstance(info, Mapping)
            or info.get("sgb_record") != "final_physical_group"
            or info.get("source") != "volume_interface"
            or info.get("solver_use") != "solver_active"
            or info.get("interface_type") not in {"MA", "MS"}
        ):
            continue
        component_id = info.get("conductor_component_id")
        net_id = info.get("net_id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError(
                "SGB solver-active conductor surface lacks conductor_component_id."
            )
        if net_id is not None and (not isinstance(net_id, str) or not net_id):
            raise ValueError(
                f"SGB conductor component {component_id!r} has invalid net_id."
            )
        attrs = _physical_group_values(info.get("phys_group"))
        if not attrs:
            raise ValueError(
                f"SGB conductor component {component_id!r} has no attributes."
            )
        component_attrs.setdefault(component_id, set()).update(attrs)
        component_nets.setdefault(component_id, set()).add(net_id)
        for attr in attrs:
            previous = attribute_components.setdefault(attr, component_id)
            if previous != component_id:
                raise ValueError(
                    f"SGB attribute {attr} belongs to multiple conductor components."
                )
    if not component_attrs:
        raise ValueError("SGB electrostatic configuration has no conductor components.")

    component_net: dict[str, str | None] = {}
    for component_id, nets in component_nets.items():
        if len(nets) != 1:
            raise ValueError(
                f"SGB conductor component {component_id!r} has conflicting "
                "terminal nets."
            )
        component_net[component_id] = next(iter(nets))

    terminal_nets: set[str] = set()
    assigned_components: set[str] = set()
    terminal_entries: list[dict[str, object]] = []
    for index, terminal in enumerate(terminals, start=1):
        net_id = terminal.net_id or _selector_terminal_net_id(terminal)
        if net_id in terminal_nets:
            raise ValueError(f"SGB terminal net {net_id!r} appears more than once.")
        terminal_nets.add(net_id)
        matching_components = sorted(
            component_id
            for component_id, value in component_net.items()
            if value == net_id
        )
        if not matching_components:
            raise ValueError(f"SGB terminal {terminal.name!r} has no exact net match.")
        if assigned_components.intersection(matching_components):
            raise ValueError(
                f"SGB terminal {terminal.name!r} splits a conductor component."
            )
        assigned_components.update(matching_components)
        terminal_entries.append(
            {
                "Index": index,
                "Attributes": sorted(
                    attr
                    for component_id in matching_components
                    for attr in component_attrs[component_id]
                ),
            }
        )

    unassigned = set(component_attrs).difference(assigned_components)
    policy = (
        electrostatic_config.unassigned_conductor_policy
        if electrostatic_config is not None
        else "ground"
    )
    if unassigned and policy == "error":
        raise ValueError(
            "Unassigned SGB conductor components: " + ", ".join(sorted(unassigned))
        )

    ground_attrs = {
        attr for component_id in unassigned for attr in component_attrs[component_id]
    }
    exterior_policy = (
        electrostatic_config.exterior_boundary_policy
        if electrostatic_config is not None
        else "none"
    )
    if exterior_policy == "ground":
        exterior_attrs: set[int] = set()
        for info in groups.get("boundary_surfaces", {}).values():
            if (
                isinstance(info, Mapping)
                and info.get("sgb_record") == "final_physical_group"
                and info.get("source") == "domain_boundary"
                and info.get("solver_use") == "solver_active"
            ):
                exterior_attrs.update(_physical_group_values(info.get("phys_group")))
        if not exterior_attrs:
            raise ValueError("SGB exterior Boundary Ground has no attributes.")
        ground_attrs.update(exterior_attrs)

    boundaries: dict[str, object] = {"Terminal": terminal_entries}
    if ground_attrs:
        boundaries["Ground"] = {"Attributes": sorted(ground_attrs)}
    return boundaries


def collect_mesh_stats() -> dict:
    """Collect mesh statistics from gmsh after mesh generation.

    Must be called while gmsh is initialized and the mesh is generated.

    Returns:
        Dict with mesh statistics including:
        - bbox: Bounding box coordinates
        - nodes: Number of nodes
        - elements: Total element count
        - tetrahedra: Tet count
        - quality: Shape quality metrics (gamma)
        - sicn: Signed Inverse Condition Number
        - edge_length: Min/max edge lengths
        - groups: Physical group info
    """
    stats = {}

    # Get bounding box
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        stats["bbox"] = {
            "xmin": xmin,
            "ymin": ymin,
            "zmin": zmin,
            "xmax": xmax,
            "ymax": ymax,
            "zmax": zmax,
        }
    except Exception:
        pass

    # Get node count
    try:
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats["nodes"] = len(node_tags)
    except Exception:
        pass

    # Get element counts and collect tet tags for quality
    tet_tags = []
    try:
        element_types, element_tags, _ = gmsh.model.mesh.getElements()
        total_elements = sum(len(tags) for tags in element_tags)
        stats["elements"] = total_elements

        # Count tetrahedra (type 4) and save tags
        for etype, tags in zip(element_types, element_tags, strict=False):
            if etype == 4:  # 4-node tetrahedron
                stats["tetrahedra"] = len(tags)
                tet_tags = list(tags)
    except Exception:
        pass

    # Get mesh quality for tetrahedra
    if tet_tags:
        # Gamma: inscribed/circumscribed radius ratio (shape quality)
        try:
            qualities = gmsh.model.mesh.getElementQualities(tet_tags, "gamma")
            if len(qualities) > 0:
                stats["quality"] = {
                    "min": round(min(qualities), 3),
                    "max": round(max(qualities), 3),
                    "mean": round(sum(qualities) / len(qualities), 3),
                }
        except Exception:
            pass

        # SICN: Signed Inverse Condition Number (negative = invalid element)
        try:
            sicn = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
            if len(sicn) > 0:
                sicn_min = min(sicn)
                invalid_count = sum(1 for s in sicn if s < 0)
                stats["sicn"] = {
                    "min": round(sicn_min, 3),
                    "mean": round(sum(sicn) / len(sicn), 3),
                    "invalid": invalid_count,
                }
        except Exception:
            pass

        # Edge lengths
        try:
            min_edges = gmsh.model.mesh.getElementQualities(tet_tags, "minEdge")
            max_edges = gmsh.model.mesh.getElementQualities(tet_tags, "maxEdge")
            if len(min_edges) > 0 and len(max_edges) > 0:
                stats["edge_length"] = {
                    "min": round(min(min_edges), 3),
                    "max": round(max(max_edges), 3),
                }
        except Exception:
            pass

    # Get physical groups with tags
    try:
        groups = {"volumes": [], "surfaces": []}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            entry = {"name": name, "tag": tag}
            if dim == 3:
                groups["volumes"].append(entry)
            elif dim == 2:
                groups["surfaces"].append(entry)
        stats["groups"] = groups
    except Exception:
        pass

    return stats


def write_config(
    mesh_result,
    stack: LayerStack,
    ports: list[PalacePort],
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    boundary_mode_config: BoundaryModeConfig | None = None,
    absorbing_boundary: bool = True,
    hints: dict[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: list[TerminalConfig] | None = None,
    magnetostatic_config: MagnetostaticConfig | None = None,
    current_sources: list[Any] | None = None,
    palace_version: str = "0.16.0",
    validate_schema: bool = False,
    material_overlay: Any | None = None,
    postprocessing_config: Mapping[str, list[dict[str, Any]]] | None = None,
    boundary_postprocessing_config: Mapping[str, list[dict[str, Any]]] | None = None,
) -> Path:
    """Write Palace config.json from a MeshResult.

    Use this to generate config separately after mesh().

    Args:
        mesh_result: Result from generate_mesh(write_config=False)
        stack: LayerStack for material properties
        ports: List of PalacePort objects
        driven_config: Optional DrivenConfig for frequency sweep settings.
            When provided, material dispersion is evaluated at the center
            frequency of the sweep band.
        eigenmode_config: Optional EigenmodeConfig for eigenproblems settings
        absorbing_boundary: Whether to add absorbing (PML) boundary
        hints: Additional config hints merged into the JSON

    Returns:
        Path to the generated config.json

    Raises:
        ValueError: If mesh_result has no groups data

    Example:
        >>> result = sim.mesh(output_dir, write_config=False)
        >>> config_path = write_config(result, stack, ports, driven_config)
    """
    if not mesh_result.groups:
        raise ValueError(
            "MeshResult has no groups data. Was it generated with write_config=False?"
        )

    config_path = generate_palace_config(
        groups=mesh_result.groups,
        ports=ports,
        port_info=mesh_result.port_info,
        stack=stack,
        output_path=mesh_result.output_dir,
        model_name=mesh_result.model_name,
        fmax=mesh_result.fmax,
        simulation_type=simulation_type,
        driven_config=driven_config,
        eigenmode_config=eigenmode_config,
        numerical_config=numerical_config,
        boundary_mode_config=boundary_mode_config,
        absorbing_boundary=absorbing_boundary,
        periodic_axis=mesh_result.periodic_axis,
        hints=hints,
        electrostatic_config=electrostatic_config,
        terminals=terminals,
        magnetostatic_config=magnetostatic_config,
        current_sources=current_sources,
        palace_version=palace_version,
        validate_schema=validate_schema,
        material_overlay=material_overlay,
        postprocessing_config=postprocessing_config,
        boundary_postprocessing_config=boundary_postprocessing_config,
    )

    # Update the mesh_result with the config path
    mesh_result.config_path = config_path

    return config_path


__all__ = [
    "collect_mesh_stats",
    "generate_palace_config",
    "write_config",
]
