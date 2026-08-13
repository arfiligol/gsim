"""Palace configuration assembly from mesh-side physical groups.

This module owns the final Palace ``Domains`` and ``Boundaries`` sections and
the material-resolution sidecar mapping from mesh groups to stack or overlay
material properties. It consumes mesh groups, manifests, ports, terminals, and
solver settings after Gmsh has assigned physical names. Region activation and
Gmsh geometry construction stay in the simulation API and mesh geometry layers;
this module maps their finalized groups into Palace config artifacts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import gmsh

from gsim.palace.config_validation import validate_palace_config
from gsim.palace.models.ports import (
    PalaceDirectionInput,
    PortType,
    normalize_palace_direction,
)
from gsim.palace.models.versions import (
    DEFAULT_PALACE_CONFIG_VERSION,
    PalaceConfigVersion,
    normalize_palace_config_version,
)
from gsim.palace.run_folder import palace_run_folder, prepare_palace_run_folder

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import (
        BoundaryModeConfig,
        CurrentSourceConfig,
        DrivenConfig,
        EigenmodeConfig,
        ElectrostaticConfig,
        MagnetostaticConfig,
        NumericalConfig,
    )
    from gsim.palace.models.ports import PalacePort, TerminalConfig
    from gsim.palace.models.sources import CurrentDirection


def _palace_direction(value: CurrentDirection) -> str | list[float]:
    """Return the Palace JSON value for a validated current-source direction."""
    if isinstance(value, str):
        return value
    return [float(item) for item in value]


def _palace_lumped_port_direction(value: PalaceDirectionInput) -> list[float]:
    """Return the Palace JSON vector for a LumpedPort direction."""
    return [float(item) for item in normalize_palace_direction(value)]


def _default_refinement_config() -> dict[str, Any]:
    """Return the standard Palace mesh-refinement defaults."""
    return {
        "Tol": 1.0e-2,
        "MaxIts": 0,
        "MaxSize": 0,
        "UpdateFraction": 0.7,
        "Nonconformal": False,
        "UniformLevels": 0,
        "Boxes": [],
        "Spheres": [],
    }


def _merged_refinement_config(
    refinement_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge caller refinement overrides into the standard defaults."""
    refinement = _default_refinement_config()
    if refinement_config:
        _deep_merge_config(refinement, dict(refinement_config))
    return refinement


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
    refinement_config: Mapping[str, Any] | None = None,
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION,
    validate_schema: bool = True,
    absorbing_boundary: bool = True,
    periodic_axis: str | None = None,
    hints: dict[str, Any] | None = None,
    problem_output_formats: Mapping[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: list[TerminalConfig] | None = None,
    magnetostatic_config: MagnetostaticConfig | None = None,
    current_sources: list[CurrentSourceConfig] | None = None,
    postprocessing_config: dict[str, Any] | None = None,
    boundary_postprocessing_config: dict[str, Any] | None = None,
    material_overlay: Any | None = None,
    prepare_run_folder: bool = True,
    boundary_mode_config: BoundaryModeConfig | None = None,
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
        numerical_config: Optional NumericalConfig for generic solver settings.
        refinement_config: Optional Palace ``Model.Refinement`` fragment.
        palace_version: Target Palace configuration schema version.
        validate_schema: Validate the final assembled JSON against the target
            Palace version schema before writing.
        absorbing_boundary: Whether to add absorbing (PML) boundary
        periodic_axis: Optional periodic axis identifier
        hints: Additional config hints merged into the JSON
        problem_output_formats: Optional Palace ``Problem.OutputFormats`` fragment.
        postprocessing_config: Optional Palace ``Domains.Postprocessing`` entries
            merged into the default empty postprocessing block.
        boundary_postprocessing_config: Optional Palace
            ``Boundaries.Postprocessing`` entries merged into the generated
            boundary section.
        material_overlay: Optional PDK material overlay path, raw overlay
            mapping, or loaded overlay mapping used to resolve Palace material
            values without mutating the source layer stack.
        prepare_run_folder: Create the canonical Palace run-folder skeleton
            before writing config and sidecars.

    Returns:
        Path to the generated config.json
    """
    run_folder = (
        prepare_palace_run_folder(output_path)
        if prepare_run_folder
        else palace_run_folder(output_path)
    )

    if simulation_type not in (
        "driven",
        "eigenmode",
        "boundarymode",
        "electrostatic",
        "electrostatics",
        "magnetostatic",
    ):
        raise ValueError(f"Unsupported simulation type: {simulation_type}")

    palace_version = normalize_palace_config_version(palace_version)

    # Use driven_config if provided, otherwise fall back to legacy parameters
    if driven_config is not None:
        solver_driven = driven_config.to_palace_config()
    else:
        # Legacy behavior - compute from fmax
        freq_step = fmax / 40e9
        solver_driven = {
            "Samples": [
                {
                    "Type": "Linear",
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
        solver_eigenmode = {
            "N": 10,
            "Tol": 1.0e-6,
            "Target": fmax / 1e9,
        }

    solver_boundarymode = (
        boundary_mode_config.to_palace_config()
        if boundary_mode_config is not None
        else {
            "Freq": fmax / 1e9,
            "N": 1,
            "Save": 0,
            "Target": 0.0,
            "Tol": 1e-6,
            "Type": "Default",
        }
    )

    solver_conf: dict[str, object]
    if numerical_config is not None:
        solver_conf = dict(
            numerical_config.to_solver_config(palace_version=palace_version)
        )
    else:
        # Backward-compatible defaults for direct generate_palace_config() calls
        # that do not provide a NumericalConfig.
        from gsim.palace.models import NumericalConfig

        solver_conf = dict(
            NumericalConfig().to_solver_config(palace_version=palace_version)
        )

    if simulation_type == "driven":
        solver_conf["Driven"] = solver_driven
    elif simulation_type == "eigenmode":
        solver_conf["Eigenmode"] = solver_eigenmode
    elif simulation_type == "boundarymode":
        solver_conf["BoundaryMode"] = solver_boundarymode
    elif simulation_type in ("electrostatic", "electrostatics"):
        if electrostatic_config is not None:
            solver_conf["Electrostatic"] = electrostatic_config.to_palace_config()
        else:
            solver_conf["Electrostatic"] = {"Save": 0}
    elif simulation_type == "magnetostatic":
        if magnetostatic_config is not None:
            solver_conf["Magnetostatic"] = magnetostatic_config.to_palace_config()
        else:
            solver_conf["Magnetostatic"] = {"Save": 0}

    palace_problem_type = {
        "driven": "Driven",
        "eigenmode": "Eigenmode",
        "boundarymode": "BoundaryMode",
        "electrostatic": "Electrostatic",
        "electrostatics": "Electrostatic",
        "magnetostatic": "Magnetostatic",
    }[simulation_type]

    problem_config: dict[str, object] = {
        "Type": palace_problem_type,
        "Verbose": 3,
        "Output": "results/palace",
    }
    if problem_output_formats:
        problem_config["OutputFormats"] = deepcopy(dict(problem_output_formats))

    config: dict[str, object] = {
        "Problem": problem_config,
        "Model": {
            "Mesh": f"{model_name}.msh",
            "L0": 1e-6,  # um
            "Refinement": _merged_refinement_config(refinement_config),
        },
        "Solver": solver_conf,
    }

    # Build domains section
    # Evaluate dispersion models at the center frequency of the sweep band
    material_frequency = (
        driven_config.center_frequency if driven_config is not None else fmax
    )
    stack_materials = stack.materials
    material_resolution_by_name: dict[str, dict[str, Any]] = {}
    material_resolution_rows: list[dict[str, Any]] = []
    if driven_config is not None or material_overlay is not None:
        from gsim.palace.materials import resolve_palace_materials_with_report

        stack_materials, material_resolution_report = (
            resolve_palace_materials_with_report(
                stack.materials,
                material_frequency,
                material_overlay=material_overlay,
            )
        )
        material_resolution_by_name = {
            str(row["stack_material_name"]): row
            for row in material_resolution_report.get("materials", ())
            if isinstance(row, dict) and row.get("stack_material_name") is not None
        }

    materials_by_lower = {
        str(name).lower().strip(): props for name, props in stack_materials.items()
    }

    def lookup_material(name: str) -> dict[str, Any]:
        direct = stack_materials.get(name)
        if isinstance(direct, dict):
            return direct
        resolved = materials_by_lower.get(name.lower().strip())
        return resolved if isinstance(resolved, dict) else {}

    materials: list[dict[str, object]] = []
    for volume_name, info in groups["volumes"].items():
        material_name = volume_name
        is_via = info.get("is_via", False)
        is_shaped_dielectric = info.get("is_shaped_dielectric", False)

        if is_via or is_shaped_dielectric:
            layer = stack.layers.get(material_name)
            if layer is None:
                stack_material_name = material_name
                mat_props = lookup_material(stack_material_name)
            else:
                stack_material_name = layer.material
                mat_props = lookup_material(stack_material_name)
        elif info.get("stack_layer") is not None:
            stack_layer_name = str(info["stack_layer"])
            layer = stack.layers.get(stack_layer_name)
            if layer is None:
                raise ValueError(
                    f"Mesh volume '{volume_name}' references missing stack layer "
                    f"'{stack_layer_name}'."
                )
            stack_material_name = str(info.get("material") or layer.material)
            mat_props = lookup_material(stack_material_name)
            if not mat_props:
                raise ValueError(
                    f"Activated region '{volume_name}' uses material "
                    f"'{stack_material_name}', but that material is not defined "
                    "in the layer stack or material overlay."
                )
        else:
            stack_material_name = material_name
            mat_props = lookup_material(stack_material_name)

        mat_entry: dict[str, object] = {"Attributes": [info["phys_group"]]}

        if volume_name in {"airbox", "air"}:
            mat_entry["Permittivity"] = mat_props.get("permittivity", 1.0)
            mat_entry["LossTan"] = mat_props.get("loss_tangent", 0.0)

            if "permeability" in mat_props:
                mat_entry["Permeability"] = mat_props["permeability"]

            if "material_axes" in mat_props:
                mat_entry["MaterialAxes"] = mat_props["material_axes"]
        elif is_via:
            sigma = mat_props.get("conductivity", 0.0)
            mat_entry["Permittivity"] = 1.0
            if isinstance(sigma, (int, float)) and sigma > 0:
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
            if (isinstance(sigma, (int, float)) and sigma > 0) or isinstance(
                sigma, list
            ):
                mat_entry["Conductivity"] = sigma
            elif isinstance(lt, list) or (isinstance(lt, (int, float)) and lt > 0):
                mat_entry["LossTan"] = lt
            else:
                mat_entry["LossTan"] = 0.0

            if "material_axes" in mat_props:
                mat_entry["MaterialAxes"] = mat_props["material_axes"]

        materials.append(mat_entry)
        material_resolution = material_resolution_by_name.get(stack_material_name)
        if material_resolution is not None:
            material_resolution_rows.append(
                _material_resolution_config_row(
                    material_row_index=len(materials),
                    material_attribute=info["phys_group"],
                    material_attributes=[info["phys_group"]],
                    volume_name=volume_name,
                    stack_material_name=stack_material_name,
                    palace_material=mat_entry,
                    resolution=material_resolution,
                )
            )

    postprocessing: dict[str, object] = {"Energy": [], "Probe": []}
    if postprocessing_config:
        postprocessing.update(deepcopy(postprocessing_config))

    config["Domains"] = {
        "Materials": materials,
        "Postprocessing": postprocessing,
    }

    # Build boundaries section
    conductors: list[dict[str, object]] = []

    for name, info in groups["conductor_surfaces"].items():
        if info.get("postprocessing_only"):
            continue
        boundary_attrs = _boundary_attributes_for_conductor_surface(groups, info)
        if not boundary_attrs:
            continue
        # Extract layer name from "layer_xy" or "layer_z"
        layer_name = str(info.get("layer", name.rsplit("_", 1)[0]))
        layer = stack.layers.get(layer_name)
        if layer:
            mat_props = lookup_material(layer.material)
            conductors.append(
                {
                    "Attributes": boundary_attrs,
                    "Conductivity": mat_props.get("conductivity", 5.8e7),
                    "Thickness": layer.zmax - layer.zmin,
                }
            )

    # Handle PEC surfaces (planar conductors + PEC blocks)
    pec_attrs: list[int] = []
    split_pec_sources: set[str] = set()
    for name, info in groups.get("pec_surfaces", {}).items():
        child_attrs = _finite_conductor_split_boundary_attributes(
            groups,
            str(name),
            representation="A",
        )
        if child_attrs:
            pec_attrs.extend(child_attrs)
            split_pec_sources.add(str(name))
            continue
        pec_attrs.extend(_physical_group_values(info.get("phys_group")))
    for source_id in _finite_conductor_split_source_ids(groups, representation="A"):
        if source_id in split_pec_sources:
            continue
        pec_attrs.extend(
            _finite_conductor_split_boundary_attributes(
                groups,
                source_id,
                representation="A",
            )
        )
    for representation in ("A", "B"):
        pec_attrs.extend(
            _volume_interface_boundary_attributes(groups, representation=representation)
        )
    pec_attrs = sorted(set(pec_attrs))

    is_electrostatic = simulation_type in ("electrostatic", "electrostatics")
    is_magnetostatic = simulation_type == "magnetostatic"
    boundaries: dict[str, object]

    if (
        is_electrostatic
        and terminals
        and _has_sgb_structured_conductor_surfaces(groups)
    ):
        boundaries = _sgb_electrostatic_boundaries(groups, terminals)

    elif is_electrostatic and terminals:
        terminal_layer_names: set[str] = {t.layer for t in terminals}
        via_boundary = groups.get("via_boundary_surfaces", {})
        terminal_entries, assigned_pgs, vias_on_terminal = _selector_entries(
            groups=groups,
            stack=stack,
            selectors=terminals,
        )

        # Anything that wasn't assigned to a terminal becomes Ground.
        ground_attrs: list[int] = []
        for surf_info in groups.get("conductor_surfaces", {}).values():
            if surf_info.get("postprocessing_only"):
                continue
            ground_attrs.extend(
                pg
                for pg in _boundary_attributes_for_conductor_surface(
                    groups,
                    surf_info,
                )
                if pg not in assigned_pgs
            )
        for source_id in _finite_conductor_split_source_ids(groups):
            child_attrs = _finite_conductor_split_boundary_attributes(
                groups,
                source_id,
            )
            if child_attrs and not set(child_attrs) & assigned_pgs:
                ground_attrs.extend(child_attrs)
        for representation in ("A", "B"):
            ground_attrs.extend(
                pg
                for pg in _volume_interface_boundary_attributes(
                    groups,
                    representation=representation,
                )
                if pg not in assigned_pgs
            )
        pec_surfaces = groups.get("pec_surfaces", {})
        for pec_name, surf_info in pec_surfaces.items():
            pec_child_attrs = _finite_conductor_split_boundary_attributes(
                groups,
                str(pec_name),
                representation="A",
            )
            if pec_child_attrs:
                ground_attrs.extend(
                    pg for pg in pec_child_attrs if pg not in assigned_pgs
                )
            else:
                ground_attrs.extend(
                    pg
                    for pg in _physical_group_values(surf_info.get("phys_group"))
                    if pg not in assigned_pgs
                )

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
                if _via_touches_layer(stack, via_name, cond_name):
                    ground_attrs.extend(via_pgs)
                    break

        boundaries = {
            "Terminal": terminal_entries,
        }
        if ground_attrs:
            boundaries["Ground"] = {"Attributes": sorted(set(ground_attrs))}

    elif simulation_type == "boundarymode":
        boundaries = {}
        if conductors:
            boundaries["Conductivity"] = conductors
        if pec_attrs:
            boundaries["PEC"] = {"Attributes": sorted(set(pec_attrs))}

    elif is_magnetostatic and current_sources:
        surface_currents: list[dict[str, object]] = []
        magnetic_fluxes: list[dict[str, object]] = []
        for index, source in enumerate(current_sources, start=1):
            surface_current, attrs = _surface_current_entry(
                groups=groups,
                stack=stack,
                source=source,
                index=index,
            )
            surface_currents.append(surface_current)
            magnetic_fluxes.append(
                {
                    "Index": index,
                    "Attributes": attrs,
                    "Type": "Magnetic",
                    "TwoSided": False,
                }
            )

        boundaries = {
            "SurfaceCurrent": surface_currents,
        }
        pmc_attrs = _outer_boundary_attributes(groups)
        if pmc_attrs:
            boundaries["PMC"] = {"Attributes": pmc_attrs}
        if magnetic_fluxes:
            boundaries["Postprocessing"] = {"SurfaceFlux": magnetic_fluxes}

    elif is_magnetostatic:
        raise ValueError(
            "Magnetostatic config generation requires at least one current source."
        )

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

                if port.multi_element:
                    # Multi-element port (CPW)
                    if port_group.get("type") == "cpw":
                        elements = [
                            {
                                "Attributes": [elem["phys_group"]],
                                "Direction": _palace_lumped_port_direction(
                                    elem["direction"]
                                ),
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
                        direction = _palace_lumped_port_direction(
                            cast(PalaceDirectionInput, port.direction)
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
            port_idx += 1

        # Assign unique indices to passive reactive ports now that all primary
        # indices are consumed (port_idx is one past the last primary index).
        synthetic_idx = port_idx
        for entry in passive_reactive_ports:
            entry["Index"] = synthetic_idx
            synthetic_idx += 1
        lumped_ports.extend(passive_reactive_ports)

        boundaries = {
            "Conductivity": conductors,
            "LumpedPort": lumped_ports,
            "WavePort": wave_ports,
        }

        # Add PEC boundaries if any exist
        if pec_attrs:
            boundaries["PEC"] = {"Attributes": pec_attrs}

    if "absorbing" in groups["boundary_surfaces"] and absorbing_boundary:
        absorbing_pg = groups["boundary_surfaces"]["absorbing"]["phys_group"]
        # phys_group may be a list (multiple ___None groups) or a single int
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

        model_config = cast(dict[str, Any], config["Model"])
        floquet_vector = eigenmode_config.compute_floquet_wave_vector(
            periodic_axis=axis_lit,
            l0=float(model_config["L0"]),
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

    if boundary_postprocessing_config:
        # Boundary postprocessing is a Palace contract, not a private layout
        # convention. Keep it as an explicit merge point so role-based builders
        # can wire EPR/flux domains without hand-editing config.json.
        existing_boundary_postprocessing = boundaries.get("Postprocessing")
        boundary_postprocessing: dict[str, Any] = (
            dict(cast(Mapping[str, Any], existing_boundary_postprocessing))
            if isinstance(existing_boundary_postprocessing, dict)
            else {}
        )
        boundary_postprocessing.update(deepcopy(boundary_postprocessing_config))
        interface_resolution_rows = _resolve_boundary_dielectric_interfaces(
            boundary_postprocessing,
            material_frequency_hz=material_frequency,
            material_overlay=material_overlay,
        )
        boundaries["Postprocessing"] = boundary_postprocessing
    else:
        interface_resolution_rows = []

    # Merge any extra hints into the config
    if hints:
        _reject_protected_config_hints(hints)
        _deep_merge_config(config, hints)

    if validate_schema:
        validate_palace_config(
            cast(dict[str, Any], config),
            palace_version=palace_version,
        )

    # Write config file
    config_path = run_folder.config_path
    with config_path.open("w") as f:
        json.dump(config, f, indent=4)

    if material_resolution_rows or interface_resolution_rows:
        material_resolution_path = run_folder.material_resolution_path
        material_resolution_path.parent.mkdir(parents=True, exist_ok=True)
        with material_resolution_path.open("w") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "materials": material_resolution_rows,
                    "interfaces": interface_resolution_rows,
                },
                f,
                indent=4,
            )

    # Write port information file
    port_info_path = run_folder.port_information_path
    port_info_path.parent.mkdir(parents=True, exist_ok=True)
    port_info_struct = {"ports": port_info, "unit": 1e-6, "name": model_name}
    with port_info_path.open("w") as f:
        json.dump(port_info_struct, f, indent=4)

    return config_path


def _material_resolution_config_row(
    *,
    material_row_index: int,
    material_attribute: int,
    material_attributes: list[int],
    volume_name: str,
    stack_material_name: str,
    palace_material: dict[str, object],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """Build one material-resolution provenance row for generated config."""
    row = {
        "material_row_index": material_row_index,
        "material_attribute": material_attribute,
        "material_attributes": material_attributes,
        "volume_name": volume_name,
        "stack_material_name": stack_material_name,
        "palace_material": dict(palace_material),
    }
    row.update(dict(resolution))
    return row


def _has_sgb_structured_conductor_surfaces(groups: dict[str, Any]) -> bool:
    """Return whether groups originate from current structured SGB surfaces."""
    return any(
        isinstance(info, Mapping)
        and info.get("sgb_record") == "final_physical_group"
        and info.get("source") == "volume_interface"
        and info.get("solver_use") == "solver_active"
        and info.get("interface_type") in {"MA", "MS"}
        for info in groups.get("boundary_surfaces", {}).values()
    )


def _sgb_electrostatic_boundaries(
    groups: dict[str, Any],
    terminals: list[Any],
) -> dict[str, object]:
    """Assign whole structured SGB conductor components to terminals or ground."""
    component_attrs: dict[str, set[int]] = {}
    component_nets: dict[str, set[str | None]] = {}
    attribute_components: dict[int, str] = {}
    for info in groups.get("boundary_surfaces", {}).values():
        if (
            not isinstance(info, Mapping)
            or info.get("sgb_record") != "final_physical_group"
        ):
            continue
        if info.get("source") != "volume_interface":
            continue
        if info.get("solver_use") != "solver_active":
            continue
        if info.get("interface_type") not in {"MA", "MS"}:
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
                "SGB conductor component "
                f"{component_id!r} has conflicting terminal nets."
            )
        component_net[component_id] = next(iter(nets))

    terminal_nets: set[str] = set()
    terminal_entries: list[dict[str, object]] = []
    assigned_components: set[str] = set()
    for index, terminal in enumerate(terminals, start=1):
        if getattr(terminal, "center", None) is None:
            raise ValueError(f"SGB terminal {terminal.name!r} needs center.")
        net_id = _sgb_terminal_net_id(terminal)
        if net_id in terminal_nets:
            raise ValueError(f"SGB terminal net {net_id!r} appears more than once.")
        terminal_nets.add(net_id)
        matching_components = sorted(
            component_id
            for component_id, component_value in component_net.items()
            if component_value == net_id
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
                    {
                        attr
                        for component_id in matching_components
                        for attr in component_attrs[component_id]
                    }
                ),
            }
        )

    boundaries: dict[str, object] = {"Terminal": terminal_entries}
    ground_attrs = sorted(
        {
            attr
            for component_id, attrs in component_attrs.items()
            if component_id not in assigned_components
            for attr in attrs
        }
    )
    if ground_attrs:
        boundaries["Ground"] = {"Attributes": ground_attrs}
    return boundaries


def _sgb_terminal_net_id(terminal: Any) -> str:
    """Match the exact terminal net id emitted by the SGB stack lowering."""
    label = getattr(terminal, "physical_label", None) or terminal.name
    if not isinstance(label, str) or not label.strip():
        raise ValueError("SGB terminal needs a non-empty name or physical_label.")
    return re.sub(r"[^A-Za-z0-9_@]+", "_", f"{terminal.layer}@{label}".strip()).strip(
        "_"
    )


def _selector_entries(
    *,
    groups: dict[str, Any],
    stack: LayerStack,
    selectors: list[Any],
) -> tuple[list[dict[str, object]], set[int], set[str]]:
    """Resolve terminal selectors into Palace entries and selected attributes."""
    entries: list[dict[str, object]] = []
    assigned_pgs: set[int] = set()
    selected_vias: set[str] = set()
    pec_surfaces = groups.get("pec_surfaces", {})
    via_boundary = groups.get("via_boundary_surfaces", {})

    for idx, selector in enumerate(selectors, start=1):
        unique_attrs, selector_vias = _selector_attributes(
            groups=groups,
            stack=stack,
            selector=selector,
            pec_surfaces=pec_surfaces,
            via_boundary=via_boundary,
        )
        duplicate_attrs = sorted(attr for attr in unique_attrs if attr in assigned_pgs)
        if duplicate_attrs:
            selector_name = getattr(selector, "name", f"T{idx}")
            raise ValueError(
                f"Terminal {selector_name!r} on layer {selector.layer!r} matched "
                "attributes already selected by earlier terminals: "
                f"{duplicate_attrs}."
            )
        selected_vias.update(selector_vias)
        assigned_pgs.update(unique_attrs)
        entries.append(
            {
                "Index": idx,
                "Attributes": unique_attrs,
            }
        )

    return entries, assigned_pgs, selected_vias


def _selector_attributes(
    *,
    groups: dict[str, Any],
    stack: LayerStack,
    selector: Any,
    pec_surfaces: dict[str, Any] | None = None,
    via_boundary: dict[str, Any] | None = None,
) -> tuple[list[int], set[str]]:
    """Resolve a layer/center selector to generated boundary attributes."""
    if not selector.layer:
        raise ValueError("Selector layer is required for physical-group lookup.")

    attrs: list[int] = []
    selected_vias: set[str] = set()
    resolved_pec_surfaces = (
        groups.get("pec_surfaces", {}) if pec_surfaces is None else pec_surfaces
    )
    resolved_via_boundary = (
        groups.get("via_boundary_surfaces", {})
        if via_boundary is None
        else via_boundary
    )

    for surf_name, surf_info in groups.get("conductor_surfaces", {}).items():
        if surf_info.get("postprocessing_only"):
            continue
        if _conductor_matches_selector(surf_name, surf_info, selector):
            attrs.extend(_boundary_attributes_for_conductor_surface(groups, surf_info))
    c_volume_ids = _c_volume_ids_for_selector(groups, selector)
    for surf_info in groups.get("boundary_surfaces", {}).values():
        if not isinstance(surf_info, dict) or surf_info.get("postprocessing_only"):
            continue
        if surf_info.get("source") != "volume_interface":
            continue
        if surf_info.get("metal_body_id") != selector.layer:
            continue
        volume_id = surf_info.get("metal_volume_id")
        if selector.center is not None and volume_id not in c_volume_ids:
            continue
        attrs.extend(_physical_group_values(surf_info.get("phys_group")))
    attrs.extend(
        _volume_interface_boundary_attributes(
            groups,
            selector=selector,
            volume_ids=c_volume_ids,
        )
    )
    for source_id in _finite_conductor_split_sources_for_selector(
        groups,
        selector,
    ):
        attrs.extend(_finite_conductor_split_boundary_attributes(groups, source_id))

    for pec_name, pec_info in resolved_pec_surfaces.items():
        if _pec_matches_selector(pec_name, pec_info, selector):
            child_attrs = _finite_conductor_split_boundary_attributes(
                groups,
                str(pec_name),
                representation="A",
            )
            if child_attrs:
                attrs.extend(child_attrs)
            else:
                attrs.extend(_physical_group_values(pec_info.get("phys_group")))

    for via_name, via_pgs in resolved_via_boundary.items():
        if _via_touches_layer(stack, via_name, selector.layer):
            attrs.extend(_physical_group_values(via_pgs))
            selected_vias.add(via_name)

    return sorted(set(attrs)), selected_vias


def _volume_interface_boundary_attributes(
    groups: dict[str, Any],
    *,
    representation: str | None = None,
    selector: Any | None = None,
    volume_ids: set[str] | None = None,
) -> list[int]:
    """Return boundary attributes for matching volume-interface child surfaces."""
    attrs: list[int] = []
    for surf_info in _volume_interface_child_surfaces(groups):
        if (
            representation is not None
            and str(surf_info.get("representation", "")).upper() != representation
        ):
            continue
        if selector is not None and surf_info.get("layer") != selector.layer:
            continue
        if (
            selector is not None
            and selector.center is not None
            and surf_info.get("metal_volume_id") not in (volume_ids or set())
        ):
            continue
        attrs.extend(_physical_group_values(surf_info.get("phys_group")))
    return sorted(set(attrs))


def _volume_interface_child_surfaces(groups: dict[str, Any]):
    """Yield active volume-interface child surface records."""
    for surf_info in groups.get("boundary_surfaces", {}).values():
        if not isinstance(surf_info, dict):
            continue
        if surf_info.get("postprocessing_only"):
            continue
        if surf_info.get("source") != "volume_interface":
            continue
        if surf_info.get("metal_body_id") is None:
            continue
        yield surf_info


def _c_volume_ids_for_selector(groups: dict[str, Any], selector: Any) -> set[str]:
    """Return conductor-volume identifiers selected by a terminal center."""
    if selector.center is None:
        return set()
    volume_ids: set[str] = set()
    for surf_info in groups.get("boundary_surfaces", {}).values():
        if not isinstance(surf_info, dict):
            continue
        if surf_info.get("source") != "volume_interface":
            continue
        if surf_info.get("metal_body_id") != selector.layer:
            continue
        volume_id = surf_info.get("metal_volume_id")
        if not isinstance(volume_id, str):
            continue
        if _bbox_contains_center(
            surf_info.get("bbox"),
            selector.center,
        ):
            volume_ids.add(volume_id)
    return volume_ids


def _boundary_attributes_for_conductor_surface(
    groups: dict[str, Any],
    surf_info: dict[str, object],
) -> list[int]:
    """Return terminal boundary attributes for one conductor-surface record."""
    if surf_info.get("source") == "finite_conductor_terminal_shell":
        source_id = surf_info.get("source_id")
        if isinstance(source_id, str) and source_id:
            child_attrs = _finite_conductor_split_boundary_attributes(
                groups,
                source_id,
            )
            if child_attrs:
                return child_attrs
    return _physical_group_values(surf_info.get("phys_group"))


def _finite_conductor_split_boundary_attributes(
    groups: dict[str, Any],
    source_id: str,
    *,
    representation: str | None = None,
) -> list[int]:
    """Return active child attributes for a finite-conductor split source."""
    attrs: list[int] = []
    for info in groups.get("conductor_surfaces", {}).values():
        if not isinstance(info, dict):
            continue
        if (
            representation is not None
            and str(info.get("representation", "")).upper() != representation
        ):
            continue
        if info.get("source_id") != source_id:
            continue
        if info.get("postprocessing_only") or info.get("surface_epr"):
            continue
        attrs.extend(_physical_group_values(info.get("phys_group")))
    return sorted(set(attrs))


def _finite_conductor_split_source_ids(
    groups: dict[str, Any],
    *,
    representation: str | None = None,
) -> tuple[str, ...]:
    """Return ordered unique active finite-conductor split source identifiers."""
    source_ids: list[str] = []
    for info in groups.get("conductor_surfaces", {}).values():
        if not isinstance(info, dict):
            continue
        if (
            representation is not None
            and str(info.get("representation", "")).upper() != representation
        ):
            continue
        if info.get("postprocessing_only") or info.get("surface_epr"):
            continue
        source_id = info.get("source_id")
        if isinstance(source_id, str) and source_id:
            source_ids.append(source_id)
    return tuple(dict.fromkeys(source_ids))


def _finite_conductor_split_sources_for_selector(
    groups: dict[str, Any],
    selector: Any,
    *,
    representation: str | None = None,
) -> tuple[str, ...]:
    """Return finite-conductor split sources matching a terminal selector."""
    source_ids: list[str] = []
    for info in groups.get("conductor_surfaces", {}).values():
        if not isinstance(info, dict):
            continue
        if (
            representation is not None
            and str(info.get("representation", "")).upper() != representation
        ):
            continue
        if info.get("postprocessing_only") or info.get("surface_epr"):
            continue
        if info.get("layer") != selector.layer:
            continue
        if selector.center is not None and not _bbox_contains_center(
            info.get("bbox"), selector.center
        ):
            continue
        source_id = info.get("source_id")
        if isinstance(source_id, str) and source_id:
            source_ids.append(source_id)
    return tuple(dict.fromkeys(source_ids))


def _conductor_matches_selector(
    surf_name: str,
    surf_info: dict[str, object],
    selector: Any,
) -> bool:
    """Return whether a conductor surface record matches a terminal selector."""
    surf_layer = surf_info.get("layer", surf_name.rsplit("_", 1)[0])
    if surf_layer != selector.layer and surf_name != selector.layer:
        return False
    if selector.center is None:
        return True
    return _bbox_contains_center(surf_info.get("bbox"), selector.center)


def _surface_current_entry(
    *,
    groups: dict[str, Any],
    stack: LayerStack,
    source: CurrentSourceConfig,
    index: int,
) -> tuple[dict[str, object], list[int]]:
    """Build one Palace SurfaceCurrent row from selector-based source intent."""
    if source.elements:
        elements: list[dict[str, object]] = []
        all_attrs: list[int] = []
        selected_attrs: dict[int, int] = {}
        for element_index, element in enumerate(source.elements, start=1):
            attrs, _ = _selector_attributes(
                groups=groups,
                stack=stack,
                selector=element,
            )
            if not attrs:
                raise ValueError(
                    f"Current source {source.name!r} element {element_index} "
                    f"on layer {element.layer!r} did not match any conductor "
                    "surface attributes."
                )
            duplicate_attrs = sorted(attr for attr in attrs if attr in selected_attrs)
            if duplicate_attrs:
                raise ValueError(
                    f"Current source {source.name!r} element {element_index} "
                    "matched attributes already selected by earlier elements: "
                    f"{duplicate_attrs}."
                )
            for attr in attrs:
                selected_attrs[attr] = element_index
            element_entry: dict[str, object] = {
                "Attributes": attrs,
                "Direction": _palace_direction(element.direction),
            }
            if element.coordinate_system is not None:
                element_entry["CoordinateSystem"] = element.coordinate_system
            elements.append(element_entry)
            all_attrs.extend(attrs)
        return {"Index": index, "Elements": elements}, sorted(set(all_attrs))

    attrs, _ = _selector_attributes(groups=groups, stack=stack, selector=source)
    if not attrs:
        raise ValueError(
            f"Current source {source.name!r} on layer {source.layer!r} "
            "did not match any conductor surface attributes."
        )
    entry: dict[str, object] = {
        "Index": index,
        "Attributes": attrs,
        "Direction": _palace_direction(source.direction),
    }
    if source.coordinate_system is not None:
        entry["CoordinateSystem"] = source.coordinate_system
    return entry, attrs


def _pec_matches_selector(
    pec_name: str,
    pec_info: dict[str, object],
    selector: Any,
) -> bool:
    """Return whether a PEC surface record matches a terminal selector."""
    pec_layer = pec_info.get("layer", pec_name)
    if pec_layer != selector.layer and pec_name != selector.layer:
        return False
    if selector.center is None:
        return True
    return _bbox_contains_center(pec_info.get("bbox"), selector.center)


def _bbox_contains_center(
    bbox: Any,
    center: tuple[float, float],
) -> bool:
    """Return whether a two-dimensional center lies within a bounding box."""
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 6
        or not all(isinstance(value, (int, float)) for value in bbox)
    ):
        return False
    x, y = center
    tol = 1e-6
    return (
        float(bbox[0]) - tol <= x <= float(bbox[3]) + tol
        and float(bbox[1]) - tol <= y <= float(bbox[4]) + tol
    )


def _via_touches_layer(
    stack: LayerStack, via_name: str, conductor_layer_name: str
) -> bool:
    """Return whether a via z-range overlaps or touches a conductor layer."""
    via = stack.layers.get(via_name)
    cond = stack.layers.get(conductor_layer_name)
    if via is None or cond is None:
        return False
    return via.zmin <= cond.zmax and via.zmax >= cond.zmin


def _physical_group_values(value: Any) -> list[int]:
    """Normalize one physical-group value into integer attributes."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (str, bytes)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if isinstance(item, int)]
    return []


def _outer_boundary_attributes(groups: dict[str, Any]) -> list[int]:
    """Return physical attributes for the generated absorbing outer boundary."""
    boundary_info = groups.get("boundary_surfaces", {}).get("absorbing")
    if not isinstance(boundary_info, dict):
        return []
    return sorted(set(_physical_group_values(boundary_info.get("phys_group"))))


def _resolve_boundary_dielectric_interfaces(
    boundary_postprocessing: dict[str, Any],
    *,
    material_frequency_hz: float,
    material_overlay: Any | None,
) -> list[dict[str, Any]]:
    """Resolve material-backed boundary dielectric rows and record provenance."""
    dielectric_rows = boundary_postprocessing.get("Dielectric")
    if not isinstance(dielectric_rows, list):
        return []

    resolution_rows: list[dict[str, Any]] = []
    for interface_row_index, interface in enumerate(dielectric_rows, start=1):
        if not isinstance(interface, dict):
            continue
        interface = cast(dict[str, Any], interface)
        material_name = interface.pop("_MaterialName", None)
        if material_name is None:
            continue

        material_name = str(material_name)
        resolved_material, resolution = _resolve_single_interface_material(
            material_name,
            material_frequency_hz=material_frequency_hz,
            material_overlay=material_overlay,
        )
        if "permittivity" in resolved_material:
            interface["Permittivity"] = resolved_material["permittivity"]
        if "loss_tangent" in resolved_material:
            interface["LossTan"] = resolved_material["loss_tangent"]
        elif "LossTan" not in interface:
            interface["LossTan"] = 0.0

        if "Permittivity" not in interface:
            msg = (
                "Dielectric interface material resolution did not provide "
                f"Permittivity for {material_name!r}."
            )
            raise ValueError(msg)

        resolution_rows.append(
            _interface_material_resolution_config_row(
                interface_row_index=interface_row_index,
                surface_index=_optional_int(interface.get("Index")),
                surface_attributes=_int_list(interface.get("Attributes")),
                interface_type=interface.get("Type"),
                material_name=material_name,
                palace_interface=interface,
                resolution=resolution,
            )
        )
    return resolution_rows


def _deep_merge_config(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Recursively merge config updates into a mutable target mapping."""
    for key, value in updates.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
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


def _hint_path(path: tuple[str, ...]) -> str:
    """Format a nested config hint path for diagnostics."""
    return ".".join(path)


def _reject_protected_config_hints(
    hints: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Reject hints that would overwrite typed Palace configuration owners."""
    for key, value in hints.items():
        current_path = (*path, str(key))
        if current_path in _PROTECTED_HINT_PATHS:
            raise ValueError(
                "Palace config hints cannot overwrite "
                f"{_hint_path(current_path)}. Use the typed owner API instead."
            )
        if isinstance(value, Mapping):
            _reject_protected_config_hints(value, path=current_path)


def _resolve_single_interface_material(
    material_name: str,
    *,
    material_frequency_hz: float,
    material_overlay: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one named interface material with its provenance record."""
    from gsim.palace.materials import resolve_palace_materials_with_report

    resolved, report = resolve_palace_materials_with_report(
        {material_name: {}},
        material_frequency_hz,
        material_overlay=material_overlay,
    )
    resolution_rows = report.get("materials", ())
    resolution = (
        dict(resolution_rows[0])
        if isinstance(resolution_rows, list) and resolution_rows
        else {}
    )
    return dict(resolved.get(material_name, {})), resolution


def _interface_material_resolution_config_row(
    *,
    interface_row_index: int,
    surface_index: int | None,
    surface_attributes: list[int],
    interface_type: Any,
    material_name: str,
    palace_interface: dict[str, object],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """Build one interface-material resolution provenance row."""
    row = {
        "interface_row_index": interface_row_index,
        "surface_index": surface_index,
        "surface_attributes": surface_attributes,
        "interface_type": interface_type,
        "interface_material_name": material_name,
        "palace_interface": dict(palace_interface),
    }
    row.update(dict(resolution))
    return row


def _optional_int(value: Any) -> int | None:
    """Return an integer-like value unless it is boolean or absent."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _int_list(value: Any) -> list[int]:
    """Normalize an iterable of numeric values to integer attributes."""
    if isinstance(value, (str, bytes)) or value is None:
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


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
    refinement_config: Mapping[str, Any] | None = None,
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION,
    validate_schema: bool = True,
    absorbing_boundary: bool = True,
    hints: dict[str, Any] | None = None,
    problem_output_formats: Mapping[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: list[TerminalConfig] | None = None,
    magnetostatic_config: MagnetostaticConfig | None = None,
    current_sources: list[CurrentSourceConfig] | None = None,
    postprocessing_config: dict[str, Any] | None = None,
    boundary_postprocessing_config: dict[str, Any] | None = None,
    material_overlay: Any | None = None,
    prepare_run_folder: bool = True,
    boundary_mode_config: BoundaryModeConfig | None = None,
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
        problem_output_formats: Optional Palace ``Problem.OutputFormats`` fragment.
        postprocessing_config: Optional Palace ``Domains.Postprocessing`` entries
            merged into the default empty postprocessing block.
        boundary_postprocessing_config: Optional Palace
            ``Boundaries.Postprocessing`` entries merged into the generated
            boundary section.
        material_overlay: Optional PDK material overlay path, raw overlay
            mapping, or loaded overlay mapping used to resolve Palace material
            values without mutating the source layer stack.
        prepare_run_folder: Create the canonical Palace run-folder skeleton
            before writing config and sidecars.

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
        boundary_mode_config=boundary_mode_config,
        numerical_config=numerical_config,
        refinement_config=refinement_config,
        palace_version=palace_version,
        validate_schema=validate_schema,
        absorbing_boundary=absorbing_boundary,
        periodic_axis=mesh_result.periodic_axis,
        hints=hints,
        problem_output_formats=problem_output_formats,
        electrostatic_config=electrostatic_config,
        terminals=terminals,
        magnetostatic_config=magnetostatic_config,
        current_sources=current_sources,
        postprocessing_config=postprocessing_config,
        boundary_postprocessing_config=boundary_postprocessing_config,
        material_overlay=material_overlay,
        prepare_run_folder=prepare_run_folder,
    )

    # Update the mesh_result with the config path
    mesh_result.config_path = config_path

    return config_path


__all__ = [
    "collect_mesh_stats",
    "generate_palace_config",
    "write_config",
]
