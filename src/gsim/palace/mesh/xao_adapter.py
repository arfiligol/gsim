"""Optional adapter for Semantic Geometry Builder XAO route geometry.

This module is a narrow handoff point between the standalone
``semantic_geometry_builder`` package and gsim's Palace mesh/config pipeline.
SGB is not the native gsim geometry path: callers enter this adapter only when
they explicitly request Surface EPR route A/B/C geometry or pass an existing SGB
XAO plus ``metadata/semantic_geometry`` sidecar directory.

The ownership split is contract-first. SGB owns route topology, physical-group
plans, XAO export, and semantic sidecars. Today the interface ownership contract
is encoded in final physical group names such as ``MA__...``, ``MS__...``, and
``SA__...``; this adapter parses that grammar until SGB exports first-class
interface fields. gsim owns mesh generation from that exported contract, Palace
config generation, mesh manifests, and downstream result/report semantics. This
adapter validates that the expected SGB files are present and fails loudly when
the optional contract is unavailable or incomplete.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import gmsh

from gsim.palace.mesh.config_generator import collect_mesh_stats, generate_palace_config
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.manifest import build_mesh_manifest
from gsim.palace.models.versions import DEFAULT_PALACE_CONFIG_VERSION
from gsim.palace.run_folder import prepare_palace_run_folder

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import (
        ActivatedRegion,
        CurrentSourceConfig,
        DrivenConfig,
        EigenmodeConfig,
        ElectrostaticConfig,
        MagnetostaticConfig,
        NumericalConfig,
        PalaceConfigVersion,
        PalacePort,
        TerminalConfig,
    )

_SURFACE_EPR_INTERFACE_TYPES = {"MA", "MS", "SA", "MS_MA"}
_FACE_KIND_SEGMENTS = {"TOP": "top", "BOTTOM": "bottom", "SIDEWALL": "sidewall"}
_SEMANTIC_ID_RE = re.compile(r"[^A-Za-z0-9_@]+")


def generate_mesh_from_semantic_geometry_builder(
    *,
    component: Any,
    stack: LayerStack,
    ports: Sequence[PalacePort] = (),
    output_dir: str | Path,
    route: Literal["A", "B", "C"],
    activated_regions: Sequence[ActivatedRegion],
    terminals: Sequence[TerminalConfig] = (),
    model_name: str = "palace",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    fmax: float = 100e9,
    simulation_type: Literal[
        "driven",
        "eigenmode",
        "electrostatic",
        "magnetostatic",
    ] = "electrostatic",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    refinement_config: Mapping[str, Any] | None = None,
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION,
    validate_schema: bool = True,
    absorbing_boundary: bool = True,
    problem_output_formats: Mapping[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    magnetostatic_config: MagnetostaticConfig | None = None,
    current_sources: Sequence[CurrentSourceConfig] = (),
    postprocessing_config: dict[str, Any] | None = None,
    boundary_postprocessing_config: dict[str, Any] | None = None,
    material_overlay: Any | None = None,
    write_config: bool = True,
    show_gui: bool = False,
    high_order_elements: bool = False,
    high_order_order: int = 2,
    high_order_optimize: bool = True,
) -> MeshResult:
    """Build optional SGB route geometry, then mesh the exported XAO.

    This entrypoint is used only for explicit Surface EPR route A/B/C requests.
    It lowers the gsim component/stack inputs into SGB's reviewed GDS plus stack
    JSON contract, asks SGB to write the route XAO and semantic sidecars, and
    then hands those artifacts back to gsim for mesh/config generation. Missing
    SGB installation, missing component input, unsupported route names, or
    absent route artifacts are hard failures rather than native-path fallbacks.
    """
    if component is None:
        raise ValueError("SGB Surface EPR route meshing requires a component.")
    normalized_route = route.upper()
    if normalized_route == "A":
        route = "A"
    elif normalized_route == "B":
        route = "B"
    elif normalized_route == "C":
        route = "C"
    else:
        raise ValueError(f"Unsupported SGB Surface EPR route: {normalized_route!r}.")
    run_folder = prepare_palace_run_folder(output_dir)
    semantic_root = run_folder.metadata_dir / "semantic_geometry"
    xao_path = run_folder.geometry_dir / f"semantic_geometry_route_{route.lower()}.xao"
    gds_path = run_folder.geometry_dir / "semantic_geometry_input.gds"
    stack_path = run_folder.geometry_dir / "semantic_geometry_input.stack.json"

    try:
        from semantic_geometry_builder import (
            SemanticGeometryBuilder,
            build_gds_stack_geometry_input,
        )
    except ImportError as error:
        msg = (
            "Surface EPR A/B/C route meshing requires the "
            "semantic-geometry-builder package installed with gsim."
        )
        raise ImportError(msg) from error

    _write_component_gds(component, gds_path)
    top_cell_name = _component_gds_top_cell_name(component, gds_path)
    stack_mapping, semantic_layer_map, semantic_stack_layer_map = (
        _sgb_stack_mapping_from_gsim_inputs(
            component=component,
            stack=stack,
            gds_path=gds_path,
            activated_regions=activated_regions,
            terminals=terminals,
        )
    )
    stack_path.write_text(json.dumps(stack_mapping, indent=2) + "\n")

    build_input = build_gds_stack_geometry_input(
        gds_file=gds_path,
        stack_file=stack_path,
        top_cell_name=top_cell_name,
        metadata={"source": "gsim.semantic_geometry_builder"},
    )
    SemanticGeometryBuilder().build(
        build_input,
        route=route,  # type: ignore[arg-type]
        run_folder=run_folder.root,
    )
    if not xao_path.is_file():
        raise FileNotFoundError(xao_path)
    return generate_mesh_from_semantic_xao(
        xao_path=xao_path,
        semantic_metadata_dir=semantic_root,
        output_dir=run_folder.root,
        stack=stack,
        ports=ports,
        model_name=model_name,
        refined_mesh_size=refined_mesh_size,
        max_mesh_size=max_mesh_size,
        fmax=fmax,
        simulation_type=simulation_type,
        driven_config=driven_config,
        eigenmode_config=eigenmode_config,
        numerical_config=numerical_config,
        refinement_config=refinement_config,
        palace_version=palace_version,
        validate_schema=validate_schema,
        absorbing_boundary=absorbing_boundary,
        problem_output_formats=problem_output_formats,
        electrostatic_config=electrostatic_config,
        terminals=terminals,
        magnetostatic_config=magnetostatic_config,
        current_sources=current_sources,
        postprocessing_config=postprocessing_config,
        boundary_postprocessing_config=boundary_postprocessing_config,
        material_overlay=material_overlay,
        write_config=write_config,
        semantic_layer_map=semantic_layer_map,
        semantic_stack_layer_map=semantic_stack_layer_map,
        show_gui=show_gui,
        high_order_elements=high_order_elements,
        high_order_order=high_order_order,
        high_order_optimize=high_order_optimize,
    )


def generate_mesh_from_semantic_xao(
    *,
    xao_path: str | Path,
    semantic_metadata_dir: str | Path,
    output_dir: str | Path,
    stack: LayerStack,
    ports: Sequence[PalacePort] = (),
    model_name: str = "palace",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    fmax: float = 100e9,
    simulation_type: Literal[
        "driven",
        "eigenmode",
        "electrostatic",
        "magnetostatic",
    ] = "electrostatic",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    numerical_config: NumericalConfig | None = None,
    refinement_config: Mapping[str, Any] | None = None,
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION,
    validate_schema: bool = True,
    absorbing_boundary: bool = True,
    problem_output_formats: Mapping[str, Any] | None = None,
    electrostatic_config: ElectrostaticConfig | None = None,
    terminals: Sequence[TerminalConfig] = (),
    magnetostatic_config: MagnetostaticConfig | None = None,
    current_sources: Sequence[CurrentSourceConfig] = (),
    postprocessing_config: dict[str, Any] | None = None,
    boundary_postprocessing_config: dict[str, Any] | None = None,
    material_overlay: Any | None = None,
    write_config: bool = True,
    semantic_layer_map: Mapping[str, str] | None = None,
    semantic_stack_layer_map: Mapping[str, str] | None = None,
    show_gui: bool = False,
    high_order_elements: bool = False,
    high_order_order: int = 2,
    high_order_optimize: bool = True,
) -> MeshResult:
    """Generate a Palace mesh from an SGB XAO contract without retopologizing it.

    The XAO and ``metadata/semantic_geometry/04_export_physical_groups.json``
    sidecar are treated as the external input contract. gsim opens the exported
    topology, maps live physical groups into its Palace ``groups`` schema,
    applies mesh sizing/config generation, and preserves SGB identity in the
    manifest metadata. It does not repair missing SGB sidecars or discover new
    route semantics from native gsim geometry.
    """
    xao = Path(xao_path)
    metadata_dir = Path(semantic_metadata_dir)
    run_folder = prepare_palace_run_folder(output_dir)
    if not xao.is_file():
        raise FileNotFoundError(xao)
    groups_sidecar = metadata_dir / "04_export_physical_groups.json"
    if not groups_sidecar.is_file():
        raise FileNotFoundError(groups_sidecar)
    if not model_name:
        raise ValueError("model_name must be non-empty")

    if xao.parent != run_folder.geometry_dir:
        shutil.copy2(xao, run_folder.geometry_dir / xao.name)
    if metadata_dir != run_folder.metadata_dir / "semantic_geometry":
        shutil.copytree(
            metadata_dir,
            run_folder.metadata_dir / "semantic_geometry",
            dirs_exist_ok=True,
        )

    records = json.loads(groups_sidecar.read_text())
    if not isinstance(records, list):
        raise TypeError("04_export_physical_groups.json must contain a list.")

    mesh_path = run_folder.root / f"{model_name}.msh"
    config_path: Path | None = None
    manifest = None
    mesh_stats: dict[str, Any] = {}
    gmsh.initialize()
    try:
        gmsh.open(str(xao))
        gmsh.option.setNumber("General.Terminal", 1 if show_gui else 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", refined_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)
        groups = _groups_from_sgb_records(
            records=records,
            semantic_layer_map=semantic_layer_map or {},
            semantic_stack_layer_map=semantic_stack_layer_map or {},
        )
        if not groups["volumes"]:
            raise ValueError(
                "SGB XAO did not expose any live volume physical groups. "
                "Check that 04_export_physical_groups.json matches the XAO file."
            )
        _setup_xao_refinement(groups, refined_mesh_size, max_mesh_size)
        gmsh.model.mesh.generate(3)
        if high_order_elements:
            gmsh.model.mesh.setOrder(high_order_order)
            if high_order_optimize:
                gmsh.model.mesh.optimize("HighOrder")
        mesh_stats = collect_mesh_stats()
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(mesh_path))
        _assert_tetra_mesh_valid(mesh_path)
        if write_config:
            config_path = generate_palace_config(
                groups=groups,
                ports=list(ports),
                port_info=[],
                stack=stack,
                output_path=run_folder.root,
                model_name=model_name,
                fmax=fmax,
                simulation_type=simulation_type,
                driven_config=driven_config,
                eigenmode_config=eigenmode_config,
                numerical_config=numerical_config,
                refinement_config=refinement_config,
                palace_version=palace_version,
                validate_schema=validate_schema,
                absorbing_boundary=absorbing_boundary,
                problem_output_formats=problem_output_formats,
                electrostatic_config=electrostatic_config,
                terminals=list(terminals),
                magnetostatic_config=magnetostatic_config,
                current_sources=list(current_sources),
                postprocessing_config=postprocessing_config,
                boundary_postprocessing_config=boundary_postprocessing_config,
                material_overlay=material_overlay,
            )
        manifest = build_mesh_manifest(groups)
    finally:
        gmsh.clear()
        gmsh.finalize()

    return MeshResult(
        mesh_path=mesh_path,
        config_path=config_path,
        port_info=[],
        mesh_stats=mesh_stats,
        groups=groups,
        output_dir=run_folder.root,
        model_name=model_name,
        fmax=fmax,
        manifest=manifest or build_mesh_manifest(groups),
    )


def _write_component_gds(component: Any, path: Path) -> None:
    """Write a component GDS, accepting path and string writer APIs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gds = getattr(component, "write_gds", None)
    if write_gds is None:
        raise TypeError("component must provide write_gds(path)")
    try:
        write_gds(path)
    except TypeError:
        write_gds(str(path))


def _component_gds_top_cell_name(component: Any, path: Path) -> str | None:
    """Resolve the exported component's top-cell name from its GDS file."""
    try:
        import gdstk
    except ImportError:
        return None

    library = gdstk.read_gds(str(path))
    cell_names = {cell.name for cell in library.cells}
    component_name = str(getattr(component, "name", ""))
    candidates = tuple(
        candidate
        for candidate in (
            component_name,
            component_name.split("$", 1)[0],
        )
        if candidate
    )
    for candidate in candidates:
        if candidate in cell_names:
            return candidate

    top_cells = [
        cell.name for cell in library.top_level() if not cell.name.startswith("$$$")
    ]
    if len(top_cells) == 1:
        return top_cells[0]
    return None


def _sgb_stack_mapping_from_gsim_inputs(
    *,
    component: Any,
    stack: LayerStack,
    gds_path: Path,
    activated_regions: Sequence[ActivatedRegion],
    terminals: Sequence[TerminalConfig],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Lower active gsim regions and layers to the SGB stack contract."""
    if not activated_regions:
        raise ValueError(
            "SGB Surface EPR route meshing requires explicit activated regions. "
            "Call activate_substrate() and activate_outer_vacuum() before mesh()."
        )
    present_layers = _gds_layers_in_file(gds_path)
    bounds = _component_bounds(component)
    solution_regions, solution_stack_layers, host_void_regions = (
        _solution_regions_from_activated(
            stack=stack,
            activated_regions=activated_regions,
            component_bounds=bounds,
        )
    )
    air_semantic_id = "AIR_ABOVE"
    terminals_by_layer: dict[str, list[Any]] = {}
    for terminal in terminals:
        if terminal.center is not None:
            terminals_by_layer.setdefault(terminal.layer, []).append(terminal)

    layer_records: list[dict[str, Any]] = []
    semantic_layer_map: dict[str, str] = {}
    for layer_name, layer in sorted(stack.layers.items()):
        if layer.layer_type not in {"conductor", "via"}:
            continue
        gds_layer = tuple(layer.gds_layer)
        if gds_layer not in present_layers:
            continue
        selectors = terminals_by_layer.get(layer_name) or (None,)
        for selector in selectors:
            semantic_id = (
                _semantic_id(f"{layer_name}@{selector.physical_label or selector.name}")
                if selector is not None
                else _semantic_id(layer_name)
            )
            semantic_layer_map[semantic_id] = layer_name
            host_void_semantic_id = _host_void_for_layer(
                layer,
                host_void_regions,
                default=air_semantic_id,
            )
            record = _sgb_layer_record(
                semantic_id=semantic_id,
                layer_name=layer_name,
                layer=layer,
                host_void_semantic_id=host_void_semantic_id,
                selector=selector,
            )
            layer_records.append(record)

    if not layer_records:
        raise ValueError(
            "No SGB conductor/via layer records matched the component GDS."
        )
    return (
        {
            "metadata": {
                "schema": "semantic_geometry_stack_v1",
                "units": "um",
                "source": str(gds_path),
                "adapter": "gsim",
            },
            "solution_regions": solution_regions,
            "layers": layer_records,
        },
        semantic_layer_map,
        solution_stack_layers | semantic_layer_map,
    )


def _solution_regions_from_activated(
    *,
    stack: LayerStack,
    activated_regions: Sequence[ActivatedRegion],
    component_bounds: tuple[float, float, float, float],
) -> tuple[dict[str, Any], dict[str, str], tuple[tuple[str, float, float], ...]]:
    """Build SGB solution-region records from explicitly activated regions."""
    regions: dict[str, Any] = {}
    stack_layers: dict[str, str] = {}
    host_void_regions: list[tuple[str, float, float]] = []
    substrate_bounds: tuple[float, float, float, float] | None = None
    substrate_zmin: float | None = None
    substrate_zmax: float | None = None

    for region in activated_regions:
        if region.role != "substrate":
            continue
        layer = stack.layers[region.layer]
        xy_bounds = _expanded_bounds(
            component_bounds,
            margin_x=region.margin_x,
            margin_y=region.margin_y,
        )
        substrate_bounds = (
            xy_bounds
            if substrate_bounds is None
            else (
                min(substrate_bounds[0], xy_bounds[0]),
                min(substrate_bounds[1], xy_bounds[1]),
                max(substrate_bounds[2], xy_bounds[2]),
                max(substrate_bounds[3], xy_bounds[3]),
            )
        )
        substrate_zmin = (
            float(layer.zmin)
            if substrate_zmin is None
            else min(substrate_zmin, float(layer.zmin))
        )
        substrate_zmax = (
            float(layer.zmax)
            if substrate_zmax is None
            else max(substrate_zmax, float(layer.zmax))
        )
        regions[region.layer] = _solution_region_record(
            semantic_id=region.layer,
            material_id=region.material or layer.material,
            bounds=xy_bounds,
            zmin=layer.zmin,
            zmax=layer.zmax,
            stack_layer=region.layer,
        )
        stack_layers[region.layer] = region.layer

    for region in activated_regions:
        if region.role != "inter_die_vacuum":
            continue
        layer = stack.layers[region.layer]
        xy_bounds = _expanded_bounds(
            component_bounds,
            margin_x=region.margin_x,
            margin_y=region.margin_y,
        )
        zmin = float(layer.zmin)
        zmax = _layer_zmax(layer)
        regions[region.layer] = _solution_region_record(
            semantic_id=region.layer,
            material_id=region.material or layer.material,
            bounds=xy_bounds,
            zmin=zmin,
            zmax=zmax,
            stack_layer=region.layer,
        )
        stack_layers[region.layer] = region.layer
        host_void_regions.append((region.layer, zmin, zmax))

    for region in activated_regions:
        if region.role != "outer_vacuum":
            continue
        layer = stack.layers[region.layer]
        if substrate_bounds is None or substrate_zmin is None or substrate_zmax is None:
            raise ValueError(
                "SGB AIR_ABOVE region requires an activated substrate region."
            )
        xy_bounds = _expanded_bounds(
            substrate_bounds,
            margin_x=region.margin_x,
            margin_y=region.margin_y,
        )
        zmin = substrate_zmax
        zmax = substrate_zmax + region.z_above
        if zmax <= zmin:
            zmax = layer.zmax
        regions["AIR_ABOVE"] = _solution_region_record(
            semantic_id="AIR_ABOVE",
            material_id=region.material or layer.material,
            bounds=xy_bounds,
            zmin=zmin,
            zmax=zmax,
            stack_layer=region.layer,
        )
        stack_layers["AIR_ABOVE"] = region.layer
        host_void_regions.append(("AIR_ABOVE", zmin, zmax))
        if region.z_below > 0.0:
            lower_zmin = substrate_zmin - region.z_below
            lower_zmax = substrate_zmin
            regions["AIR_BELOW"] = _solution_region_record(
                semantic_id="AIR_BELOW",
                material_id=region.material or layer.material,
                bounds=xy_bounds,
                zmin=lower_zmin,
                zmax=lower_zmax,
                stack_layer=region.layer,
            )
            stack_layers["AIR_BELOW"] = region.layer
            host_void_regions.append(("AIR_BELOW", lower_zmin, lower_zmax))

    if "AIR_ABOVE" not in regions:
        raise ValueError(
            "SGB Surface EPR route meshing requires an AIR_ABOVE solution region."
        )
    return regions, stack_layers, tuple(host_void_regions)


def _host_void_for_layer(
    layer: Any,
    host_void_regions: Sequence[tuple[str, float, float]],
    *,
    default: str,
) -> str:
    """Return the host void region with greatest z-overlap for a layer."""
    zmin = float(layer.zmin)
    zmax = _layer_zmax(layer)
    best_name = default
    best_overlap = 0.0
    for name, region_zmin, region_zmax in host_void_regions:
        overlap = min(zmax, region_zmax) - max(zmin, region_zmin)
        if overlap > best_overlap:
            best_name = name
            best_overlap = overlap
    return best_name


def _layer_zmax(layer: Any) -> float:
    """Return a layer's explicit or thickness-derived upper z coordinate."""
    zmax = getattr(layer, "zmax", None)
    if zmax is not None:
        return float(zmax)
    return float(layer.zmin) + float(layer.thickness)


def _solution_region_record(
    *,
    semantic_id: str,
    material_id: str,
    bounds: tuple[float, float, float, float],
    zmin: float,
    zmax: float,
    stack_layer: str,
) -> dict[str, Any]:
    """Build one SGB solution-region record."""
    xmin, ymin, xmax, ymax = bounds
    return {
        "role": "solution_region",
        "material_id": material_id,
        "geometry_kind": "domain",
        "geometry": {
            "domain": semantic_id,
            "padding_um": 0.0,
            "z_min_um": zmin,
            "z_max_um": zmax,
            "domain_bounds_um": {
                "x_min_um": xmin,
                "y_min_um": ymin,
                "x_max_um": xmax,
                "y_max_um": ymax,
            },
        },
        "metadata": {"gsim_stack_layer": stack_layer},
    }


def _sgb_layer_record(
    *,
    semantic_id: str,
    layer_name: str,
    layer: Any,
    host_void_semantic_id: str,
    selector: Any | None,
) -> dict[str, Any]:
    """Build one SGB layout-extrusion record for a conductor or via layer."""
    is_via = layer.layer_type == "via"
    geometry = {
        "z_um": float(layer.zmin),
        "thickness_um": float(layer.thickness),
        "geometry_source": "gds_polygon",
    }
    if selector is not None:
        geometry["selector_point_um"] = [
            float(selector.center[0]),
            float(selector.center[1]),
        ]
    return {
        "layer": int(layer.gds_layer[0]),
        "datatype": int(layer.gds_layer[1]),
        "semantic_id": semantic_id,
        "role": "metal",
        "material_id": layer.material,
        "priority": int(getattr(layer, "mesh_order", 0) or 0),
        "part_role": "bump_body" if is_via else "face_metal",
        "net_id": semantic_id,
        "geometry_kind": "layout_extrusion",
        "host_void_semantic_id": host_void_semantic_id,
        "geometry": geometry,
        "route_representations": (
            {
                "A": "cutout_boundary_shell",
                "B": "cutout_boundary_shell",
                "C": "material_volume",
            }
            if is_via
            else {
                "A": "surface_sheet",
                "B": "cutout_boundary_shell",
                "C": "material_volume",
            }
        ),
        "metadata": {"source_layer_name": layer_name},
    }


def _gds_layers_in_file(path: Path) -> set[tuple[int, int]]:
    """Return GDS layer/datatype pairs present in an exported layout file."""
    try:
        import gdstk
    except ImportError as error:
        raise ImportError(
            "semantic-geometry-builder requires gdstk for GDS input."
        ) from error
    library = gdstk.read_gds(str(path))
    layers: set[tuple[int, int]] = set()
    for cell in library.cells:
        for polygon in cast(Any, cell).get_polygons(apply_repetitions=True):
            layers.add((int(polygon.layer), int(polygon.datatype)))
    return layers


def _component_bounds(component: Any) -> tuple[float, float, float, float]:
    """Return a component's planar bounding box as numeric coordinates."""
    bbox_np = getattr(component, "bbox_np", None)
    if bbox_np is not None:
        bbox = bbox_np()
        return (
            float(bbox[0][0]),
            float(bbox[0][1]),
            float(bbox[1][0]),
            float(bbox[1][1]),
        )
    bbox = component.bbox()
    return (float(bbox.left), float(bbox.bottom), float(bbox.right), float(bbox.top))


def _expanded_bounds(
    bounds: tuple[float, float, float, float],
    *,
    margin_x: float,
    margin_y: float,
) -> tuple[float, float, float, float]:
    """Expand planar bounds independently along each axis."""
    xmin, ymin, xmax, ymax = bounds
    return (xmin - margin_x, ymin - margin_y, xmax + margin_x, ymax + margin_y)


def _semantic_id(value: str) -> str:
    """Normalize a nonempty value into an SGB semantic identifier."""
    normalized = _SEMANTIC_ID_RE.sub("_", value.strip()).strip("_")
    if not normalized:
        raise ValueError("Empty semantic id.")
    return normalized


def _groups_from_sgb_records(
    *,
    records: Sequence[Any],
    semantic_layer_map: Mapping[str, str],
    semantic_stack_layer_map: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Translate live SGB physical-group records into gsim mesh groups."""
    groups: dict[str, dict[str, Any]] = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
        "refinement_lines": {},
    }
    live = _live_physical_groups()
    volume_z_centers = _semantic_volume_z_centers(records=records, live=live)
    missing: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        name = str(record.get("physical_name", ""))
        dim = int(record.get("dimension", 0) or 0)
        if not name:
            if record.get("solver_use") == "solver_active":
                missing.append(f"{dim}:<missing physical_name>")
            continue
        if (dim, name) not in live:
            if record.get("solver_use") == "solver_active":
                missing.append(f"{dim}:{name}")
            continue
        phys_group, entity_tags = live[(dim, name)]
        if dim == 3 and record.get("role") == "material_volume":
            groups["volumes"][name] = _volume_group_info(
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
                semantic_stack_layer_map=semantic_stack_layer_map,
            )
        elif dim == 2:
            _add_surface_group_records(
                groups=groups,
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
                semantic_layer_map=semantic_layer_map,
                volume_z_centers=volume_z_centers,
            )
    if missing:
        raise ValueError(
            "SGB XAO is missing solver-active physical groups from sidecar: "
            + ", ".join(sorted(missing))
        )
    return groups


def _setup_xao_refinement(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Apply line refinement around active SGB boundary and port surfaces."""
    lines: set[int] = set()
    for surface_group in ("boundary_surfaces", "pec_surfaces", "port_surfaces"):
        for info in groups.get(surface_group, {}).values():
            for tag in info.get("tags", ()):
                try:
                    boundary = gmsh.model.getBoundary(
                        [(2, int(tag))],
                        combined=False,
                        oriented=False,
                        recursive=False,
                    )
                except Exception:
                    continue
                lines.update(int(btag) for bdim, btag in boundary if bdim == 1)
    if not lines:
        return
    field_id = gmsh_utils.setup_mesh_refinement(
        sorted(lines),
        refined_cellsize,
        max_cellsize,
    )
    gmsh_utils.finalize_mesh_fields([field_id])


def _assert_tetra_mesh_valid(mesh_path: Path) -> None:
    """Raise when a readable tetrahedral mesh has invalid topology."""
    try:
        import meshio
    except ImportError:
        return

    mesh = meshio.read(mesh_path)
    faces: defaultdict[tuple[int, ...], int] = defaultdict(int)
    face_indices = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    zero_volume_tets = 0
    for block in mesh.cells:
        if not block.type.startswith("tetra"):
            continue
        for tet in block.data:
            corners = tuple(int(node) for node in tet[:4])
            a, b, c, d = (mesh.points[index] for index in corners)
            ab = b - a
            ac = c - a
            ad = d - a
            volume6 = (
                ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
                - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
                + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
            )
            if volume6 == 0.0:
                zero_volume_tets += 1
            for indices in face_indices:
                faces[tuple(sorted(corners[index] for index in indices))] += 1
    bad_counts = Counter(count for count in faces.values() if count > 2)
    messages: list[str] = []
    if zero_volume_tets:
        messages.append(f"{zero_volume_tets} tetrahedra have zero volume")
    if bad_counts:
        messages.append(
            f"{sum(bad_counts.values())} triangular faces are shared by more than "
            f"two tetrahedra; incidence counts={dict(sorted(bad_counts.items()))}"
        )
    if not messages:
        return
    raise ValueError(
        f"Invalid tetrahedral mesh topology in {mesh_path}: " + "; ".join(messages)
    )


def _live_physical_groups() -> dict[tuple[int, str], tuple[int, tuple[int, ...]]]:
    """Return named physical groups currently registered in Gmsh."""
    live: dict[tuple[int, str], tuple[int, tuple[int, ...]]] = {}
    for dim, phys_group in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_group)
        if not name:
            continue
        tags = tuple(
            int(tag) for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_group)
        )
        live[(int(dim), name)] = (int(phys_group), tags)
    return live


def _volume_group_info(
    *,
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    semantic_stack_layer_map: Mapping[str, str],
) -> dict[str, Any]:
    """Build gsim volume-group metadata from one SGB physical-group record."""
    stack_layer = semantic_stack_layer_map.get(name, name)
    return {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 3,
        "physical_name": name,
        "source": "semantic_geometry_builder",
        "stack_layer": stack_layer,
        "sgb_role": record.get("role"),
        "sgb_route": record.get("route"),
        "sgb_metadata": dict(record.get("metadata", {})),
    }


def _add_surface_group_records(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    semantic_layer_map: Mapping[str, str],
    volume_z_centers: Mapping[str, float],
) -> None:
    """Classify one SGB surface record into gsim boundary and PEC groups."""
    metadata = record.get("metadata")
    source_record_ids = (
        tuple(
            value
            for value in metadata.get("source_record_ids", ())
            if isinstance(value, str) and value
        )
        if isinstance(metadata, Mapping)
        else ()
    )
    if str(record.get("role")) == "domain_boundary":
        bbox = _entities_bbox(2, entity_tags)
        groups["boundary_surfaces"][name] = {
            "phys_group": phys_group,
            "tags": list(entity_tags),
            "dim": 2,
            "source": "domain_boundary",
            "surface_epr": False,
            "representation": str(record.get("route", "")).upper(),
            "geometry_kind": "sgb_occ",
            "physical_group_attribute": phys_group,
            "sgb_physical_name": name,
            "sgb_role": record.get("role"),
            "sgb_metadata": dict(record.get("metadata", {})),
            "source_record_ids": source_record_ids,
            "bbox": bbox,
            "centroid": _bbox_centroid(bbox),
        }
        return

    parsed = _parse_surface_physical_name(
        name,
        source_record_ids=source_record_ids,
        volume_z_centers=volume_z_centers,
    )
    if parsed is None:
        return
    interface_type, source_id, face_kind, alias_name = parsed
    owner_semantic_ids = (
        tuple(_semantic_source_parts(name.split("__")[1:]))
        if interface_type == "SA"
        else ()
    )
    layer = semantic_layer_map.get(source_id, source_id)
    bbox = _entities_bbox(2, entity_tags)
    info = {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 2,
        "source": "volume_interface",
        "surface_epr": True,
        "interface_id": alias_name,
        "interface_type": interface_type,
        "source_id": source_id,
        "metal_body_id": layer if interface_type in {"MA", "MS"} else None,
        "metal_volume_id": source_id if interface_type in {"MA", "MS"} else None,
        "layer": layer if interface_type in {"MA", "MS"} else None,
        "face_kind": face_kind,
        "representation": str(record.get("route", "")).upper(),
        "geometry_kind": "sgb_occ",
        "physical_group_attribute": phys_group,
        "sgb_physical_name": name,
        "sgb_role": record.get("role"),
        "sgb_metadata": dict(record.get("metadata", {})),
        "source_record_ids": source_record_ids,
        "surface_id": source_record_ids[0] if len(source_record_ids) == 1 else None,
        "owner_semantic_ids": owner_semantic_ids,
        "bbox": bbox,
        "centroid": _bbox_centroid(bbox),
    }
    groups["boundary_surfaces"][alias_name] = info
    route = str(record.get("route", "")).upper()
    if interface_type in {"MA", "MS"} and route in {"A", "B"}:
        groups["pec_surfaces"].setdefault(
            name,
            {
                "phys_group": phys_group,
                "tags": list(entity_tags),
                "dim": 2,
                "physical_name": name,
                "source": "semantic_geometry_builder",
                "source_id": source_id,
                "layer": layer,
                "bbox": bbox,
                "representation": route,
            },
        )


def _parse_surface_physical_name(
    name: str,
    *,
    source_record_ids: tuple[str, ...] = (),
    volume_z_centers: Mapping[str, float] | None = None,
) -> tuple[str, str, str | None, str] | None:
    """Parse the current SGB interface physical-name grammar.

    SGB does not yet export explicit ``interface_type``, ``source_id``, and
    ``face_kind`` fields on final physical group records. Until that sidecar
    grows those fields, names beginning with ``MA__``, ``MS__``, ``SA__``, or
    ``MS_MA__`` are the reviewed SGB-to-gsim interface contract.
    """
    parts = name.split("__")
    if len(parts) < 3 or parts[0] not in _SURFACE_EPR_INTERFACE_TYPES:
        return None
    prefix = parts[0]
    if prefix == "MS_MA":
        source_id = parts[1]
        alias = "MS__" + "__".join(parts[1:])
        return ("MS", source_id, "bottom", alias)
    interface_type = prefix
    face_kind = next(
        (_FACE_KIND_SEGMENTS[item] for item in parts if item in _FACE_KIND_SEGMENTS),
        None,
    )
    if interface_type == "SA" and face_kind is None:
        if volume_z_centers is None:
            raise ValueError(
                "Tokenless SA physical groups require live neighbor volume z centers."
            )
        source_id, face_kind = _tokenless_sa_source_and_face_kind(
            name=name,
            source_record_ids=source_record_ids,
            volume_z_centers=volume_z_centers,
        )
        return (interface_type, source_id, face_kind, name)
    if interface_type in {"MA", "MS"}:
        source_id = parts[1]
    else:
        source_parts = _semantic_source_parts(parts[1:])
        source_id = "__".join(source_parts)
    return (interface_type, source_id, face_kind, name)


def _semantic_volume_z_centers(
    *,
    records: Sequence[Any],
    live: Mapping[tuple[int, str], tuple[int, tuple[int, ...]]],
) -> dict[str, float]:
    """Map explicitly owned SGB volume semantics to live Gmsh z centers."""
    centers: dict[str, float] = {}
    for record in records:
        if not isinstance(record, Mapping) or int(record.get("dimension", 0) or 0) != 3:
            continue
        name = record.get("physical_name")
        if not isinstance(name, str):
            continue
        live_group = live.get((3, name))
        if live_group is None:
            continue
        _, entity_tags = live_group
        bbox = _entities_bbox(3, entity_tags)
        if len(bbox) != 6:
            continue
        centers[name] = 0.5 * (bbox[2] + bbox[5])
    return centers


def _tokenless_sa_source_and_face_kind(
    *,
    name: str,
    source_record_ids: tuple[str, ...],
    volume_z_centers: Mapping[str, float],
) -> tuple[str, str]:
    """Resolve tokenless SA source and face only from SGB topology provenance."""
    parts = name.split("__")
    owner_ids = tuple(_semantic_source_parts(parts[1:]))
    if len(owner_ids) != 2 or source_record_ids != (f"SURF__{name}",):
        raise ValueError(
            "Tokenless SA physical group requires stable owner tokens and one "
            "matching SGB source_record_id."
        )
    first_center = volume_z_centers.get(owner_ids[0])
    second_center = volume_z_centers.get(owner_ids[1])
    if first_center is None or second_center is None or first_center == second_center:
        raise ValueError(
            "Tokenless SA physical group has ambiguous owner neighbor z centers."
        )
    face_kind = "top" if second_center > first_center else "bottom"
    return "__".join(owner_ids), face_kind


def _semantic_source_parts(parts: Sequence[str]) -> Sequence[str]:
    """Remove numeric and face-kind suffixes from an interface source name."""
    source_parts = list(parts)
    while source_parts and (
        source_parts[-1].isdigit() or source_parts[-1] in _FACE_KIND_SEGMENTS
    ):
        source_parts.pop()
    return tuple(source_parts)


def _entities_bbox(dim: int, entity_tags: Sequence[int]) -> list[float]:
    """Return the enclosing Gmsh bounding box for entities of one dimension."""
    bboxes = [gmsh.model.getBoundingBox(dim, tag) for tag in entity_tags]
    if not bboxes:
        return []
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        min(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
        max(bbox[4] for bbox in bboxes),
        max(bbox[5] for bbox in bboxes),
    ]


def _bbox_centroid(bbox: Sequence[float]) -> list[float]:
    """Return the centroid of a six-coordinate Gmsh bounding box."""
    if len(bbox) != 6:
        return []
    return [
        0.5 * (float(bbox[0]) + float(bbox[3])),
        0.5 * (float(bbox[1]) + float(bbox[4])),
        0.5 * (float(bbox[2]) + float(bbox[5])),
    ]
