"""Optional adapter for Semantic Geometry Builder XAO route geometry.

This module is a narrow handoff point between the standalone
``semantic_geometry_builder`` package and gsim's Palace mesh/config pipeline.
SGB is not the native gsim geometry path: callers enter this adapter only when
they explicitly request Surface EPR route A/B geometry or pass an existing SGB
XAO plus ``metadata/semantic_geometry`` sidecar directory.

The ownership split is contract-first. SGB owns route topology, physical-group
plans, XAO export, and semantic sidecars. Route A/B structured final physical-
group fields preserve exact surface, interface, face, ownership, adjacency,
conductor-component, net, and provenance authority; names are display-only.
Same-net direct metal-metal contacts are hidden topology/provenance that join
one conductor component, not independent loss surfaces. Route A uses
zero-thickness face-metal PEC sheets with finite bump shells. Route B mirrors
HFSS-style PEC assignment to faces of a finite construction metal volume: its
closed exterior boundary shell is PEC and its interior is excluded from Palace
solution domains and tetrahedra.

This adapter is Route A/B only. Native upstream meshing remains the authority
outside this explicit optional handoff. gsim owns mesh generation from the
exported contract,
Palace config generation, mesh manifests, and downstream result/report
semantics. This adapter validates that the expected SGB files are present and
fails loudly when the optional contract is unavailable or incomplete.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import gmsh

from gsim.common.polygon import fuse_polygons
from gsim.common.polygon_utils import shapely_to_klayout
from gsim.palace.mesh.config_generator import collect_mesh_stats, generate_palace_config
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.manifest import build_mesh_manifest
from gsim.palace.mesh.postprocessing import build_terminal_index_map_from_manifest
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
_STRUCTURED_INTERFACE_TYPES = {"MA", "MS", "SA", "MS_MA"}
_FACE_KIND_SEGMENTS = {"TOP": "top", "BOTTOM": "bottom", "SIDEWALL": "sidewall"}
_STRUCTURED_FACE_KINDS = {
    **_FACE_KIND_SEGMENTS,
    "INTERFACE": "interface",
    "SHEET_CONTACT_CAP": "sheet_contact_cap",
}
_SEMANTIC_ID_RE = re.compile(r"[^A-Za-z0-9_@]+")
_SGB_PART_ROLES = {"face_metal", "contact_pad", "bump_body"}
_STRUCTURED_SURFACE_FIELDS = frozenset(
    (
        "representation",
        "surface_id",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "adjacent_solution_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
        "source_provenance",
        "physical_attribute",
    )
)


def _optional_string(value: Any) -> str | None:
    """Return a trimmed non-empty string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_nonempty_string_sequence(value: Any) -> bool:
    """Return true when a value is a non-empty list/tuple of strings."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _has_structured_record_marker(record: Mapping[str, Any]) -> bool:
    """Return whether a record claims the required SGB structure."""
    return any(field in record for field in _STRUCTURED_SURFACE_FIELDS)


def _structured_record_kind(
    record: Mapping[str, Any],
    *,
    structured_required: bool = False,
) -> str | None:
    """Classify a marked SGB record without applying surface rules to volumes."""
    if not structured_required and not _has_structured_record_marker(record):
        return None
    if (
        int(record.get("dimension", 2) or 2) == 3
        or record.get("role") == "material_volume"
    ):
        return "volume"
    return "surface"


def _has_structured_surface_marker(
    record: Mapping[str, Any],
    *,
    structured_required: bool = False,
) -> bool:
    """Return whether a record is a current structured SGB surface."""
    return (
        _structured_record_kind(record, structured_required=structured_required)
        == "surface"
    )


def generate_mesh_from_semantic_geometry_builder(
    *,
    component: Any,
    stack: LayerStack,
    ports: Sequence[PalacePort] = (),
    output_dir: str | Path,
    route: Literal["A", "B"],
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
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION,
    validate_schema: bool = True,
    absorbing_boundary: bool = True,
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

    This entrypoint is used only for explicit Surface EPR route A/B requests.
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
            "Surface EPR A/B route meshing requires the "
            "semantic-geometry-builder package installed with gsim."
        )
        raise ImportError(msg) from error

    _write_component_gds(component, gds_path)
    if route in {"A", "B"}:
        _write_sgb_input_gds(component=component, stack=stack, gds_path=gds_path)
    top_cell_name = _component_gds_top_cell_name(component, gds_path)
    stack_mapping, semantic_layer_map, semantic_stack_layer_map = (
        _sgb_stack_mapping_from_gsim_inputs(
            component=component,
            stack=stack,
            gds_path=gds_path,
            activated_regions=activated_regions,
            terminals=terminals,
            ports=ports,
            route=route,
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
        palace_version=palace_version,
        validate_schema=validate_schema,
        absorbing_boundary=absorbing_boundary,
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
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION,
    validate_schema: bool = True,
    absorbing_boundary: bool = True,
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
        _sgb_records_route_context(records)
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
        _embed_port_sheet_surfaces(groups)
        _setup_xao_refinement(groups, refined_mesh_size, max_mesh_size)
        if any(
            info.get("type") == "lumped_sheet"
            for info in groups["port_surfaces"].values()
        ):
            # Gmsh's default/HXT path rejects embedded internal sheet PLCs.
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)
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
                palace_version=palace_version,
                validate_schema=validate_schema,
                absorbing_boundary=absorbing_boundary,
                electrostatic_config=electrostatic_config,
                terminals=list(terminals),
                magnetostatic_config=magnetostatic_config,
                current_sources=list(current_sources),
                postprocessing_config=postprocessing_config,
                boundary_postprocessing_config=boundary_postprocessing_config,
                material_overlay=material_overlay,
            )
        manifest = build_mesh_manifest(groups)
        manifest.write_json(run_folder.mesh_manifest_path)
        if terminals:
            config = json.loads(config_path.read_text()) if config_path else {}
            terminal_entries = config.get("Boundaries", {}).get("Terminal", [])
            index_map = build_terminal_index_map_from_manifest(
                manifest,
                terminal_entries,
                terminal_names=tuple(terminal.name for terminal in terminals),
            )
            index_map.write_json(run_folder.index_map_path)
    finally:
        gmsh.clear()
        gmsh.finalize()

    return MeshResult(
        mesh_path=mesh_path,
        config_path=config_path,
        port_info=_port_information_from_sgb_groups(groups),
        mesh_stats=mesh_stats,
        groups=groups,
        output_dir=run_folder.root,
        model_name=model_name,
        fmax=fmax,
        manifest=manifest or build_mesh_manifest(groups),
    )


def _port_information_from_sgb_groups(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Preserve layout-sheet port labels in the canonical result sidecar."""
    rows: list[dict[str, Any]] = []
    for key, info in groups.get("port_surfaces", {}).items():
        if info.get("type") != "lumped_sheet":
            continue
        attribute = info.get("physical_attribute")
        if not isinstance(attribute, Mapping):
            raise TypeError(f"SGB layout sheet {key!r} needs physical_attribute.")
        rows.append(
            {
                "portnumber": attribute["port_index"],
                "name": attribute["port_name"],
                "type": "lumped_sheet",
                "direction": list(info["direction"]),
                "physical_name": info["physical_name"],
                "attributes": [info["phys_group"]],
                "source_layer": attribute["source_layer"],
                "target_layer": attribute["target_layer"],
            }
        )
    return rows


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


def _write_sgb_input_gds(*, component: Any, stack: LayerStack, gds_path: Path) -> None:
    """Replace SGB conductor tuples with their evaluated GDS expressions."""
    try:
        import gdstk
    except ImportError as error:
        raise ImportError(
            "semantic-geometry-builder requires gdstk for GDS input."
        ) from error

    expressions: dict[tuple[int, int], Any] = {}
    for layer_name, layer in sorted(stack.layers.items()):
        if layer.exclude_from_simulation:
            continue
        if layer.layer_type not in {"conductor", "via"}:
            continue
        if getattr(layer, "part_role", None) not in _SGB_PART_ROLES:
            continue
        expression = getattr(layer, "_source_expression", None)
        if expression is None:
            raise ValueError(
                f"SGB typed layer {layer_name!r} has no authored source expression."
            )
        gds_layer = tuple(layer.gds_layer)
        previous = expressions.get(gds_layer)
        if previous is not None and previous != expression:
            raise ValueError(
                "SGB typed layers target the same GDS tuple with different "
                f"source expressions: {gds_layer!r}."
            )
        expressions[gds_layer] = expression

    library = gdstk.read_gds(str(gds_path))
    top_cell = _select_component_gds_top_cell(component, library)
    flattened = top_cell.copy(top_cell.name, deep_copy=True)
    flattened.flatten()
    for layer, datatype in expressions:
        flattened.remove(
            *[
                polygon
                for polygon in flattened.polygons
                if (int(polygon.layer), int(polygon.datatype)) == (layer, datatype)
            ]
        )
    for (layer, datatype), expression in expressions.items():
        fused = fuse_polygons(component, expression)
        for island in _fused_polygon_islands(fused):
            flattened.add(
                *_gdstk_polygons_from_island(
                    island=island,
                    layer=layer,
                    datatype=datatype,
                )
            )
    output = gdstk.Library(unit=library.unit, precision=library.precision)
    output.add(flattened)
    output.write_gds(str(gds_path))


def _select_component_gds_top_cell(component: Any, library: Any) -> Any:
    """Select one deterministic top cell; SGB input never retains hierarchy."""
    cells_by_name = {cell.name: cell for cell in library.cells}
    component_name = str(getattr(component, "name", ""))
    for candidate in (component_name, component_name.split("$", 1)[0]):
        if candidate and candidate in cells_by_name:
            return cells_by_name[candidate]
    top_cells = sorted(
        (cell for cell in library.top_level() if not cell.name.startswith("$$$")),
        key=lambda cell: cell.name,
    )
    if len(top_cells) != 1:
        raise ValueError("SGB GDS export needs exactly one deterministic top cell.")
    return top_cells[0]


def _fused_polygon_islands(geometry: Any) -> tuple[Any, ...]:
    """Return fused connected Polygon islands in a stable order."""
    if getattr(geometry, "is_empty", True):
        return ()
    if getattr(geometry, "geom_type", None) == "Polygon":
        islands = (geometry,)
    elif getattr(geometry, "geom_type", None) == "MultiPolygon":
        islands = tuple(geometry.geoms)
    else:
        raise ValueError(
            "SGB source expression did not evaluate to polygonal geometry."
        )
    return tuple(sorted(islands, key=lambda polygon: (*polygon.bounds, polygon.wkb)))


def _gdstk_polygons_from_island(
    *, island: Any, layer: int, datatype: int
) -> tuple[Any, ...]:
    """Convert one fused Shapely island, including holes, to GDS polygons."""
    import gdstk

    # Validate through the shared KLayout conversion before GDS lowering.
    if shapely_to_klayout(island) is None:
        raise ValueError("SGB source expression produced an invalid polygon island.")
    exterior = list(island.exterior.coords[:-1])
    holes = [list(hole.coords[:-1]) for hole in island.interiors]
    if not holes:
        return (gdstk.Polygon(exterior, layer=layer, datatype=datatype),)
    return tuple(
        gdstk.boolean(
            gdstk.Polygon(exterior, layer=layer, datatype=datatype),
            [gdstk.Polygon(hole, layer=layer, datatype=datatype) for hole in holes],
            "not",
            layer=layer,
            datatype=datatype,
        )
    )


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
    ports: Sequence[PalacePort],
    route: Literal["A", "B"],
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
        if terminal.net_id is None:
            if terminal.layer is None:
                raise ValueError("SGB selector terminal requires a layer.")
            terminals_by_layer.setdefault(terminal.layer, []).append(terminal)

    layer_records: list[dict[str, Any]] = []
    semantic_layer_map: dict[str, str] = {}
    selector_source_ids: set[str] = set()
    for layer_name, layer in sorted(stack.layers.items()):
        if layer.exclude_from_simulation:
            continue
        if layer.layer_type not in {"conductor", "via"}:
            continue
        gds_layer = tuple(layer.gds_layer)
        if gds_layer not in present_layers:
            continue
        part_role = getattr(layer, "part_role", None)
        if part_role not in _SGB_PART_ROLES:
            raise ValueError(
                "SGB route requires explicit typed part_role for "
                f"conductor/via layer {layer_name!r}."
            )
        selectors = terminals_by_layer.get(layer_name)
        selector_points: list[tuple[float, float]] = []
        if selectors:
            _validate_sgb_terminal_islands(
                component=component,
                layer_name=layer_name,
                layer=layer,
                selectors=selectors,
            )
            for selector in selectors:
                if selector.center is None:
                    raise ValueError(
                        f"SGB terminal {selector.name!r} on {layer_name!r} "
                        "needs center."
                    )
                selector_id = _selector_source_id(layer_name, selector)
                if selector_id in selector_source_ids:
                    raise ValueError(
                        f"SGB terminal {selector_id!r} appears more than once "
                        "across the same conductor layer."
                    )
                selector_source_ids.add(selector_id)
                selector_points.append(selector.center)
                semantic_layer_map[selector_id] = layer_name
                host_void_semantic_id = _host_void_for_layer(
                    layer,
                    host_void_regions,
                    default=air_semantic_id,
                )
                layer_records.append(
                    _sgb_layer_record(
                        route=route,
                        semantic_id=selector_id,
                        layer_name=layer_name,
                        layer=layer,
                        host_void_semantic_id=host_void_semantic_id,
                        selector=selector,
                        is_residual=False,
                    )
                )
            host_void_semantic_id = _host_void_for_layer(
                layer,
                host_void_regions,
                default=air_semantic_id,
            )
            residual_id = _semantic_id(layer_name)
            semantic_layer_map[residual_id] = layer_name
            layer_records.append(
                _sgb_layer_record(
                    route=route,
                    semantic_id=residual_id,
                    layer_name=layer_name,
                    layer=layer,
                    host_void_semantic_id=host_void_semantic_id,
                    selector=None,
                    is_residual=True,
                    exclude_selector_points_um=selector_points,
                )
            )
        else:
            host_void_semantic_id = _host_void_for_layer(
                layer,
                host_void_regions,
                default=air_semantic_id,
            )
            residual_id = _semantic_id(layer_name)
            semantic_layer_map[residual_id] = layer_name
            layer_records.append(
                _sgb_layer_record(
                    route=route,
                    semantic_id=residual_id,
                    layer_name=layer_name,
                    layer=layer,
                    host_void_semantic_id=host_void_semantic_id,
                    selector=None,
                    is_residual=True,
                )
            )

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
                "port_sheet_source_layers": _port_sheet_source_layers(ports),
            },
            "solution_regions": solution_regions,
            "layers": layer_records,
        },
        semantic_layer_map,
        solution_stack_layers | semantic_layer_map,
    )


def _port_sheet_source_layers(ports: Sequence[PalacePort]) -> list[dict[str, Any]]:
    """Lower exact gdsfactory port-layer ownership for SGB Route A/B sheets."""
    from gsim.palace.ports.config import PortGeometry, PortType

    records: list[dict[str, Any]] = []
    used_layers: set[tuple[int, int]] = set()
    for port_index, port in enumerate(ports, start=1):
        if port.sheet_layer is None:
            continue
        if (
            port.port_type != PortType.LUMPED
            or port.geometry != PortGeometry.INPLANE
            or port.multi_element
            or port.layer is None
        ):
            raise ValueError(
                f"layout_sheet port '{port.name}' must be one single inplane "
                "lumped port"
            )
        layer, datatype = port.sheet_layer
        if isinstance(layer, bool) or isinstance(datatype, bool):
            raise TypeError(f"layout_sheet port '{port.name}' has invalid GDS layer")
        source_layer = (int(layer), int(datatype))
        if source_layer in used_layers:
            raise ValueError(
                "layout_sheet ports must use distinct gdsfactory port layers: "
                f"{source_layer!r}"
            )
        used_layers.add(source_layer)
        orientation = float(port.orientation)
        if not math.isfinite(orientation):
            raise ValueError(
                f"layout_sheet port '{port.name}' has non-finite orientation"
            )
        direction = _orientation_direction(orientation)
        records.append(
            {
                "layer": source_layer[0],
                "datatype": source_layer[1],
                "name": port.name,
                "port_index": port_index,
                "target_layer": port.layer,
                "direction": direction,
                "orientation_degrees": orientation,
                "direction_sign_convention": "gdsfactory_port_orientation_outward",
                "source": "palace_lumped_port_sheet",
            }
        )
    return records


def _orientation_direction(orientation: float) -> tuple[float, float, float]:
    """Return a stable normalized XY current direction from port orientation."""
    angle = math.radians(orientation)
    x = round(math.cos(angle), 15)
    y = round(math.sin(angle), 15)
    length = math.hypot(x, y)
    if not math.isfinite(length) or length == 0.0:
        raise ValueError("layout_sheet port orientation has no finite XY direction")
    return (x / length, y / length, 0.0)


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
    route: Literal["A", "B"],
    semantic_id: str,
    layer_name: str,
    layer: Any,
    host_void_semantic_id: str,
    selector: Any | None,
    is_residual: bool,
    exclude_selector_points_um: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """Build one SGB layout-extrusion record for a conductor or via layer.

    The authored layer net is retained for residual/unselected islands. A
    terminal selector owns only its selected island, overrides that record's
    net with the deterministic terminal identity, and clears a layer-default
    equipotential so it cannot leak from the residual component.
    """
    part_role = getattr(layer, "part_role", None)
    if part_role not in _SGB_PART_ROLES:
        raise ValueError(f"SGB requires explicit part_role for layer {layer_name!r}.")
    is_finite_contact = part_role in {"contact_pad", "bump_body"}
    if route not in {"A", "B"}:
        raise ValueError("Structured selector lowering is limited to Route A/B.")
    geometry = {
        "z_um": float(layer.zmin),
        "thickness_um": float(layer.thickness),
        "geometry_source": "gds_polygon",
        # SGB must fuse edge-connected GDS fragments before selector ownership.
        # This record builder is reached only from the Route A/B lowering path.
        "route_ab_fused_selector_mode": True,
    }
    if is_residual:
        geometry["split_polygons_as_entities"] = True
    if selector is not None:
        geometry["selector_point_um"] = [
            float(selector.center[0]),
            float(selector.center[1]),
        ]
    if exclude_selector_points_um:
        geometry["exclude_selector_points_um"] = [
            [float(point[0]), float(point[1])] for point in exclude_selector_points_um
        ]
    semantic_net_id = (
        _selector_source_id(layer_name, selector) if selector is not None else None
    )
    semantic_equipotential_id = None if selector is not None else layer.equipotential_id
    return {
        "layer": int(layer.gds_layer[0]),
        "datatype": int(layer.gds_layer[1]),
        "semantic_id": semantic_id,
        "role": "metal",
        "material_id": layer.material,
        "priority": int(getattr(layer, "mesh_order", 0) or 0),
        "part_role": part_role,
        "net_id": semantic_net_id if semantic_net_id is not None else layer.net_id,
        "equipotential_id": semantic_equipotential_id,
        "attached_face_metal_semantic_id": layer.attached_face_metal_semantic_id,
        "geometry_kind": "layout_extrusion",
        "host_void_semantic_id": host_void_semantic_id,
        "geometry": geometry,
        "route_representations": (
            {
                "A": "cutout_boundary_shell",
                "B": "cutout_boundary_shell",
            }
            if is_finite_contact
            else {
                "A": "surface_sheet",
                "B": "cutout_boundary_shell",
            }
        ),
        "metadata": {
            "source_layer_name": layer_name,
            "semantic_group_id": semantic_id,
            "equipotential_id": semantic_equipotential_id,
        },
    }


def _selector_source_id(layer_name: str, selector: Any) -> str:
    """Return the deterministic SGB terminal semantic/net identifier."""
    label = getattr(selector, "physical_label", None) or getattr(selector, "name", None)
    if not isinstance(label, str) or not label.strip():
        raise ValueError(
            f"SGB terminal on {layer_name!r} needs a name or physical_label."
        )
    return _semantic_id(f"{layer_name}@{label}")


def _validate_sgb_terminal_islands(
    *,
    component: Any,
    layer_name: str,
    layer: Any,
    selectors: Sequence[Any],
) -> None:
    """Require every Route A/B terminal point to own one distinct fused island."""
    from shapely.geometry import Point

    expression = getattr(layer, "_source_expression", None)
    if expression is None:
        raise ValueError(f"SGB typed layer {layer_name!r} has no source expression.")
    islands = _fused_polygon_islands(fuse_polygons(component, expression))
    selected_islands: set[int] = set()
    for selector in selectors:
        if selector.center is None:
            raise ValueError(f"SGB terminal {selector.name!r} needs center.")
        point = Point(selector.center)
        matches = [
            index for index, island in enumerate(islands) if island.covers(point)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"SGB terminal {selector.name!r} on {layer_name!r} must select "
                "exactly one fused conductor island."
            )
        if matches[0] in selected_islands:
            raise ValueError(
                f"SGB terminal {selector.name!r} on {layer_name!r} shares a "
                "fused conductor island with another terminal."
            )
        selected_islands.add(matches[0])


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
    structured_volumes = _structured_solution_volume_records(
        records=records,
        structured_required=True,
    )
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
        structured_kind = _structured_record_kind(
            record,
            structured_required=True,
        )
        if dim == 3 and structured_kind == "surface":
            raise ValueError(f"SGB volume {name!r} has surface structured metadata.")
        if dim == 2 and structured_kind == "volume":
            raise ValueError(f"SGB surface {name!r} has volume structured metadata.")
        if dim == 3 and record.get("role") == "material_volume":
            groups["volumes"][name] = _volume_group_info(
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
                semantic_stack_layer_map=semantic_stack_layer_map,
                structured_required=True,
            )
        elif dim == 2:
            if record.get("role") == "lumped_port":
                _add_lumped_port_sheet_group(
                    groups=groups,
                    name=name,
                    phys_group=phys_group,
                    entity_tags=entity_tags,
                    record=record,
                )
                continue
            _add_surface_group_records(
                groups=groups,
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
                semantic_layer_map=semantic_layer_map,
                structured_volumes=structured_volumes,
                volume_z_centers=volume_z_centers,
                structured_required=True,
            )
    if missing:
        raise ValueError(
            "SGB XAO is missing solver-active physical groups from sidecar: "
            + ", ".join(sorted(missing))
        )
    return groups


def _add_lumped_port_sheet_group(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
) -> None:
    """Classify an SGB Route A/B lumped-port sheet before interface parsing."""
    if (
        record.get("interface_type") != "lumped_port"
        or record.get("face_kind") != "sheet"
        or record.get("representation") != "lumped_port_sheet"
        or record.get("solver_use") != "solver_active"
        or record.get("route") not in {"A", "B"}
        or _optional_string(record.get("surface_id")) is None
        or not isinstance(record.get("source_provenance"), Mapping)
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} has invalid structure.")
    owners = record.get("owner_semantic_ids")
    adjacent = record.get("adjacent_solution_volume_ids")
    if (
        not isinstance(owners, (list, tuple))
        or len(owners) != 2
        or not all(isinstance(value, str) and value for value in owners)
        or len(set(owners)) != 2
        or not isinstance(adjacent, (list, tuple))
        or len(adjacent) not in {1, 2}
        or not all(isinstance(value, str) and value for value in adjacent)
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} needs owners and solution-volume "
            "provenance."
        )
    owner_ids = tuple(str(value) for value in owners)
    adjacent_ids = tuple(str(value) for value in adjacent)
    attribute = record.get("physical_attribute")
    if not isinstance(attribute, Mapping):
        raise TypeError(f"SGB lumped-port sheet {name!r} needs physical_attribute.")
    port_index = attribute.get("port_index")
    port_name = _optional_string(attribute.get("port_name"))
    target_layer = _optional_string(attribute.get("target_layer"))
    embedded_volume_id = _optional_string(attribute.get("embedded_volume_id"))
    owner_provenance = attribute.get("owner_provenance")
    if (
        isinstance(port_index, bool)
        or not isinstance(port_index, int)
        or port_index < 1
        or port_name is None
        or target_layer is None
        or embedded_volume_id is None
        or not isinstance(owner_provenance, (list, tuple))
        or len(owner_provenance) != 2
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} has incomplete attributes.")
    provenance_owner_ids = tuple(
        _optional_string(item.get("semantic_id")) if isinstance(item, Mapping) else None
        for item in owner_provenance
    )
    if provenance_owner_ids != owner_ids:
        raise ValueError(
            f"SGB lumped-port sheet {name!r} owner provenance is inconsistent."
        )
    for item in owner_provenance:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"SGB lumped-port sheet {name!r} owner provenance must be mappings."
            )
        component_id = _optional_string(item.get("conductor_component_id"))
        net_id = item.get("net_id")
        equipotential_id = item.get("equipotential_id")
        if (
            component_id is None
            or (net_id is not None and _optional_string(net_id) is None)
            or (
                equipotential_id is not None
                and _optional_string(equipotential_id) is None
            )
        ):
            raise ValueError(
                f"SGB lumped-port sheet {name!r} has incomplete component provenance."
            )
    route = record["route"]
    if (
        route == "A"
        and (len(adjacent_ids) != 2 or embedded_volume_id not in adjacent_ids)
    ) or (route == "B" and adjacent_ids != (embedded_volume_id,)):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} has invalid Route {route} adjacency."
        )
    direction = _normalized_port_sheet_direction(attribute.get("direction"), name)
    source_provenance = record["source_provenance"]
    provenance_direction = _normalized_port_sheet_direction(
        source_provenance.get("direction"), name
    )
    if (
        source_provenance.get("source_name") != port_name
        or source_provenance.get("port_index") != port_index
        or source_provenance.get("source_layer") != attribute.get("source_layer")
        or source_provenance.get("target_layer") != target_layer
        or provenance_direction != direction
        or _optional_string(source_provenance.get("direction_sign_convention")) is None
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} source provenance is inconsistent."
        )
    key = f"P{port_index}"
    if key in groups["port_surfaces"]:
        raise ValueError(f"Duplicate SGB lumped-port sheet index {port_index}.")
    bbox = _entities_bbox(2, entity_tags)
    groups["port_surfaces"][key] = {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 2,
        "type": "lumped_sheet",
        "direction": direction,
        "embedded_volume_id": embedded_volume_id,
        "source": "semantic_geometry_builder",
        "surface_epr": True,
        "sgb_record": "final_physical_group",
        "sgb_route": record["route"],
        "representation": record["representation"],
        "geometry_kind": "sgb_occ",
        "physical_group_attribute": phys_group,
        "physical_name": name,
        "sgb_role": record["role"],
        "sgb_metadata": dict(record.get("metadata", {})),
        "surface_id": record.get("surface_id"),
        "owner_semantic_ids": owner_ids,
        "adjacent_solution_volume_ids": adjacent_ids,
        "source_provenance": record.get("source_provenance"),
        "physical_attribute": dict(attribute),
        "bbox": bbox,
        "centroid": _bbox_centroid(bbox),
    }


def _normalized_port_sheet_direction(
    value: Any, name: str
) -> tuple[float, float, float]:
    """Validate the exact numeric Route A/B current direction."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} needs a 3D direction.")
    direction = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in direction) or not math.isclose(
        direction[2], 0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} direction must be finite and in-plane."
        )
    length = math.hypot(direction[0], direction[1])
    if (
        not math.isfinite(length)
        or length == 0.0
        or not math.isclose(length, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} direction must be normalized and nonzero."
        )
    return (direction[0], direction[1], 0.0)


def _embed_port_sheet_surfaces(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Embed each Route A/B port sheet in its SGB-owned solution volume."""
    volumes = groups.get("volumes", {})
    for port_key, port_info in groups.get("port_surfaces", {}).items():
        if port_info.get("type") != "lumped_sheet":
            continue
        if port_info.get("sgb_route") == "A":
            continue
        volume_id = _optional_string(port_info.get("embedded_volume_id"))
        volume = volumes.get(volume_id) if volume_id is not None else None
        if not isinstance(volume, Mapping) or not volume.get("tags"):
            raise ValueError(
                f"SGB lumped-port sheet {port_key!r} has no live embedded volume "
                f"{volume_id!r}."
            )
        surface_tags = [int(tag) for tag in port_info.get("tags", ())]
        volume_tags = [int(tag) for tag in volume.get("tags", ())]
        if not surface_tags or not volume_tags:
            raise ValueError(f"SGB lumped-port sheet {port_key!r} has no live tags.")
        for volume_tag in volume_tags:
            gmsh.model.mesh.embed(2, surface_tags, 3, volume_tag)


def _sgb_records_route_context(records: Sequence[Any]) -> Literal["A", "B"]:
    """Require one explicit structured SGB Route A/B context."""
    routes: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        route = _optional_string(record.get("route"))
        if route not in {"A", "B"}:
            raise ValueError(
                "SGB physical-group records require explicit route A or B."
            )
        routes.add(route)
    if len(routes) != 1:
        raise ValueError("SGB physical-group sidecar must have one route context.")
    return cast(Literal["A", "B"], routes.pop())


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
    structured_required: bool,
) -> dict[str, Any]:
    """Build gsim volume-group metadata from one SGB physical-group record."""
    if (
        _structured_record_kind(record, structured_required=structured_required)
        == "volume"
    ):
        _validate_structured_volume_record(name, record)
    stack_layer = semantic_stack_layer_map.get(name, name)
    info = {
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
    if (
        _structured_record_kind(record, structured_required=structured_required)
        == "volume"
    ):
        info.update(
            {
                "sgb_record": "final_physical_group",
                "representation": record["representation"],
                "source_provenance": record["source_provenance"],
                "physical_attribute": record["physical_attribute"],
                "interface_type": record.get("interface_type"),
                "face_kind": record.get("face_kind"),
                "contact_kind": record.get("contact_kind"),
                "conductor_component_id": record.get("conductor_component_id"),
                "net_id": record.get("net_id"),
                "equipotential_id": record.get("equipotential_id"),
                "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
                "adjacent_solution_volume_ids": tuple(
                    record.get("adjacent_solution_volume_ids", ())
                ),
            }
        )
    return info


def _structured_solution_volume_records(
    *,
    records: Sequence[Any],
    structured_required: bool,
) -> dict[str, Mapping[str, Any]]:
    """Index current structured solution volumes by their exact sidecar ID."""
    volumes: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        if (
            int(record.get("dimension", 0) or 0) != 3
            or record.get("role") != "material_volume"
            or _structured_record_kind(record, structured_required=structured_required)
            != "volume"
        ):
            continue
        name = _optional_string(record.get("physical_name"))
        if name is None:
            raise ValueError("Structured SGB solution volume has empty physical_name.")
        _validate_structured_volume_record(name, record)
        if name in volumes:
            raise ValueError(f"Duplicate structured SGB solution volume {name!r}.")
        volumes[name] = record
    return volumes


def _add_surface_group_records(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    semantic_layer_map: Mapping[str, str],
    structured_volumes: Mapping[str, Mapping[str, Any]],
    volume_z_centers: Mapping[str, float],
    structured_required: bool,
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
        if _has_structured_surface_marker(
            record, structured_required=structured_required
        ):
            _validate_structured_domain_boundary_record(name, record)
        bbox = _entities_bbox(2, entity_tags)
        info = {
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
        if _has_structured_surface_marker(
            record, structured_required=structured_required
        ):
            info.update(
                {
                    "sgb_record": "final_physical_group",
                    "solver_use": record["solver_use"],
                    "surface_id": record["surface_id"],
                    "representation": record["representation"],
                    "source_provenance": record["source_provenance"],
                    "physical_attribute": record["physical_attribute"],
                    "owner_semantic_ids": tuple(record["owner_semantic_ids"]),
                    "adjacent_solution_volume_ids": tuple(
                        record["adjacent_solution_volume_ids"]
                    ),
                    "interface_type": record["interface_type"],
                    "face_kind": record["face_kind"],
                    "contact_kind": record["contact_kind"],
                }
            )
        groups["boundary_surfaces"][name] = info
        return

    parsed = _parse_surface_physical_name(
        name,
        record=record,
        source_record_ids=source_record_ids,
        volume_z_centers=volume_z_centers,
        structured_required=structured_required,
    )
    if parsed is None:
        if record.get("solver_use") == "solver_active":
            raise ValueError(
                "SGB XAO is missing or invalid structured surface metadata "
                f"for {name!r}."
            )
        return
    structured = _has_structured_surface_marker(
        record,
        structured_required=structured_required,
    )
    if structured and parsed[0] == "MS_MA":
        parsed_records = _structured_ms_ma_surface_records(
            name=name,
            record=record,
            volume_z_centers=volume_z_centers,
            structured_volumes=structured_volumes,
            sheet_z_center=_surface_z_center(entity_tags),
        )
    elif structured and parsed[0] == "SA":
        parsed_records = (
            _structured_sa_surface_record(
                name=name,
                record=record,
                volume_z_centers=volume_z_centers,
                structured_volumes=structured_volumes,
            ),
        )
    else:
        parsed_records = (parsed,)
    for interface_type, source_id, face_kind, alias_name in parsed_records:
        _add_parsed_surface_group_record(
            groups=groups,
            name=name,
            phys_group=phys_group,
            entity_tags=entity_tags,
            record=record,
            semantic_layer_map=semantic_layer_map,
            structured=structured,
            interface_type=interface_type,
            source_id=source_id,
            face_kind=face_kind,
            alias_name=alias_name,
            source_record_ids=source_record_ids,
        )


def _add_parsed_surface_group_record(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    semantic_layer_map: Mapping[str, str],
    structured: bool,
    interface_type: str,
    source_id: str,
    face_kind: str | None,
    alias_name: str,
    source_record_ids: tuple[str, ...],
) -> None:
    """Store one logical interface record for an SGB surface physical group."""
    owner_semantic_ids = (
        tuple(record.get("owner_semantic_ids", ()))
        if structured
        else (
            tuple(_semantic_source_parts(name.split("__")[1:]))
            if interface_type == "SA"
            else ()
        )
    )
    layer = (
        semantic_layer_map.get(source_id, source_id)
        if interface_type in {"MA", "MS"} and source_id
        else None
    )
    bbox = _entities_bbox(2, entity_tags)
    info = {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 2,
        "source": "volume_interface",
        "surface_epr": True,
        "interface_id": alias_name,
        "interface_type": interface_type,
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use"),
        "conductor_component_id": _optional_string(
            record.get("conductor_component_id")
        ),
        "net_id": _optional_string(record.get("net_id")),
        "equipotential_id": _optional_string(record.get("equipotential_id")),
        "physical_attribute": record.get("physical_attribute"),
        "source_id": source_id,
        "face_kind": face_kind,
        "contact_kind": _optional_string(record.get("contact_kind")),
        "representation": str(record.get("route", "")).upper(),
        "sgb_representation": record.get("representation"),
        "geometry_kind": "sgb_occ",
        "physical_group_attribute": phys_group,
        "sgb_physical_name": name,
        "sgb_role": record.get("role"),
        "sgb_metadata": dict(record.get("metadata", {})),
        "source_record_ids": source_record_ids,
        "surface_id": _optional_string(record.get("surface_id"))
        or (source_record_ids[0] if len(source_record_ids) == 1 else None),
        "owner_semantic_ids": owner_semantic_ids,
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        )
        if structured
        else (),
        "source_provenance": record.get("source_provenance"),
        "bbox": bbox,
        "centroid": _bbox_centroid(bbox),
    }
    if interface_type in {"MA", "MS"}:
        info.update(
            {
                "metal_body_id": layer,
                "metal_volume_id": source_id,
                "layer": layer,
            }
        )
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


def _structured_ms_ma_surface_records(
    *,
    name: str,
    record: Mapping[str, Any],
    volume_z_centers: Mapping[str, float],
    structured_volumes: Mapping[str, Mapping[str, Any]],
    sheet_z_center: float | None,
) -> tuple[tuple[str, str, str, str], tuple[str, str, str, str]]:
    """Split one structured zero-thickness PEC sheet into MS and MA logic."""
    substrate_id, other_id = _structured_dielectric_vacuum_adjacency(
        surface_type="MS_MA",
        name=name,
        record=record,
        structured_volumes=structured_volumes,
    )
    substrate_center = volume_z_centers.get(substrate_id)
    other_center = volume_z_centers.get(other_id)
    if (
        sheet_z_center is None
        or substrate_center is None
        or other_center is None
        or not all(
            math.isfinite(value)
            for value in (sheet_z_center, substrate_center, other_center)
        )
    ):
        raise ValueError(
            f"SGB MS_MA surface {name!r} needs finite adjacent-volume topology."
        )
    if (
        substrate_center in (other_center, sheet_z_center)
        or other_center == sheet_z_center
        or (substrate_center - sheet_z_center) * (other_center - sheet_z_center) >= 0
    ):
        raise ValueError(
            f"SGB MS_MA surface {name!r} has ambiguous adjacent-volume topology."
        )
    ms_face = "bottom" if substrate_center < other_center else "top"
    ma_face = "top" if ms_face == "bottom" else "bottom"
    source_id = _structured_conductor_source_layer(name, record)
    surface_id = str(record["surface_id"])
    return (
        ("MS", source_id, ms_face, f"{surface_id}__MS__{ms_face.upper()}"),
        ("MA", source_id, ma_face, f"{surface_id}__MA__{ma_face.upper()}"),
    )


def _structured_sa_surface_record(
    *,
    name: str,
    record: Mapping[str, Any],
    volume_z_centers: Mapping[str, float],
    structured_volumes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str, str]:
    """Classify one structured SA face from exact solution-volume topology."""
    substrate_id, vacuum_id = _structured_dielectric_vacuum_adjacency(
        surface_type="SA",
        name=name,
        record=record,
        structured_volumes=structured_volumes,
    )
    substrate_center = volume_z_centers.get(substrate_id)
    vacuum_center = volume_z_centers.get(vacuum_id)
    if (
        substrate_center is None
        or vacuum_center is None
        or not all(math.isfinite(value) for value in (substrate_center, vacuum_center))
    ):
        raise ValueError(
            f"SGB SA surface {name!r} needs finite adjacent-volume topology."
        )
    if substrate_center == vacuum_center:
        raise ValueError(
            f"SGB SA surface {name!r} has ambiguous adjacent-volume topology."
        )
    face_kind = "top" if vacuum_center > substrate_center else "bottom"
    surface_id = str(record["surface_id"])
    return ("SA", surface_id, face_kind, surface_id)


def _structured_dielectric_vacuum_adjacency(
    *,
    surface_type: str,
    name: str,
    record: Mapping[str, Any],
    structured_volumes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Resolve exact dielectric and vacuum volume IDs for one structured face."""
    adjacent_ids = tuple(record["adjacent_solution_volume_ids"])
    if len(adjacent_ids) != 2 or len(set(adjacent_ids)) != 2:
        raise ValueError(
            f"SGB {surface_type} surface {name!r} needs exactly two distinct "
            "solution volumes."
        )
    material_kinds = {
        volume_id: _structured_solution_volume_material_kind(
            surface_type=surface_type,
            surface_name=name,
            volume_id=volume_id,
            structured_volumes=structured_volumes,
        )
        for volume_id in adjacent_ids
    }
    dielectric_ids = [
        volume_id for volume_id, kind in material_kinds.items() if kind == "dielectric"
    ]
    vacuum_ids = [
        volume_id for volume_id, kind in material_kinds.items() if kind == "vacuum"
    ]
    if len(dielectric_ids) != 1 or len(vacuum_ids) != 1:
        raise ValueError(
            f"SGB {surface_type} surface {name!r} needs one dielectric and one "
            "vacuum volume."
        )
    return dielectric_ids[0], vacuum_ids[0]


def _structured_solution_volume_material_kind(
    *,
    surface_type: str,
    surface_name: str,
    volume_id: str,
    structured_volumes: Mapping[str, Mapping[str, Any]],
) -> Literal["dielectric", "vacuum"]:
    """Classify a structured solution volume from its typed material identity."""
    volume = structured_volumes.get(volume_id)
    if volume is None:
        raise ValueError(
            f"SGB {surface_type} surface {surface_name!r} lacks structured volume "
            f"{volume_id!r}."
        )
    material_ids = volume["physical_attribute"].get("material_ids")
    if not _is_nonempty_string_sequence(material_ids):
        raise ValueError(
            f"SGB {surface_type} volume {volume_id!r} needs "
            "physical_attribute.material_ids."
        )
    if len(material_ids) != 1:
        raise ValueError(
            f"SGB {surface_type} volume {volume_id!r} needs one exact material "
            "identity."
        )
    return "vacuum" if material_ids[0].casefold() in {"air", "vacuum"} else "dielectric"


def _surface_z_center(entity_tags: Sequence[int]) -> float | None:
    """Return the plane coordinate of one planar physical sheet."""
    bbox = _entities_bbox(2, entity_tags)
    if len(bbox) != 6 or bbox[5] - bbox[2] >= gmsh_utils.PLANAR_ORIENTATION_TOLERANCE:
        return None
    return 0.5 * (bbox[2] + bbox[5])


def _parse_surface_physical_name(
    name: str,
    *,
    record: Mapping[str, Any],
    source_record_ids: tuple[str, ...] = (),
    volume_z_centers: Mapping[str, float] | None = None,
    structured_required: bool = False,
) -> tuple[str, str, str | None, str] | None:
    """Parse SGB surface records by authoritative fields, with legacy fallback."""
    if _has_structured_surface_marker(
        record,
        structured_required=structured_required,
    ):
        return _structured_surface_record(name, record)

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


def _structured_surface_record(
    name: str,
    record: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """Validate and classify a current SGB surface record without name inference."""
    required_strings = (
        "surface_id",
        "interface_type",
        "face_kind",
        "representation",
    )
    missing = [
        field
        for field in required_strings
        if _optional_string(record.get(field)) is None
    ]
    missing.extend(
        field
        for field in (
            "contact_kind",
            "conductor_component_id",
            "net_id",
            "equipotential_id",
        )
        if field not in record
    )
    if missing:
        raise ValueError(
            f"SGB surface {name!r} misses structured fields: {', '.join(missing)}."
        )
    interface_type = str(record["interface_type"]).upper()
    if interface_type not in _STRUCTURED_INTERFACE_TYPES:
        raise ValueError(
            f"SGB surface {name!r} has unsupported interface_type {interface_type!r}."
        )
    face_kind = _STRUCTURED_FACE_KINDS.get(str(record["face_kind"]).upper())
    if face_kind is None:
        raise ValueError(
            f"SGB surface {name!r} has invalid face_kind {record['face_kind']!r}."
        )
    representation = _optional_string(record.get("representation"))
    route = _optional_string(record.get("route"))
    if representation is None or route not in {"A", "B"}:
        raise ValueError(f"SGB surface {name!r} lacks a Route A/B representation.")
    if not _is_nonempty_string_sequence(record.get("owner_semantic_ids")):
        raise ValueError(f"SGB surface {name!r} needs exact owner_semantic_ids.")
    if not _is_nonempty_string_sequence(record.get("adjacent_solution_volume_ids")):
        raise ValueError(
            f"SGB surface {name!r} needs exact adjacent_solution_volume_ids."
        )
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB surface {name!r} needs mapping {field}.")
    component_id = _optional_string(record.get("conductor_component_id"))
    net_id = _optional_string(record.get("net_id"))
    if interface_type == "SA":
        if component_id is not None or net_id is not None:
            raise ValueError(
                f"SGB SA surface {name!r} must not carry conductor/net identity."
            )
        source_id = str(record["surface_id"])
    else:
        if component_id is None:
            raise ValueError(f"SGB surface {name!r} needs conductor_component_id.")
        source_id = _structured_conductor_source_layer(name, record)
    return (interface_type, source_id, face_kind, str(record["surface_id"]))


def _structured_conductor_source_layer(name: str, record: Mapping[str, Any]) -> str:
    """Read the exact SGB-emitted stack source for one exposed conductor face."""
    provenance = record.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"SGB surface {name!r} needs mapping source_provenance.")
    direct = _optional_string(provenance.get("conductor_source_layer_name"))
    if direct is not None:
        return direct
    sources = provenance.get("sources")
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError(
            f"SGB surface {name!r} needs exact conductor source-layer provenance."
        )
    source_layers: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError(f"SGB surface {name!r} has invalid provenance source.")
        source_layer = _optional_string(source.get("conductor_source_layer_name"))
        if source_layer is None:
            raise ValueError(f"SGB surface {name!r} lacks conductor_source_layer_name.")
        source_layers.add(source_layer)
    if len(source_layers) != 1:
        raise ValueError(f"SGB surface {name!r} has ambiguous source-layer provenance.")
    return source_layers.pop()


def _validate_structured_domain_boundary_record(
    name: str,
    record: Mapping[str, Any],
) -> None:
    """Validate current SGB domain-boundary identity without name inference."""
    required = (
        "surface_id",
        "representation",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "adjacent_solution_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"SGB domain boundary {name!r} misses fields: {', '.join(missing)}."
        )
    if not _optional_string(record.get("surface_id")):
        raise ValueError(f"SGB domain boundary {name!r} has empty surface_id.")
    if not _optional_string(record.get("representation")):
        raise ValueError(f"SGB domain boundary {name!r} has empty representation.")
    if not _is_nonempty_string_sequence(record.get("owner_semantic_ids")):
        raise ValueError(f"SGB domain boundary {name!r} needs owner_semantic_ids.")
    if not _is_nonempty_string_sequence(record.get("adjacent_solution_volume_ids")):
        raise ValueError(
            f"SGB domain boundary {name!r} needs adjacent_solution_volume_ids."
        )
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB domain boundary {name!r} needs mapping {field}.")


def _validate_structured_volume_record(
    name: str,
    record: Mapping[str, Any],
) -> None:
    """Validate and preserve current SGB volume identity and provenance."""
    for field in ("representation", "source_provenance", "physical_attribute"):
        if field not in record:
            raise ValueError(f"SGB volume {name!r} misses {field}.")
    if not _optional_string(record.get("representation")):
        raise ValueError(f"SGB volume {name!r} has empty representation.")
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB volume {name!r} needs mapping {field}.")


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
    bboxes: list[tuple[float, float, float, float, float, float]] = []
    for tag in entity_tags:
        raw_bbox = gmsh.model.getBoundingBox(dim, tag)
        if (
            not isinstance(raw_bbox, Sequence)
            or isinstance(raw_bbox, (str, bytes))
            or len(raw_bbox) != 6
            or not all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in raw_bbox
            )
        ):
            raise ValueError(
                f"Gmsh bounding box for dimension {dim} entity {tag} is malformed."
            )
        bbox = tuple(float(value) for value in raw_bbox)
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError(
                f"Gmsh bounding box for dimension {dim} entity {tag} is non-finite."
            )
        bboxes.append(cast(tuple[float, float, float, float, float, float], bbox))
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
