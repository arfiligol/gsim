"""Geometry extraction and Gmsh entity creation for Palace meshes.

This module turns component polygons, dielectric boxes, PEC blocks, generated
port sheets, and selected authored port sheets into raw Gmsh geometry tags. It
also delegates layout-authored sheet selection to the sheet helpers when a port
requests that source.

PDK simulation-layer catalogs and port intent are supplied before this stage.
Physical group assignment, Palace JSON generation, and result resolution are
handled by later layers. In the mesh pipeline, this module creates geometry
after ``mesh.generator`` has gathered inputs and before ``mesh.groups`` assigns
physical meaning.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import gmsh
from shapely import Polygon as ShapelyPolygon
from shapely import buffer
from shapely.ops import unary_union

from gsim.palace.models.ports import PortGeometry, PortType

from . import gmsh_utils
from .sheets import (
    AuthoredSheetPolygon,
    create_authored_sheet_surface,
    select_authored_sheet_polygon,
)

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import ActivatedRegion
    from gsim.palace.models.pec import PECBlockConfig
    from gsim.palace.models.ports import PalacePort
    from gsim.palace.models.simulation_layers import SimulationLayerCatalog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetalGeometryResult:
    """Metal tags plus metadata needed by later mesh stages."""

    metal_tags: dict
    shaped_dielectric_names: set[str]
    pec_surface_bboxes: dict


@dataclass
class GeometryData:
    """Container for geometry data extracted from component."""

    polygons: list  # List of (layer_num, pts_x, pts_y, holes) tuples
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    layer_bboxes: dict  # layer_num -> (xmin, ymin, xmax, ymax)


def extract_geometry(
    component,
    stack: LayerStack,
    *,
    decimate_tolerance: float | None = None,
    exclude_gds_layers: set[tuple[int, int]] | None = None,
) -> GeometryData:
    """Extract polygon geometry from a gdsfactory component.

    Args:
        component: gdsfactory Component
        stack: LayerStack for layer mapping
        decimate_tolerance: If set, simplify polygons with Douglas-Peucker
            using this relative tolerance (passed to ``decimate()``).
            Typical values: 0.001 (conservative) to 0.01 (aggressive).
        exclude_gds_layers: Full GDS layer/datatype tuples to skip because
            another mesh input owns them, such as simulation-only solver sheets.

    Returns:
        GeometryData with polygons and bounding boxes
    """
    polygons = []
    excluded_layers = exclude_gds_layers or set()
    global_bbox = [math.inf, math.inf, -math.inf, -math.inf]
    layer_bboxes = {}

    # Get polygons from component
    polygons_by_index = component.get_polygons()

    if decimate_tolerance is not None:
        from gsim.common.polygon_utils import decimate

        total_before = 0
        total_after = 0
        decimated_by_index = {}
        for layer_index, polys in polygons_by_index.items():
            total_before += sum(p.num_points_hull() for p in polys)
            decimated_by_index[layer_index] = decimate(
                polys, relative_tolerance=decimate_tolerance, verbose=True
            )
            total_after += sum(
                p.num_points_hull() for p in decimated_by_index[layer_index]
            )
        if total_before > 0:
            pct = (total_before - total_after) / total_before * 100
            logger.info(
                "Decimation: %d -> %d pts (%.1f%% removed)",
                total_before,
                total_after,
                pct,
            )
        polygons_by_index = decimated_by_index

    # Build layer_index -> GDS tuple mapping
    layout = component.kcl.layout
    index_to_gds = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    # Build GDS tuple -> layer number mapping
    gds_to_layernum = {}
    for layer_data in stack.layers.values():
        gds_tuple = tuple(layer_data.gds_layer)
        gds_to_layernum[gds_tuple] = gds_tuple[0]

    # Convert polygons
    for layer_index, polys in polygons_by_index.items():
        gds_tuple = index_to_gds.get(layer_index)
        if gds_tuple is None:
            continue
        if gds_tuple in excluded_layers:
            continue

        layernum = gds_to_layernum.get(gds_tuple)
        if layernum is None:
            continue

        for poly in polys:
            # Convert klayout polygon to lists (nm -> um)
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue

            pts_x = [pt.x / 1000.0 for pt in points]
            pts_y = [pt.y / 1000.0 for pt in points]

            # Extract holes from polygon
            holes = []
            for hole_idx in range(poly.holes()):
                hole_pts = list(poly.each_point_hole(hole_idx))
                if len(hole_pts) >= 3:
                    hx = [pt.x / 1000.0 for pt in hole_pts]
                    hy = [pt.y / 1000.0 for pt in hole_pts]
                    holes.append((hx, hy))

            polygons.append((layernum, pts_x, pts_y, holes))

            # Update bounding boxes
            xmin, xmax = min(pts_x), max(pts_x)
            ymin, ymax = min(pts_y), max(pts_y)

            global_bbox[0] = min(global_bbox[0], xmin)
            global_bbox[1] = min(global_bbox[1], ymin)
            global_bbox[2] = max(global_bbox[2], xmax)
            global_bbox[3] = max(global_bbox[3], ymax)

            if layernum not in layer_bboxes:
                layer_bboxes[layernum] = [xmin, ymin, xmax, ymax]
            else:
                bbox = layer_bboxes[layernum]
                bbox[0] = min(bbox[0], xmin)
                bbox[1] = min(bbox[1], ymin)
                bbox[2] = max(bbox[2], xmax)
                bbox[3] = max(bbox[3], ymax)

    return GeometryData(
        polygons=polygons,
        bbox=(global_bbox[0], global_bbox[1], global_bbox[2], global_bbox[3]),
        layer_bboxes=layer_bboxes,
    )


def _snap_via_z_range(
    stack: LayerStack,
    via_name: str,
    zmin: float,
    zmax: float,
    tol: float = 0.05,
) -> tuple[float, float]:
    """Snap a via's z-range so it doesn't sliver into adjacent conductor layers.

    PDK stacks routinely model vias with z-ranges that overlap the metal layers
    they connect to (e.g. IHP `vmim` z=[5.58, 6.24] vs `topmetal1` z=[6.23, 8.23]
    — a 10 nm overlap to model fab interdiffusion). After mesh fragmentation,
    that overlap becomes a sliver volume too thin for gmsh to mesh, producing
    "No elements in volume N" warnings.

    If the via's zmax exceeds an adjacent conductor's zmin by less than *tol*,
    snap zmax down to that conductor's zmin (and symmetrically for zmin).
    """
    new_zmin, new_zmax = zmin, zmax
    for other_name, other in stack.layers.items():
        if other_name == via_name or other.layer_type != "conductor":
            continue
        # Via top extends slightly into the conductor above
        if zmin < other.zmin < zmax and (zmax - other.zmin) < tol:
            new_zmax = min(new_zmax, other.zmin)
        # Via bottom extends slightly into the conductor below
        if zmin < other.zmax < zmax and (other.zmax - zmin) < tol:
            new_zmin = max(new_zmin, other.zmax)
    if (new_zmin, new_zmax) != (zmin, zmax):
        logger.info(
            "Snapped via '%s' z-range [%.3f, %.3f] -> [%.3f, %.3f] "
            "to avoid sliver against adjacent conductor",
            via_name,
            zmin,
            zmax,
            new_zmin,
            new_zmax,
        )
    return new_zmin, new_zmax


def _merge_via_polygons(
    polys: list[tuple[list[float], list[float], list]],
    merge_distance: float,
) -> list[tuple[list[float], list[float], list]]:
    """Merge nearby via polygons using oversize/union/undersize.

    Converts polygon coordinate lists to shapely, buffers outward by
    half the merge distance so nearby vias overlap, unions them, then
    buffers back inward to restore the original outline.

    Args:
        polys: List of (pts_x, pts_y, holes) tuples
        merge_distance: Max gap between vias to merge (um)

    Returns:
        Merged polygon list in the same format
    """
    if len(polys) <= 1 or merge_distance <= 0:
        return polys

    # Small epsilon ensures vias exactly at merge_distance still overlap after buffering
    _MERGE_EPSILON = 0.01  # um
    offset = merge_distance / 2 + _MERGE_EPSILON

    shapely_polys = []
    for pts_x, pts_y, _holes in polys:
        coords = list(zip(pts_x, pts_y, strict=False))
        if len(coords) >= 3:
            shapely_polys.append(ShapelyPolygon(coords))

    if not shapely_polys:
        return polys

    # Oversize -> union -> undersize
    buffered = [buffer(p, offset, join_style="mitre") for p in shapely_polys]
    merged = buffer(unary_union(buffered), -offset, join_style="mitre")

    # Convert back to coordinate lists
    result = []
    # unary_union may return Polygon or MultiPolygon
    geoms_attr = getattr(merged, "geoms", None)
    geoms = list(geoms_attr) if geoms_attr is not None else [merged]
    for geom in geoms:
        if geom.is_empty:
            continue
        xs, ys = zip(*geom.exterior.coords[:-1], strict=True)  # drop closing duplicate
        holes = []
        for interior in geom.interiors:
            hx, hy = zip(*interior.coords[:-1], strict=True)
            holes.append((list(hx), list(hy)))
        result.append((list(xs), list(ys), holes))

    n_before = len(polys)
    n_after = len(result)
    if n_before != n_after:
        logger.info(
            "Via merging: %d polygons -> %d (distance=%.1f um)",
            n_before,
            n_after,
            merge_distance,
        )

    return result


def _is_covered_by_dielectric_box(layer, stack: LayerStack) -> bool:
    """Check if a layer is already represented as a bulk box in stack.dielectrics.

    When a dielectric layer's material and z-range are covered by a
    ``stack.dielectrics`` entry, the polygon geometry on that layer is
    just a placeholder for a full-domain bulk region (e.g. vacuum or
    cladding) and should NOT be polygon-extruded as a shaped volume.
    """
    from gsim.common.stack.materials import MATERIAL_ALIASES

    layer_mat = MATERIAL_ALIASES.get(layer.material.lower(), layer.material.lower())
    for d in stack.dielectrics:
        d_mat_name = d.get("material", "")
        d_mat = MATERIAL_ALIASES.get(d_mat_name.lower(), d_mat_name.lower())
        if (
            d_mat == layer_mat
            and d["zmin"] <= layer.zmin + 1e-6
            and d["zmax"] >= layer.zmax - 1e-6
        ):
            return True
    return False


def _detect_shaped_dielectric_layers(
    geometry: GeometryData, stack: LayerStack
) -> set[str]:
    """Detect dielectric layers that should be polygon-extruded (shaped).

    A dielectric layer is shaped when it carries polygon geometry in the
    component AND is NOT already represented as a bulk box in
    ``stack.dielectrics``.  Bulk regions (vacuum, cladding) that have
    both a Layer entry and a dielectric box entry are treated as boxes.
    Waveguide cores only exist as Layer entries with polygon geometry —
    they have no corresponding dielectric box, so they must be
    polygon-extruded as shaped volumes.

    Returns:
        Set of layer names that should be treated as shaped dielectrics.
    """
    gds_layers_with_polys = {layernum for layernum, *_ in geometry.polygons}

    shaped: set[str] = set()
    for name, layer in stack.layers.items():
        if layer.layer_type != "dielectric":
            continue
        if layer.gds_layer[0] not in gds_layers_with_polys:
            continue
        if _is_covered_by_dielectric_box(layer, stack):
            continue
        shaped.add(name)

    return shaped


def get_layer_info(stack: LayerStack, gds_layer: int) -> dict | None:
    """Get layer info from stack by GDS layer number.

    Args:
        stack: LayerStack with layer definitions
        gds_layer: GDS layer number

    Returns:
        Dict with layer info or None if not found
    """
    for name, layer in stack.layers.items():
        if layer.gds_layer[0] == gds_layer:
            return {
                "name": name,
                "zmin": layer.zmin,
                "zmax": layer.zmax,
                "thickness": layer.zmax - layer.zmin,
                "material": layer.material,
                "type": layer.layer_type,
            }
    return None


def add_metals(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    planar_conductors: bool = False,
    merge_via_distance: float = 2.0,
) -> MetalGeometryResult:
    """Add metal, via, and shaped-dielectric geometries to gmsh.

    Creates extruded volumes for vias and shells (surfaces) for conductors.
    Shaped dielectrics (auto-detected: dielectric layers with polygon
    geometry that are NOT already represented as bulk boxes in
    ``stack.dielectrics``) are extruded as 3D solid volumes, similar to
    vias, but are not hollowed out — they retain their full volume and
    carry dielectric permittivity in the Palace config.

    If planar_conductors is True, conductors are treated as 2D surfaces (PEC).

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with layer definitions
        planar_conductors: If True, treat conductors as 2D PEC surfaces
        merge_via_distance: Max gap between vias to merge (um). Nearby
            via polygons within this distance are combined into a single
            polygon before meshing, drastically reducing mesh complexity.

    Returns:
        Dict with layer_name -> {"volumes": [...], "surfaces_xy": [...],
        "surfaces_z": [...]} where volumes contains raw int tags for vias
        and shaped dielectrics, and (volumetag, surface_tags) tuples for
        conductors.
    """
    # layer_name -> {"volumes": [], "surfaces_xy": [], "surfaces_z": []}
    metal_tags: dict[str, dict[str, list]] = {}
    # Detect shaped-dielectric layers once (replaces thickness heuristic)
    shaped_dielectric_names = _detect_shaped_dielectric_layers(geometry, stack)

    # Group polygons by layer
    polygons_by_layer = {}
    for layernum, pts_x, pts_y, holes in geometry.polygons:
        if layernum not in polygons_by_layer:
            polygons_by_layer[layernum] = []
        polygons_by_layer[layernum].append((pts_x, pts_y, holes))

    # Track conductor volumes for deferred surface extraction.
    # Surface loop tags are only queried *after* removeAllDuplicates()
    # so that stale tags from merged sub-entities cannot cause crashes.
    _conductor_volumes: dict[str, list[int]] = {}

    # Process each layer
    for layernum, polys in polygons_by_layer.items():
        layer_info = get_layer_info(stack, layernum)
        if layer_info is None:
            continue

        layer_name = layer_info["name"]
        layer_type = layer_info["type"]
        zmin = layer_info["zmin"]
        thickness = layer_info["thickness"]
        zmax = zmin + thickness
        is_shaped_dielectric = layer_name in shaped_dielectric_names

        if layer_type not in ("conductor", "via") and not is_shaped_dielectric:
            continue

        # Snap via z-range so it does not sliver into an adjacent conductor
        if layer_type == "via":
            zmin, zmax = _snap_via_z_range(stack, layer_name, zmin, zmax)
            thickness = zmax - zmin

        if layer_name not in metal_tags:
            metal_tags[layer_name] = {
                "volumes": [],
                "surfaces_xy": [],
                "surfaces_z": [],
            }

        if is_shaped_dielectric:
            shaped_dielectric_names.add(layer_name)

        min_volume_thickness = 0.05  # um — thinner volumes can't mesh as 3D
        is_planar = (
            planar_conductors or thickness == 0 or thickness < min_volume_thickness
        )

        # Merge nearby via polygons before creating gmsh surfaces
        if layer_type == "via":
            polys = _merge_via_polygons(polys, merge_via_distance)

        # Create surfaces for all polygons on this layer
        surfaces = []
        for pts_x, pts_y, holes in polys:
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel, pts_x, pts_y, zmin, holes=holes
            )
            if surfacetag is not None:
                surfaces.append(surfacetag)

        if not surfaces:
            continue

        if is_shaped_dielectric:
            # Shaped dielectric: extrude as solid 3D volume (like a via)
            # but keep the full volume (no shell extraction). The volume
            # carries dielectric permittivity in the Palace config.
            if thickness == 0 or thickness < min_volume_thickness:
                logger.warning(
                    "Shaped dielectric layer '%s' too thin for 3D meshing "
                    "(%.3f um < %.3f um), skipping shaped extrusion",
                    layer_name,
                    thickness,
                    min_volume_thickness,
                )
                continue
            # Fuse overlapping same-layer surfaces before extrusion
            if len(surfaces) > 1:
                dimtags = [(2, s) for s in surfaces]
                fused, _ = kernel.fuse(
                    [dimtags[0]],
                    dimtags[1:],
                    removeObject=True,
                    removeTool=True,
                )
                kernel.synchronize()
                surfaces = [t for d, t in fused if d == 2]

            logger.info(
                "Shaped dielectric layer '%s': 3D volume "
                "(material=%s, thickness=%.3f um)",
                layer_name,
                layer_info["material"],
                thickness,
            )
            for surfacetag in surfaces:
                result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                volumetag = result[1][1]
                metal_tags[layer_name]["volumes"].append(volumetag)
        elif layer_type == "conductor" and is_planar:
            # Zero/thin-thickness or explicitly planar -> 2D PEC surface
            metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
            # Also create explicit wire loops for mesh refinement.  Embedded
            # planar surfaces lose their boundary curves after boolean
            # fragmentation, so the conductor edges cannot drive refinement.
            # Adding independent line loops at the conductor z-height gives
            # gmsh explicit curves to refine around the metal perimeter.
            for pts_x, pts_y, holes in polys:
                loop_tag = gmsh_utils._create_wire_loop(  # noqa: SLF001
                    kernel, list(pts_x), list(pts_y), zmin
                )
                if loop_tag is not None:
                    metal_tags[layer_name].setdefault("refinement_lines", []).append(
                        loop_tag
                    )
                for hx, hy in holes:
                    hole_loop = gmsh_utils._create_wire_loop(  # noqa: SLF001
                        kernel, list(hx), list(hy), zmin
                    )
                    if hole_loop is not None:
                        metal_tags[layer_name].setdefault(
                            "refinement_lines", []
                        ).append(hole_loop)
        elif layer_type == "via":
            # Decide between 3D volume (with conductivity) and 2D PEC fallback
            material_name = layer_info["material"]
            mat_props = stack.materials.get(material_name, {})
            conductivity = mat_props.get("conductivity", 0.0)
            via_too_thin = thickness == 0 or thickness < min_volume_thickness

            if via_too_thin:
                logger.warning(
                    "Via layer '%s' too thin for 3D meshing "
                    "(%.3f um < %.3f um), falling back to 2D PEC surface",
                    layer_name,
                    thickness,
                    min_volume_thickness,
                )
                metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
            elif conductivity <= 0:
                logger.warning(
                    "Via layer '%s' has no conductivity for material '%s', "
                    "falling back to 2D PEC surface",
                    layer_name,
                    material_name,
                )
                metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
            else:
                # Extrude via as 3D volume with finite conductivity
                logger.info(
                    "Via layer '%s': 3D volume (material=%s, "
                    "\u03c3=%.2e S/m, thickness=%.3f um)",
                    layer_name,
                    material_name,
                    conductivity,
                    thickness,
                )
                for surfacetag in surfaces:
                    result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                    volumetag = result[1][1]
                    metal_tags[layer_name]["volumes"].append(volumetag)
        elif thickness > 0:
            # Fuse overlapping same-layer surfaces before extrusion so that
            # overlapping polygons (e.g. ground planes and spines in a GSG
            # electrode) become a single merged surface per layer.
            if len(surfaces) > 1:
                dimtags = [(2, s) for s in surfaces]
                fused, _ = kernel.fuse(
                    [dimtags[0]],
                    dimtags[1:],
                    removeObject=True,
                    removeTool=True,
                )
                kernel.synchronize()
                surfaces = [t for d, t in fused if d == 2]

            for surfacetag in surfaces:
                result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                volumetag = result[1][1]

                if layer_type == "via":
                    # Keep vias as volumes
                    metal_tags[layer_name]["volumes"].append(volumetag)
                else:
                    # Defer shell extraction until after removeAllDuplicates
                    _conductor_volumes.setdefault(layer_name, []).append(volumetag)

    # Record bounding boxes of BOTH via and conductor volumes BEFORE
    # removeAllDuplicates. That call renumbers ALL entity tags globally —
    # not just the ones it merges — so original tags are never trustworthy
    # afterwards. We re-identify volumes by bbox after the call.
    _via_bboxes: dict[str, list[tuple[float, ...]]] = {}
    _conductor_bboxes: dict[str, list[tuple[float, ...]]] = {}
    # Also record surface bboxes for planar conductors — these survive as
    # dielectric boundary faces after boolean and are used to find the
    # conductor perimeter curves for mesh refinement.
    _pec_surface_bboxes: dict[str, list[tuple[float, ...]]] = {}
    kernel.synchronize()
    for layer_name, tag_info in metal_tags.items():
        for vtag in tag_info["volumes"]:
            if isinstance(vtag, int):
                bbox = kernel.getBoundingBox(3, vtag)
                _via_bboxes.setdefault(layer_name, []).append(bbox)

        for stag in tag_info.get("surfaces_xy", []):
            try:
                bbox = kernel.getBoundingBox(2, stag)
                _pec_surface_bboxes.setdefault(layer_name, []).append(bbox)
            except Exception:
                logger.debug(
                    "Could not get bbox for planar conductor surface %d, skipping", stag
                )

    for layer_name, vol_tags in _conductor_volumes.items():
        for vtag in vol_tags:
            try:
                bbox = kernel.getBoundingBox(3, vtag)
                _conductor_bboxes.setdefault(layer_name, []).append(bbox)
            except Exception:
                logger.debug(
                    "Could not get bbox for conductor volume %d, skipping", vtag
                )

    kernel.removeAllDuplicates()
    kernel.synchronize()

    # After removeAllDuplicates(), some independent curve-loop entities
    # (created for planar-conductor refinement) may be merged away.
    # Refresh the refinement line tags so downstream consumers see only
    # valid curves.
    for tag_info in metal_tags.values():
        old_line_tags = tag_info.get("refinement_lines", [])
        if not old_line_tags:
            continue
        valid_lines = []
        for ltag in old_line_tags:
            try:
                kernel.getBoundingBox(1, ltag)
                valid_lines.append(ltag)
            except Exception:
                pass  # Curve was merged away
        # Try to find the merged successor by looking at all remaining curves
        # that have the same z-coordinate and bounding box.
        if len(valid_lines) < len(old_line_tags):
            # Some curves were merged — find replacements by matching bboxes.
            # getEntities(1) returns curves that survived dedup.
            all_curves = list(kernel.getEntities(1))
            all_curve_bboxes: dict[int, tuple] = {}
            for _, ctag in all_curves:
                with contextlib.suppress(Exception):
                    all_curve_bboxes[ctag] = kernel.getBoundingBox(1, ctag)
            for ltag in old_line_tags:
                if ltag in valid_lines:
                    continue
                try:
                    old_bbox = kernel.getBoundingBox(1, ltag)
                except Exception:
                    continue
                for ctag, bbox in all_curve_bboxes.items():
                    if ctag in valid_lines:
                        continue
                    if all(
                        abs(a - b) < 0.01 for a, b in zip(bbox, old_bbox, strict=True)
                    ):
                        valid_lines.append(ctag)
                        break
        tag_info["refinement_lines"] = sorted(set(valid_lines))

    # Build a single bbox lookup for all post-dedup volumes (avoids O(n²) calls).
    all_vols = kernel.getEntities(3)
    all_vol_bboxes: dict[int, tuple] = {}
    for _, vtag in all_vols:
        try:
            all_vol_bboxes[vtag] = kernel.getBoundingBox(3, vtag)
        except Exception:
            logger.debug("Could not get bbox for volume %d after dedup", vtag)

    # Re-identify via volumes by matching bounding boxes.
    for layer_name, bboxes in _via_bboxes.items():
        metal_tags[layer_name]["volumes"] = []
        for target_bbox in bboxes:
            for vtag, bbox in all_vol_bboxes.items():
                if all(
                    abs(a - b) < 0.01 for a, b in zip(bbox, target_bbox, strict=True)
                ):
                    metal_tags[layer_name]["volumes"].append(vtag)
                    break

    # Re-identify conductor volumes by bbox and update _conductor_volumes.
    # Without this, removeAllDuplicates()'s global renumbering makes every
    # original tag appear missing, so all conductors are silently dropped.
    for layer_name, bboxes in _conductor_bboxes.items():
        new_vol_tags = []
        for target_bbox in bboxes:
            for vtag, bbox in all_vol_bboxes.items():
                if all(
                    abs(a - b) < 0.01 for a, b in zip(bbox, target_bbox, strict=True)
                ):
                    new_vol_tags.append(vtag)
                    break
            else:
                logger.warning(
                    "Conductor volume on %s lost during dedup (no bbox match)",
                    layer_name,
                )
        _conductor_volumes[layer_name] = new_vol_tags

    # Extract shell surfaces from conductor volumes (now with correct post-dedup tags).
    current_vols = {t for _, t in all_vols}
    for layer_name, vol_tags in _conductor_volumes.items():
        for volumetag in vol_tags:
            if volumetag not in current_vols:
                logger.warning(
                    "Conductor volume %d on %s missing after bbox re-identification",
                    volumetag,
                    layer_name,
                )
                continue
            _, surfaceloops = kernel.getSurfaceLoops(volumetag)
            if surfaceloops:
                metal_tags[layer_name]["volumes"].append((volumetag, surfaceloops[0]))
            kernel.remove([(3, volumetag)])

    if _conductor_volumes:
        kernel.synchronize()

    return MetalGeometryResult(
        metal_tags=metal_tags,
        shaped_dielectric_names=shaped_dielectric_names,
        pec_surface_bboxes=_pec_surface_bboxes,
    )


def add_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    margin_x: float,
    margin_y: float | None = None,
    air_margin: float = 0.0,
    airbox_margin_x: float | None = None,
    airbox_margin_y: float | None = None,
    airbox_z_above: float | None = None,
    airbox_z_below: float | None = None,
    activated_regions: Sequence[ActivatedRegion] | None = None,
) -> dict:
    """Add dielectric volumes to gmsh.

    All boxes are created at their exact geometric bounds.  Coincident
    faces and overlapping volumes are resolved later by
    ``run_boolean_pipeline``, which performs priority-based cuts and
    conformal fragmentation — no artificial offsets needed.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with dielectric definitions
        margin_x: X margin around design (um). Also used as Y margin
            when *margin_y* is not provided (backward compat).
        margin_y: Y margin around design (um). Defaults to margin_x.
        air_margin: Legacy isotropic extra margin for the surrounding
            airbox (um). Used as a fallback for all airbox directions.
        airbox_margin_x: Extra x-margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        airbox_margin_y: Extra y-margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        airbox_z_above: Extra +z margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        airbox_z_below: Extra -z margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        activated_regions: Stack layers explicitly selected as Palace mesh
            regions. These use the stack layer name as the physical group name.

    Returns:
        Dict with material_name -> list of volume_tags
    """
    if margin_y is None:
        margin_y = margin_x

    has_activated_regions = bool(activated_regions)
    if has_activated_regions:
        _reject_activated_region_airbox_controls(
            air_margin=air_margin,
            airbox_margin_x=airbox_margin_x,
            airbox_margin_y=airbox_margin_y,
            airbox_z_above=airbox_z_above,
            airbox_z_below=airbox_z_below,
        )
    else:
        if airbox_margin_x is None:
            airbox_margin_x = air_margin
        if airbox_margin_y is None:
            airbox_margin_y = air_margin
        if airbox_z_above is None:
            airbox_z_above = air_margin
        if airbox_z_below is None:
            airbox_z_below = air_margin

    dielectric_tags: dict[str, list[int]] = {}

    xmin0, ymin0, xmax0, ymax0 = geometry.bbox
    xmin_air = xmin0 - margin_x
    ymin_air = ymin0 - margin_y
    xmax_air = xmax0 + margin_x
    ymax_air = ymax0 + margin_y

    def _contains_air_token(name: str | None) -> bool:
        """Return True when *name* clearly denotes air/vacuum."""
        if not name:
            return False

        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"air", "vacuum"}:
            return True

        tokens = [tok for tok in normalized.split("_") if tok]
        return "air" in tokens or "vacuum" in tokens

    def _is_air_or_vacuum(
        material_name: str,
        dielectric_name: str | None = None,
    ) -> bool:
        """Return True when *material_name* represents air/vacuum.

        Uses stack material metadata (permittivity ~1) first, then
        falls back to name matching for robustness with custom stacks.
        """
        mat = stack.materials.get(material_name)
        if isinstance(mat, dict):
            mat_type = str(mat.get("type", "")).strip().lower()
            eps = mat.get("permittivity")
        else:
            mat_type = str(getattr(mat, "type", "")).strip().lower()
            eps = getattr(mat, "permittivity", None)

        if mat_type == "dielectric":
            try:
                if eps is not None and abs(float(eps) - 1.0) <= 1e-9:
                    return True
            except (TypeError, ValueError):
                pass

        if _contains_air_token(material_name):
            return True

        return _contains_air_token(dielectric_name)

    z_min_all = math.inf
    z_max_all = -math.inf

    use_airbox = not has_activated_regions and any(
        m is not None and m > 0.0
        for m in (
            airbox_margin_x,
            airbox_margin_y,
            airbox_z_above,
            airbox_z_below,
        )
    )

    activated_dielectrics = _activated_region_dielectrics(
        geometry, stack, activated_regions
    )
    activated_names = {str(dielectric["name"]) for dielectric in activated_dielectrics}
    dielectric_specs = [
        dielectric
        for dielectric in stack.dielectrics
        if str(dielectric.get("name")) not in activated_names
    ]
    dielectric_specs.extend(activated_dielectrics)

    for dielectric in dielectric_specs:
        dielectric_name = dielectric.get("name")
        material = dielectric["material"]
        is_activated_region = bool(dielectric.get("activated_region"))

        is_air_like = _is_air_or_vacuum(material, dielectric_name=dielectric_name)

        # When building an explicit airbox, skip explicit air/vacuum layers.
        if is_air_like and use_airbox:
            continue

        d_zmin = dielectric["zmin"]
        d_zmax = dielectric["zmax"]

        z_min_all = min(z_min_all, d_zmin)
        z_max_all = max(z_max_all, d_zmax)

        physical_name = str(dielectric_name) if is_activated_region else material
        dielectric_tags.setdefault(physical_name, [])

        if is_activated_region:
            xmin = dielectric["xmin"]
            ymin = dielectric["ymin"]
            xmax = dielectric["xmax"]
            ymax = dielectric["ymax"]
        else:
            xmin = xmin_air if is_air_like else xmin0
            ymin = ymin_air if is_air_like else ymin0
            xmax = xmax_air if is_air_like else xmax0
            ymax = ymax_air if is_air_like else ymax0

        # When shaped dielectrics exist, ALL non-air dielectric boxes must
        # extend to the air margins.  A shaped dielectric (e.g. waveguide
        # core) carves out of the surrounding oxide/substrate boxes, so those
        # boxes need to be large enough to fully surround the shaped volume.
        #
        # Also extend substrates (dielectrics starting at z ~ 0) to margins.
        # A bulk substrate like sapphire or silicon should fill the same
        # transverse extent as the surrounding air so the mesh domain is
        # consistent and the substrate edge does not artificially truncate
        # fields.
        layer = stack.layers.get(str(dielectric_name))
        is_bulk_substrate = d_zmin <= 1e-6 or (
            layer is not None and layer.layer_type == "substrate"
        )
        if (
            (is_bulk_substrate or _detect_shaped_dielectric_layers(geometry, stack))
            and not is_air_like
            and not is_activated_region
        ):
            xmin = xmin_air
            ymin = ymin_air
            xmax = xmax_air
            ymax = ymax_air

        box_tag = gmsh_utils.create_box(
            kernel,
            xmin,
            ymin,
            d_zmin,
            xmax,
            ymax,
            d_zmax,
        )
        dielectric_tags[physical_name].append(box_tag)

    # Resolve stack z envelope even if dielectric list is sparse.
    if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
        for layer in stack.layers.values():
            z_min_all = min(z_min_all, layer.zmin)
            z_max_all = max(z_max_all, layer.zmax)

    # Explicit single airbox (boolean pipeline handles overlap/subtraction).
    if use_airbox:
        if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
            raise ValueError(
                "Cannot create airbox because stack z extents could not be resolved"
            )

        airbox_x = airbox_margin_x
        airbox_y = airbox_margin_y
        airbox_above = airbox_z_above
        airbox_below = airbox_z_below
        if (
            airbox_x is None
            or airbox_y is None
            or airbox_above is None
            or airbox_below is None
        ):
            raise ValueError("Explicit airbox margins must all be provided")

        airbox_tag = gmsh_utils.create_box(
            kernel,
            xmin_air - airbox_x,
            ymin_air - airbox_y,
            z_min_all - airbox_below,
            xmax_air + airbox_x,
            ymax_air + airbox_y,
            z_max_all + airbox_above,
        )
        dielectric_tags["airbox"] = [airbox_tag]

    kernel.synchronize()

    return dielectric_tags


def _activated_region_dielectrics(
    geometry: GeometryData,
    stack: LayerStack,
    activated_regions: Sequence[ActivatedRegion] | None,
) -> list[dict]:
    """Return dielectric specs for activated stack-layer regions."""
    regions = tuple(activated_regions or ())
    if not regions:
        return []

    layer_by_region: dict[str, Any] = {}
    substrate_layers: list[Any] = []
    for region in regions:
        layer = stack.layers.get(region.layer)
        if layer is None:
            raise ValueError(
                f"Activated region layer '{region.layer}' is not present in the stack."
            )
        if layer.layer_type in {"conductor", "via"}:
            raise ValueError(
                f"Activated region layer '{region.layer}' must be a dielectric "
                "or substrate stack layer."
            )
        if region.role == "substrate" and layer.layer_type != "substrate":
            raise ValueError(
                f"Activated substrate layer '{region.layer}' must be a substrate "
                "stack layer."
            )
        if region.role in {"inter_die_vacuum", "outer_vacuum"} and (
            layer.layer_type != "dielectric"
        ):
            raise ValueError(
                f"Activated vacuum layer '{region.layer}' must be a dielectric "
                "stack layer."
            )
        layer_by_region[region.layer] = layer
        if region.role == "substrate":
            substrate_layers.append(layer)

    substrate_zmin = min((layer.zmin for layer in substrate_layers), default=None)
    substrate_zmax = max((layer.zmax for layer in substrate_layers), default=None)
    substrate_xy_bounds = [
        _activated_region_xy_bounds(geometry, region)
        for region in regions
        if region.role == "substrate"
    ]
    substrate_xy = (
        (
            min(bounds[0] for bounds in substrate_xy_bounds),
            min(bounds[1] for bounds in substrate_xy_bounds),
            max(bounds[2] for bounds in substrate_xy_bounds),
            max(bounds[3] for bounds in substrate_xy_bounds),
        )
        if substrate_xy_bounds
        else None
    )

    specs: list[dict] = []
    for region in regions:
        layer = layer_by_region[region.layer]
        if region.role == "outer_vacuum":
            if substrate_zmin is None or substrate_zmax is None or substrate_xy is None:
                raise ValueError(
                    "Activated outer vacuum requires at least one active substrate "
                    "region."
                )
            (
                substrate_xmin,
                substrate_ymin,
                substrate_xmax,
                substrate_ymax,
            ) = substrate_xy
            xmin = substrate_xmin - region.margin_x
            ymin = substrate_ymin - region.margin_y
            xmax = substrate_xmax + region.margin_x
            ymax = substrate_ymax + region.margin_y
            zmin = substrate_zmin - region.z_below
            zmax = substrate_zmax + region.z_above
        else:
            xmin, ymin, xmax, ymax = _activated_region_xy_bounds(geometry, region)
            zmin = layer.zmin
            zmax = layer.zmax
        if zmax <= zmin:
            raise ValueError(
                f"Activated region layer '{region.layer}' has non-positive z extent."
            )
        specs.append(
            {
                "name": layer.name,
                "zmin": zmin,
                "zmax": zmax,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "material": region.material or layer.material,
                "activated_region": True,
                "activated_region_role": region.role,
                "die": region.die,
                "lower_die": region.lower_die,
                "upper_die": region.upper_die,
                "margin_x": region.margin_x,
                "margin_y": region.margin_y,
                "z_above": region.z_above,
                "z_below": region.z_below,
                "material_override": region.material,
            }
        )
    return specs


def _activated_region_xy_bounds(
    geometry: GeometryData,
    region: ActivatedRegion,
) -> tuple[float, float, float, float]:
    xmin0, ymin0, xmax0, ymax0 = geometry.bbox
    return (
        xmin0 - region.margin_x,
        ymin0 - region.margin_y,
        xmax0 + region.margin_x,
        ymax0 + region.margin_y,
    )


def _reject_activated_region_airbox_controls(
    *,
    air_margin: float,
    airbox_margin_x: float | None,
    airbox_margin_y: float | None,
    airbox_z_above: float | None,
    airbox_z_below: float | None,
) -> None:
    if air_margin > 0:
        raise ValueError(
            "Explicit activated regions cannot be mixed with airbox controls "
            "(air_margin > 0)."
        )
    provided = [
        name
        for name, value in (
            ("airbox_margin_x", airbox_margin_x),
            ("airbox_margin_y", airbox_margin_y),
            ("airbox_z_above", airbox_z_above),
            ("airbox_z_below", airbox_z_below),
        )
        if value is not None
    ]
    if provided:
        names = ", ".join(provided)
        raise ValueError(
            "Explicit activated regions cannot be mixed with airbox controls "
            f"({names})."
        )


def resolve_mesh_domain_bounds(
    geometry: GeometryData,
    stack: LayerStack,
    *,
    margin_x: float,
    margin_y: float | None = None,
    air_margin: float = 0.0,
    airbox_margin_x: float | None = None,
    airbox_margin_y: float | None = None,
    airbox_z_above: float | None = None,
    airbox_z_below: float | None = None,
    activated_regions: Sequence[ActivatedRegion] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Resolve outer mesh-domain bounds (including explicit airbox when used)."""
    if margin_y is None:
        margin_y = margin_x

    if activated_regions:
        _reject_activated_region_airbox_controls(
            air_margin=air_margin,
            airbox_margin_x=airbox_margin_x,
            airbox_margin_y=airbox_margin_y,
            airbox_z_above=airbox_z_above,
            airbox_z_below=airbox_z_below,
        )
        specs = _activated_region_dielectrics(geometry, stack, activated_regions)
        return (
            min(spec["xmin"] for spec in specs),
            min(spec["ymin"] for spec in specs),
            min(spec["zmin"] for spec in specs),
            max(spec["xmax"] for spec in specs),
            max(spec["ymax"] for spec in specs),
            max(spec["zmax"] for spec in specs),
        )

    if airbox_margin_x is None:
        airbox_margin_x = air_margin
    if airbox_margin_y is None:
        airbox_margin_y = air_margin
    if airbox_z_above is None:
        airbox_z_above = air_margin
    if airbox_z_below is None:
        airbox_z_below = air_margin

    xmin0, ymin0, xmax0, ymax0 = geometry.bbox
    xmin_air = xmin0 - margin_x
    ymin_air = ymin0 - margin_y
    xmax_air = xmax0 + margin_x
    ymax_air = ymax0 + margin_y

    # Robust stack z-envelope: include dielectric and layer extents.
    z_min_all = math.inf
    z_max_all = -math.inf
    for dielectric in stack.dielectrics:
        z_min_all = min(z_min_all, dielectric["zmin"])
        z_max_all = max(z_max_all, dielectric["zmax"])

    if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
        z_try_min, z_try_max = stack.get_z_range()
        z_min_all = min(z_min_all, z_try_min)
        z_max_all = max(z_max_all, z_try_max)

    if stack.layers:
        z_min_layers = min(layer.zmin for layer in stack.layers.values())
        z_max_layers = max(layer.zmax for layer in stack.layers.values())
        z_min_all = min(z_min_all, z_min_layers)
        z_max_all = max(z_max_all, z_max_layers)

    if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
        raise ValueError("Cannot resolve stack z extents for domain bounds")

    use_airbox = any(
        m > 0.0
        for m in (
            airbox_margin_x,
            airbox_margin_y,
            airbox_z_above,
            airbox_z_below,
        )
    )

    if use_airbox:
        return (
            xmin_air - airbox_margin_x,
            ymin_air - airbox_margin_y,
            z_min_all - airbox_z_below,
            xmax_air + airbox_margin_x,
            ymax_air + airbox_margin_y,
            z_max_all + airbox_z_above,
        )

    return (xmin_air, ymin_air, z_min_all, xmax_air, ymax_air, z_max_all)


def add_patterned_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    min_volume_thickness: float = 0.05,
    curve_fit_mode: Literal["line", "spline", "bspline"] = "line",
    curve_fit_layers: list[str] | None = None,
    curve_fit_tolerance_um: float = 0.0,
    curve_fit_min_points: int = 8,
    curve_fit_corner_angle_deg: float = 45.0,
) -> dict[str, list[int]]:
    """Add patterned dielectric volumes from stack dielectric layers.

    This extrudes component polygons for stack layers classified as
    ``dielectric`` so patterned optical/routing cores become explicit
    3D dielectric regions in the boolean pipeline.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with layer definitions
        min_volume_thickness: Skip very thin dielectric layers that cannot
            be robustly meshed as 3D volumes.
        curve_fit_mode: Boundary curve mode for selected layers.
        curve_fit_layers: Layer names where spline/bspline fitting is allowed.
        curve_fit_tolerance_um: Point merge tolerance before curve fitting.
        curve_fit_min_points: Minimum contour points to attempt curve fitting.
        curve_fit_corner_angle_deg: Turn-angle threshold for corner detection
            during spline/bspline segmentation.

    Returns:
        Dict mapping dielectric layer name -> list of volume tags.
    """
    patterned_tags: dict[str, list[int]] = {}
    curve_layers = set(curve_fit_layers or [])

    # Shaped dielectric layers are already extruded by ``add_metals``.
    # Re-extruding them here would create overlapping volumes with duplicate
    # ``Entity`` names, which collide in the boolean/physical-group pipeline.
    shaped_dielectric_names = _detect_shaped_dielectric_layers(geometry, stack)

    polygons_by_layer: dict[int, list[tuple[list[float], list[float], list]]] = {}
    for layernum, pts_x, pts_y, holes in geometry.polygons:
        polygons_by_layer.setdefault(layernum, []).append((pts_x, pts_y, holes))

    for layernum, polys in polygons_by_layer.items():
        layer_info = get_layer_info(stack, layernum)
        if layer_info is None or layer_info["type"] != "dielectric":
            continue

        layer_name = layer_info["name"]
        if layer_name in shaped_dielectric_names:
            continue

        # Skip dielectric layers already covered by bulk dielectric boxes
        # from stack.dielectrics. Re-extruding them would create overlapping
        # volumes that get consumed in the boolean pipeline, causing the
        # intended bulk material groups to disappear.
        layer_obj = stack.layers.get(layer_name)
        if layer_obj is not None and _is_covered_by_dielectric_box(layer_obj, stack):
            continue

        zmin = layer_info["zmin"]
        thickness = layer_info["thickness"]

        if thickness <= 0 or thickness < min_volume_thickness:
            logger.debug(
                "Skipping patterned dielectric layer '%s' with thickness %.3f um",
                layer_name,
                thickness,
            )
            continue

        surfaces = []
        surface_loop_mode = (
            curve_fit_mode
            if curve_fit_mode != "line" and layer_name in curve_layers
            else "line"
        )
        for pts_x, pts_y, holes in polys:
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel,
                pts_x,
                pts_y,
                zmin,
                holes=holes,
                loop_mode=surface_loop_mode,
                fit_tolerance_um=curve_fit_tolerance_um,
                min_points_for_curve_fit=curve_fit_min_points,
                corner_turn_threshold_deg=curve_fit_corner_angle_deg,
            )
            if surfacetag is not None:
                surfaces.append(surfacetag)

        if not surfaces:
            continue

        if len(surfaces) > 1:
            dimtags = [(2, s) for s in surfaces]
            fused, _ = kernel.fuse(
                [dimtags[0]],
                dimtags[1:],
                removeObject=True,
                removeTool=True,
            )
            kernel.synchronize()
            surfaces = [t for d, t in fused if d == 2]

        volumes = []
        for surfacetag in surfaces:
            result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
            volumes.append(result[1][1])

        if volumes:
            patterned_tags.setdefault(layer_name, []).extend(volumes)

    kernel.synchronize()
    return patterned_tags


def extract_pec_polygons(component, gds_layer: tuple[int, int]) -> list:
    """Extract polygons from an arbitrary GDS layer on a component.

    Uses the same klayout polygon parsing pattern as ``extract_geometry()``.

    Args:
        component: gdsfactory Component
        gds_layer: GDS layer tuple (layer, datatype) to extract polygons from

    Returns:
        List of (pts_x, pts_y, holes) tuples in microns
    """
    polygons_by_index = component.get_polygons()

    layout = component.kcl.layout
    index_to_gds = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    result = []
    for layer_index, polys in polygons_by_index.items():
        if index_to_gds.get(layer_index) != gds_layer:
            continue
        for poly in polys:
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue
            pts_x = [pt.x / 1000.0 for pt in points]
            pts_y = [pt.y / 1000.0 for pt in points]
            holes = []
            for hole_idx in range(poly.holes()):
                hole_pts = list(poly.each_point_hole(hole_idx))
                if len(hole_pts) >= 3:
                    hx = [pt.x / 1000.0 for pt in hole_pts]
                    hy = [pt.y / 1000.0 for pt in hole_pts]
                    holes.append((hx, hy))
            result.append((pts_x, pts_y, holes))

    return result


def add_pec_blocks(
    kernel,
    component,
    pec_configs: list[PECBlockConfig],
    stack: LayerStack,
) -> dict:
    """Add PEC block geometries to gmsh.

    For each PEC config:
    1. Extract polygons from the specified GDS layer
    2. Create surfaces at ``from_layer.zmin``
    3. Fuse overlapping surfaces
    4. Extrude to ``to_layer.zmax - from_layer.zmin``
    5. Remove volumes but keep shell surfaces
    6. Classify shells as xy/z using ``is_vertical_surface()``

    Args:
        kernel: gmsh OCC kernel
        component: gdsfactory Component
        pec_configs: List of PECBlockConfig objects
        stack: LayerStack with layer definitions

    Returns:
        Dict: ``{"pec_block_0": {"surfaces_xy": [...], "surfaces_z": [...]}, ...}``
    """
    pec_block_tags: dict[str, dict[str, list[int]]] = {}

    for idx, cfg in enumerate(pec_configs):
        block_name = f"pec_block_{idx}"
        from_layer = stack.layers.get(cfg.from_layer)
        to_layer = stack.layers.get(cfg.to_layer)
        if from_layer is None or to_layer is None:
            continue

        polys = extract_pec_polygons(component, cfg.gds_layer)
        if not polys:
            continue

        zmin = from_layer.zmin
        height = to_layer.zmax - from_layer.zmin

        # Create surfaces for each polygon
        surfaces = []
        for pts_x, pts_y, holes in polys:
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel, pts_x, pts_y, zmin, holes=holes
            )
            if surfacetag is not None:
                surfaces.append(surfacetag)

        if not surfaces:
            continue

        # Fuse overlapping surfaces
        if len(surfaces) > 1:
            dimtags = [(2, s) for s in surfaces]
            fused, _ = kernel.fuse(
                [dimtags[0]],
                dimtags[1:],
                removeObject=True,
                removeTool=True,
            )
            kernel.synchronize()
            surfaces = [t for d, t in fused if d == 2]

        # Extrude to create volumes
        volumes = []
        for surfacetag in surfaces:
            result = kernel.extrude([(2, surfacetag)], 0, 0, height)
            volumetag = result[1][1]
            volumes.append(volumetag)

        kernel.removeAllDuplicates()
        kernel.synchronize()

        # Extract shell surfaces and classify as xy/z
        xy_tags: list[int] = []
        z_tags: list[int] = []
        for volumetag in volumes:
            _, surfaceloops = kernel.getSurfaceLoops(volumetag)
            if surfaceloops:
                for tag in surfaceloops[0]:
                    if gmsh_utils.is_vertical_surface(tag):
                        z_tags.append(tag)
                    else:
                        xy_tags.append(tag)
            kernel.remove([(3, volumetag)])

        kernel.synchronize()

        pec_block_tags[block_name] = {
            "surfaces_xy": xy_tags,
            "surfaces_z": z_tags,
        }

    return pec_block_tags


def _finite_conductor_interface_name(
    source_id: str,
    face_kind: str,
) -> str:
    return f"{source_id}__SHELL__{face_kind.upper()}__TOTAL"


_DEFAULT_SURFACE_EPR_INSET_MARGINS_UM = (0.0, 0.05)
_SURFACE_EPR_MARGIN_TOL_UM = 1e-12


def _finite_conductor_band_interface_name(
    source_id: str,
    face_kind: str,
    label: str,
) -> str:
    return f"{source_id}__SHELL__{face_kind.upper()}__{label}"


def _surface_epr_distance_label(distance_um: float) -> str:
    if distance_um == 0.0:
        return "0"
    distance_nm = distance_um * 1000.0
    if distance_nm < 1000.0:
        value = round(distance_nm)
        if math.isclose(distance_nm, value, rel_tol=0.0, abs_tol=1e-9):
            return f"{value:g}NM"
        return f"{distance_nm:g}NM"
    value_um = round(distance_um)
    if math.isclose(distance_um, value_um, rel_tol=0.0, abs_tol=1e-9):
        return f"{value_um:g}UM"
    return f"{distance_um:g}UM"


def _surface_epr_band_label(lower_um: float, upper_um: float | None) -> str:
    lower_label = _surface_epr_distance_label(lower_um)
    if upper_um is None:
        return f"CORE_AFTER_{lower_label}"
    upper_label = _surface_epr_distance_label(upper_um)
    return f"BAND_{lower_label}_{upper_label}"


def _surface_epr_inset_margins(
    margins_um: Sequence[float] | None,
    *,
    max_margin_um: float | None = None,
) -> tuple[float, ...]:
    raw_margins = margins_um or _DEFAULT_SURFACE_EPR_INSET_MARGINS_UM
    margins = tuple(sorted({float(value) for value in raw_margins}))
    if any(value < 0.0 or not math.isfinite(value) for value in margins):
        raise ValueError("Surface EPR inset margins must be finite values >= 0.")
    if 0.0 not in margins:
        margins = (0.0, *margins)
    if max_margin_um is None:
        return margins
    return tuple(
        value
        for value in margins
        if value == 0.0 or value < max_margin_um - _SURFACE_EPR_MARGIN_TOL_UM
    )


def _surface_epr_split_ranges(
    margins_um: Sequence[float] | None,
    *,
    max_margin_um: float | None = None,
) -> tuple[tuple[str, float, float | None, str], ...]:
    margins = _surface_epr_inset_margins(
        margins_um,
        max_margin_um=max_margin_um,
    )
    if len(margins) < 2:
        return ()
    ranges: list[tuple[str, float, float | None, str]] = []
    for lower_um, upper_um in pairwise(margins):
        if upper_um <= lower_um:
            continue
        ranges.append(
            (
                _surface_epr_band_label(lower_um, upper_um),
                lower_um,
                upper_um,
                "band",
            )
        )
    last_margin = margins[-1]
    if last_margin > 0.0:
        ranges.append(
            (
                _surface_epr_band_label(last_margin, None),
                last_margin,
                None,
                "core",
            )
        )
    return tuple(ranges)


def _surface_boundary_points_3d(surface_tag: int) -> list[tuple[float, float, float]]:
    boundary = gmsh.model.getBoundary(
        [(2, surface_tag)],
        combined=False,
        oriented=True,
        recursive=False,
    )
    points: list[tuple[float, float, float]] = []
    for curve_dim, signed_curve_tag in boundary:
        if curve_dim != 1:
            continue
        curve_tag = abs(signed_curve_tag)
        endpoints = gmsh.model.getBoundary(
            [(1, curve_tag)],
            combined=False,
            oriented=True,
            recursive=False,
        )
        endpoint_tags = [tag for dim, tag in endpoints if dim == 0]
        if len(endpoint_tags) < 2:
            continue
        point_tag = endpoint_tags[0] if signed_curve_tag > 0 else endpoint_tags[-1]
        x, y, z = gmsh.model.getValue(0, point_tag, [])
        point = (float(x), float(y), float(z))
        if not points or point != points[-1]:
            points.append(point)
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    return points


def _surface_polygon_xy(surface_tag: int) -> tuple[ShapelyPolygon, float]:
    boundary_points = _surface_boundary_points_3d(surface_tag)
    points = [(x, y) for x, y, _z in boundary_points]
    z_values = [z for _x, _y, z in boundary_points]

    if len(points) < 3:
        raise ValueError(
            f"Surface EPR planar band split requires a polygonal shell face; "
            f"surface {surface_tag} has fewer than three boundary points."
        )
    if max(z_values) - min(z_values) > 1e-6:
        raise ValueError(
            "Surface EPR inset bands only support planar XY faces; "
            f"surface {surface_tag} is not coplanar in z."
        )
    polygon = ShapelyPolygon(points)
    if not polygon.is_valid:
        raise ValueError(
            "Surface EPR inset bands only support simple planar shell-face "
            f"polygons; surface {surface_tag} has an unsupported boundary."
        )
    if polygon.is_empty:
        raise ValueError(
            f"Surface EPR planar band split produced an empty face for {surface_tag}."
        )
    return polygon, sum(z_values) / len(z_values)


def _surface_polygon_vertical(
    surface_tag: int,
) -> tuple[ShapelyPolygon, tuple[float, float], tuple[float, float]]:
    boundary_points = _surface_boundary_points_3d(surface_tag)
    if len(boundary_points) < 3:
        raise ValueError(
            f"Surface EPR sidewall band split requires a polygonal shell face; "
            f"surface {surface_tag} has fewer than three boundary points."
        )

    origin_x, origin_y, _origin_z = boundary_points[0]
    axis_x = axis_y = 0.0
    max_dist = 0.0
    for index, (x0, y0, _z0) in enumerate(boundary_points):
        for x1, y1, _z1 in boundary_points[index + 1 :]:
            dist = math.hypot(x1 - x0, y1 - y0)
            if dist > max_dist:
                max_dist = dist
                axis_x = (x1 - x0) / dist
                axis_y = (y1 - y0) / dist
    if max_dist <= 1e-9:
        raise ValueError(
            "Surface EPR sidewall band split requires horizontal extent; "
            f"surface {surface_tag} has none."
        )

    coords = [
        ((x - origin_x) * axis_x + (y - origin_y) * axis_y, z)
        for x, y, z in boundary_points
    ]
    polygon = ShapelyPolygon(coords)
    if not polygon.is_valid:
        raise ValueError(
            "Surface EPR sidewall bands only support simple vertical shell-face "
            f"polygons; surface {surface_tag} has an unsupported boundary."
        )
    if polygon.is_empty:
        raise ValueError(
            f"Surface EPR sidewall band split produced an empty face for {surface_tag}."
        )
    return polygon, (origin_x, origin_y), (axis_x, axis_y)


def _iter_planar_polygons(geometry: Any) -> tuple[ShapelyPolygon, ...]:
    if isinstance(geometry, ShapelyPolygon):
        return (geometry,) if not geometry.is_empty else ()
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return ()
    return tuple(
        polygon
        for item in geoms
        for polygon in _iter_planar_polygons(item)
        if not polygon.is_empty
    )


def _surface_epr_band_geometry(
    polygon: ShapelyPolygon,
    lower_um: float,
    upper_um: float | None,
) -> Any:
    outer = (
        polygon
        if lower_um == 0.0
        else buffer(polygon, -lower_um, join_style="mitre")
    )
    if outer.is_empty or upper_um is None:
        return outer
    inner = buffer(polygon, -upper_um, join_style="mitre")
    return outer.difference(inner)


def _create_planar_surface_epr_faces(geometry: Any, z: float) -> list[int]:
    tags: list[int] = []
    for polygon in _iter_planar_polygons(geometry):
        if polygon.is_empty:
            continue
        exterior = list(polygon.exterior.coords[:-1])
        if len(exterior) < 3:
            continue
        holes = [
            (
                [float(x) for x, _y in interior.coords[:-1]],
                [float(y) for _x, y in interior.coords[:-1]],
            )
            for interior in polygon.interiors
            if len(interior.coords) >= 4
        ]
        tag = gmsh_utils.create_polygon_surface(
            gmsh.model.occ,
            [float(x) for x, _y in exterior],
            [float(y) for _x, y in exterior],
            z,
            holes=holes,
        )
        if tag is not None:
            tags.append(tag)
    return tags


def _create_mapped_surface_epr_faces(
    geometry: Any,
    map_point: Any,
) -> list[int]:
    tags: list[int] = []
    kernel = gmsh.model.occ
    for polygon in _iter_planar_polygons(geometry):
        if polygon.is_empty:
            continue
        exterior = list(polygon.exterior.coords[:-1])
        if len(exterior) < 3:
            continue
        loops: list[int] = []
        rings = (
            exterior,
            *(interior.coords[:-1] for interior in polygon.interiors),
        )
        for ring in rings:
            points = [
                kernel.addPoint(*map_point(float(u), float(v)), 0, -1)
                for u, v in ring
            ]
            if len(points) < 3:
                continue
            lines = [
                kernel.addLine(points[index], points[(index + 1) % len(points)], -1)
                for index in range(len(points))
            ]
            loops.append(kernel.addCurveLoop(lines, tag=-1))
        if loops:
            tags.append(kernel.addPlaneSurface(loops, tag=-1))
    return tags


def _finite_conductor_split_records(
    *,
    source_id: str,
    face_kind: str,
    base_metadata: Mapping[str, Any],
    split_tags_by_label: Mapping[str, Sequence[int]],
    split_ranges: Sequence[tuple[str, float, float | None, str]],
) -> dict[str, dict[str, Any]]:
    parent_name = _finite_conductor_interface_name(source_id, face_kind)
    records: dict[str, dict[str, Any]] = {}
    for label, lower_um, upper_um, summary_kind in split_ranges:
        split_tags = split_tags_by_label.get(label, ())
        if not split_tags:
            continue
        name = _finite_conductor_band_interface_name(source_id, face_kind, label)
        records[name] = {
            **dict(base_metadata),
            "tags": sorted(set(split_tags)),
            "parent_interface_id": parent_name,
            "surface_epr_band_label": label,
            "surface_epr_band_min_um": lower_um,
            "surface_epr_band_max_um": upper_um,
            "surface_epr_exclude_below_um": lower_um,
            "surface_epr_summary_kind": summary_kind,
            "postprocessing_only": True,
        }
    return records


def _split_finite_conductor_planar_interface(
    *,
    source_id: str,
    face_kind: str,
    tags: Sequence[int],
    base_metadata: Mapping[str, Any],
    surface_epr_inset_margins_um: Sequence[float] | None,
) -> dict[str, dict[str, Any]]:
    split_ranges = _surface_epr_split_ranges(surface_epr_inset_margins_um)
    if not split_ranges:
        return {}
    split_tags_by_label: dict[str, list[int]] = {
        label: [] for label, _lower_um, _upper_um, _summary_kind in split_ranges
    }
    removed_parent_tags: list[tuple[int, int]] = []
    for tag in tags:
        polygon, z = _surface_polygon_xy(tag)
        for label, lower_um, upper_um, _summary_kind in split_ranges:
            split_tags_by_label[label].extend(
                _create_planar_surface_epr_faces(
                    _surface_epr_band_geometry(polygon, lower_um, upper_um),
                    z,
                )
            )
        removed_parent_tags.append((2, tag))

    if removed_parent_tags:
        gmsh.model.occ.remove(removed_parent_tags, recursive=False)
        gmsh.model.occ.synchronize()

    return _finite_conductor_split_records(
        source_id=source_id,
        face_kind=face_kind,
        base_metadata=base_metadata,
        split_tags_by_label=split_tags_by_label,
        split_ranges=split_ranges,
    )


def _split_finite_conductor_vertical_interface(
    *,
    source_id: str,
    face_kind: str,
    tags: Sequence[int],
    base_metadata: Mapping[str, Any],
    surface_epr_inset_margins_um: Sequence[float] | None,
    max_margin_um: float | None,
) -> dict[str, dict[str, Any]]:
    split_ranges = _surface_epr_split_ranges(
        surface_epr_inset_margins_um,
        max_margin_um=max_margin_um,
    )
    if not split_ranges:
        return {}
    split_tags_by_label: dict[str, list[int]] = {
        label: [] for label, _lower_um, _upper_um, _summary_kind in split_ranges
    }
    removed_parent_tags: list[tuple[int, int]] = []
    for tag in tags:
        polygon, origin, axis = _surface_polygon_vertical(tag)
        origin_x, origin_y = origin
        axis_x, axis_y = axis

        def map_point(
            u: float,
            z: float,
            origin_x: float = origin_x,
            origin_y: float = origin_y,
            axis_x: float = axis_x,
            axis_y: float = axis_y,
        ) -> tuple[float, float, float]:
            return origin_x + u * axis_x, origin_y + u * axis_y, z

        for label, lower_um, upper_um, _summary_kind in split_ranges:
            split_tags_by_label[label].extend(
                _create_mapped_surface_epr_faces(
                    _surface_epr_band_geometry(polygon, lower_um, upper_um),
                    map_point,
                )
            )
        removed_parent_tags.append((2, tag))

    if removed_parent_tags:
        gmsh.model.occ.remove(removed_parent_tags, recursive=False)
        gmsh.model.occ.synchronize()

    return _finite_conductor_split_records(
        source_id=source_id,
        face_kind=face_kind,
        base_metadata=base_metadata,
        split_tags_by_label=split_tags_by_label,
        split_ranges=split_ranges,
    )


def _finite_conductor_shell_interfaces(
    *,
    layer_name: str,
    source_id: str,
    surface_tags: Sequence[int],
    stack: LayerStack | None,
    surface_epr_inset_margins_um: Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    layer = None if stack is None else stack.layers.get(layer_name)
    if layer is None:
        return {}

    by_face: dict[str, list[int]] = {"bottom": [], "top": [], "sidewall": []}
    for tag in surface_tags:
        by_face.setdefault(_finite_conductor_face_kind(tag, layer), []).append(tag)

    records: dict[str, dict[str, Any]] = {}
    for face_kind, tags in by_face.items():
        if not tags:
            continue
        metadata = {
            "tags": sorted(set(tags)),
            "surface_epr": True,
            "face_kind": face_kind,
            "geometry_kind": "vertical_ruled"
            if face_kind == "sidewall"
            else "planar_xy",
            "representation": "B",
            "metal_body_id": source_id,
            "source_id": source_id,
            "postprocessing_only": True,
            "surface_epr_band_min_um": 0.0,
            "surface_epr_band_max_um": None,
            "surface_epr_exclude_below_um": 0.0,
        }
        if face_kind in {"bottom", "top"}:
            split_records = _split_finite_conductor_planar_interface(
                source_id=source_id,
                face_kind=face_kind,
                tags=tags,
                base_metadata=metadata,
                surface_epr_inset_margins_um=surface_epr_inset_margins_um,
            )
            if split_records:
                records.update(split_records)
            else:
                name = _finite_conductor_interface_name(source_id, face_kind)
                records[name] = metadata
            continue
        if face_kind == "sidewall":
            split_records = _split_finite_conductor_vertical_interface(
                source_id=source_id,
                face_kind=face_kind,
                tags=tags,
                base_metadata=metadata,
                surface_epr_inset_margins_um=surface_epr_inset_margins_um,
                max_margin_um=float(layer.thickness),
            )
            if split_records:
                records.update(split_records)
            else:
                name = _finite_conductor_interface_name(source_id, face_kind)
                records[name] = metadata
            continue
        name = _finite_conductor_interface_name(source_id, face_kind)
        records[name] = metadata
    return records


def _finite_conductor_face_kind(tag: int, layer: Any) -> str:
    if gmsh_utils.is_vertical_surface(tag):
        return "sidewall"
    try:
        bbox = gmsh.model.getBoundingBox(2, tag)
    except Exception:
        return "unsupported_3d"
    z_mid = (float(bbox[2]) + float(bbox[5])) / 2.0
    return "bottom" if abs(z_mid - layer.zmin) <= abs(z_mid - layer.zmax) else "top"


def build_entities(
    metal_tags: dict,
    dielectric_tags: dict,
    patterned_dielectric_tags: dict | None,
    port_tags: dict,
    port_info: list,
    pec_block_tags: dict | None = None,
    stack: LayerStack | None = None,
    activated_regions: Sequence[ActivatedRegion] | None = None,
    shaped_dielectric_names: set[str] | None = None,
    surface_epr_inset_margins_um: Sequence[float] | None = None,
) -> list[gmsh_utils.Entity]:
    """Convert geometry tag dicts into Entity objects for the boolean pipeline.

    Mesh-order convention (lower = higher priority, gets cut first):
        0  - conductor (2D PEC) surfaces and PEC block surfaces
        1  - via volumes (3D, higher priority than dielectrics) and port surfaces
        2  - patterned dielectric volumes from stack layers
        3  - background dielectric boxes (non-airbox volumes)
        4  - airbox volume (lowest priority, carved by everything else)

    Args:
        metal_tags: from ``add_metals()``
        dielectric_tags: from ``add_dielectrics()``
        patterned_dielectric_tags: from ``add_patterned_dielectrics()``
        port_tags: from ``add_ports()``
        port_info: metadata list from ``add_ports()``
        pec_block_tags: from ``add_pec_blocks()``, optional
        stack: LayerStack for distinguishing via vs conductor vs shaped
            dielectric layers
        activated_regions: Explicit region roles from the public simulation API.
        surface_epr_inset_margins_um: Surface EPR inset margins used when
            lowering finite conductor shells into postprocessing faces.

    Returns:
        List of Entity objects ready for ``run_boolean_pipeline()``.
    """
    Entity = gmsh_utils.Entity
    entities: list[gmsh_utils.Entity] = []

    # Build set of via and shaped-dielectric layer names for quick lookup
    via_layers: set[str] = set()
    shaped_dielectric_layers = set(shaped_dielectric_names or ())
    if stack:
        via_layers = {
            n for n, layer in stack.layers.items() if layer.layer_type == "via"
        }
    outer_vacuum_layers = {
        region.layer
        for region in activated_regions or ()
        if region.role == "outer_vacuum"
    }

    # --- Conductors, vias, and shaped dielectrics ---
    for layer_name, tag_info in metal_tags.items():
        is_via = layer_name in via_layers
        is_shaped_dielectric = layer_name in shaped_dielectric_layers

        # PEC / zero-thickness surfaces
        if tag_info.get("surfaces_xy"):
            # Via PEC surfaces get higher priority (lower mesh_order) so they
            # are processed first and survive boolean cuts against conductor
            # shell surfaces that sit at the same z-height.
            pec_mesh_order = -1 if is_via else 0
            surface_tags = tag_info["surfaces_xy"]
            if len(surface_tags) == 1:
                entities.append(
                    Entity(
                        name=f"{layer_name}_pec",
                        dim=2,
                        mesh_order=pec_mesh_order,
                        tags=surface_tags,
                    )
                )
            else:
                for index, surface_tag in enumerate(surface_tags):
                    entities.append(
                        Entity(
                            name=f"{layer_name}_pec_{index}",
                            dim=2,
                            mesh_order=pec_mesh_order,
                            tags=[surface_tag],
                        )
                    )

        if tag_info.get("volumes"):
            if is_via:
                # Via volumes: 3D entities, higher priority than dielectrics
                via_vol_tags = [
                    item for item in tag_info["volumes"] if isinstance(item, int)
                ]
                if via_vol_tags:
                    entities.append(
                        Entity(
                            name=layer_name,
                            dim=3,
                            mesh_order=1,
                            tags=via_vol_tags,
                        )
                    )
            elif is_shaped_dielectric:
                # Shaped dielectric volumes: 3D entities with same priority as
                # vias so they carve out of surrounding dielectric boxes.
                shaped_vol_tags = [
                    item for item in tag_info["volumes"] if isinstance(item, int)
                ]
                if shaped_vol_tags:
                    entities.append(
                        Entity(
                            name=layer_name,
                            dim=3,
                            mesh_order=1,
                            tags=shaped_vol_tags,
                        )
                    )
            else:
                # Volumetric conductors: shell surfaces (volume already removed)
                shell_tags = []
                interface_surfaces: dict[str, dict[str, Any]] = {}
                terminal_shells: dict[str, dict[str, Any]] = {}
                for volume_index, item in enumerate(tag_info["volumes"]):
                    if not isinstance(item, tuple):
                        continue
                    _volumetag, surface_tags = item
                    surface_tags = list(surface_tags)
                    shell_tags.extend(surface_tags)
                    shell_name = f"{layer_name}__CONDUCTOR_SHELL_{volume_index}"
                    volume_interfaces = _finite_conductor_shell_interfaces(
                        layer_name=layer_name,
                        source_id=shell_name,
                        surface_tags=surface_tags,
                        stack=stack,
                        surface_epr_inset_margins_um=surface_epr_inset_margins_um,
                    )
                    terminal_tags: list[int] = []
                    for name, interface in volume_interfaces.items():
                        tags = list(interface.get("tags", ()))
                        terminal_tags.extend(tags)
                        existing = interface_surfaces.get(name)
                        if existing is None:
                            interface_surfaces[name] = {**interface, "tags": tags}
                        else:
                            existing["tags"] = sorted(
                                set(existing.get("tags", ())) | set(tags)
                            )
                    if terminal_tags:
                        terminal_shells[shell_name] = {
                            "tags": sorted(set(terminal_tags)),
                            "layer": layer_name,
                            "source": "finite_conductor_terminal_shell",
                            "source_id": shell_name,
                        }
                if interface_surfaces:
                    tag_info["surface_epr_interfaces"] = interface_surfaces
                    tag_info["terminal_shells"] = terminal_shells
                    for name, interface in interface_surfaces.items():
                        entities.append(
                            Entity(
                                name=name,
                                dim=2,
                                mesh_order=0,
                                tags=list(interface["tags"]),
                            )
                        )
                elif shell_tags:
                    xy_tags = []
                    z_tags = []
                    for tag in shell_tags:
                        if gmsh_utils.is_vertical_surface(tag):
                            z_tags.append(tag)
                        else:
                            xy_tags.append(tag)
                    if xy_tags:
                        entities.append(
                            Entity(
                                name=f"{layer_name}_xy",
                                dim=2,
                                mesh_order=0,
                                tags=xy_tags,
                            )
                        )
                    if z_tags:
                        entities.append(
                            Entity(
                                name=f"{layer_name}_z",
                                dim=2,
                                mesh_order=0,
                                tags=z_tags,
                            )
                        )

    # --- PEC block surfaces (dim=2, highest priority) ---
    if pec_block_tags:
        for block_name, tag_info in pec_block_tags.items():
            if tag_info["surfaces_xy"]:
                entities.append(
                    Entity(
                        name=f"{block_name}_xy",
                        dim=2,
                        mesh_order=0,
                        tags=tag_info["surfaces_xy"],
                    )
                )
            if tag_info["surfaces_z"]:
                entities.append(
                    Entity(
                        name=f"{block_name}_z",
                        dim=2,
                        mesh_order=0,
                        tags=tag_info["surfaces_z"],
                    )
                )

    # --- Port surfaces (dim=2) ---
    for port_name, surf_tags in port_tags.items():
        port_num = int(port_name[1:])
        info = next(
            (p for p in port_info if p["portnumber"] == port_num),
            None,
        )
        if info and info.get("type") == "cpw":
            # One entity per CPW element
            for i, tag in enumerate(surf_tags):
                entities.append(
                    Entity(
                        name=f"{port_name}_E{i}",
                        dim=2,
                        mesh_order=-1,
                        tags=[tag],
                    )
                )
        else:
            entities.append(
                Entity(
                    name=port_name,
                    dim=2,
                    mesh_order=-1,
                    tags=surf_tags,
                )
            )

    # --- Dielectric volumes (dim=3) ---
    if patterned_dielectric_tags:
        for layer_name, vol_tags in patterned_dielectric_tags.items():
            entities.append(
                Entity(
                    name=layer_name,
                    dim=3,
                    mesh_order=2,
                    tags=vol_tags,
                )
            )

    patterned_names = set(patterned_dielectric_tags or {})
    for material, vol_tags in dielectric_tags.items():
        # If a patterned dielectric entity already uses this name, prefer the
        # patterned volume entity to avoid name collisions in group assignment.
        entity_name = "air" if material == "airbox" else str(material)
        if entity_name in patterned_names:
            continue
        order = 4 if entity_name == "air" or entity_name in outer_vacuum_layers else 3
        entities.append(
            Entity(
                name=entity_name,
                dim=3,
                mesh_order=order,
                tags=vol_tags,
            )
        )

    return entities


def _direction_to_list(
    direction: tuple[float, float, float] | list[float],
) -> list[float]:
    """Return a JSON-ready copy of a normalized Palace direction vector."""
    return [float(component) for component in direction]


def _orientation_basis(
    orientation: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return longitudinal and transverse unit vectors from port orientation."""
    angle = math.radians(float(orientation) if orientation is not None else 0.0)
    longitudinal = (math.cos(angle), math.sin(angle))
    transverse = (-math.sin(angle), math.cos(angle))
    return longitudinal, transverse


def _horizontal_port_corners(
    *,
    center: tuple[float, float],
    length: float,
    width: float,
    orientation: float | None,
) -> list[tuple[float, float]]:
    """Build the XY corners of a generated in-plane port sheet."""
    longitudinal, transverse = _orientation_basis(orientation)
    cx, cy = center
    half_length = length / 2
    half_width = width / 2

    offsets = (
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    )
    return [
        (
            cx + longitudinal[0] * lscale + transverse[0] * wscale,
            cy + longitudinal[1] * lscale + transverse[1] * wscale,
        )
        for lscale, wscale in offsets
    ]


def _horizontal_port_bbox(
    corners: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _create_horizontal_port_sheet(
    kernel,
    *,
    center: tuple[float, float],
    length: float,
    width: float,
    orientation: float | None,
    z: float,
) -> tuple[int | None, list[tuple[float, float]]]:
    """Create an in-plane port sheet whose geometry follows port orientation."""
    corners = _horizontal_port_corners(
        center=center,
        length=length,
        width=width,
        orientation=orientation,
    )
    surfacetag = gmsh_utils.create_polygon_surface(
        kernel,
        [corner[0] for corner in corners],
        [corner[1] for corner in corners],
        z,
    )
    return surfacetag, corners


def add_ports(
    kernel,
    ports: list[PalacePort],
    stack: LayerStack,
    domain_bbox: tuple[float, float, float, float] | None = None,
    domain_bounds: tuple[float, float, float, float, float, float] | None = None,
    simulation_layers: SimulationLayerCatalog | None = None,
    authored_sheet_polygons: (
        dict[tuple[int, int], list[AuthoredSheetPolygon]] | None
    ) = None,
) -> tuple[dict, list]:
    """Add port surfaces to gmsh.

    Args:
        kernel: gmsh OCC kernel
        ports: List of PalacePort objects (single or multi-element)
        stack: Layer stack
        domain_bbox: (xmin, ymin, xmax, ymax) of the simulation domain
            (geometry bbox with margin applied). Required when any port
            has ``max_size=True``.
        domain_bounds: (xmin, ymin, zmin, xmax, ymax, zmax) of the outer
            simulation domain. When provided, ``max_size=True`` waveports
            are clipped to this exact 3D domain envelope.
        simulation_layers: PDK-declared simulation-only layer catalog used when
            a port requests layout-authored sheets.
        authored_sheet_polygons: Polygons extracted from registered simulation
            layers, keyed by full GDS layer tuple.

    Returns:
        (port_tags dict, port_info list)

    For single-element ports: port_tags["P{num}"] = [surface_tag]
    For multi-element ports: port_tags["P{num}"] = [surface_tag, surface_tag, ...]
    """
    port_tags = {}  # "P{num}" -> [surface_tag(s)]
    port_info = []
    port_num = 1
    authored_polygons = authored_sheet_polygons or {}

    for port in ports:
        if port.multi_element:
            # Multi-element port (CPW)
            if port.layer is None or port.centers is None or port.directions is None:
                continue
            target_layer = stack.layers.get(port.layer)
            if target_layer is None:
                continue

            zmin = target_layer.zmin
            sheet_length = port.length or port.width
            sheet_layer = None
            if not port.generate_sheet:
                if simulation_layers is None:
                    raise ValueError(
                        f"CPW port '{port.name}' requested authored sheets; call "
                        "set_simulation_layers() before meshing."
                    )
                if port.sheet_gds_layer is None:
                    raise ValueError(
                        f"CPW port '{port.name}' requested authored sheets, but its "
                        "gdsfactory port layer was not recorded."
                    )
                sheet_gds_layer = port.sheet_gds_layer
                sheet_layer = simulation_layers.for_gds_layer(sheet_gds_layer)
                if sheet_layer is None:
                    raise ValueError(
                        f"CPW port '{port.name}' uses GDS layer "
                        f"{sheet_gds_layer}, which is not registered as a "
                        "simulation solver sheet layer."
                    )
                zmin = sheet_layer.resolve_z(stack)

            surfaces = []
            elements = []
            for (cx, cy), direction in zip(port.centers, port.directions, strict=True):
                if port.generate_sheet:
                    surf, corners = _create_horizontal_port_sheet(
                        kernel,
                        center=(cx, cy),
                        length=sheet_length,
                        width=port.width,
                        orientation=port.orientation,
                        z=zmin,
                    )
                    element_metadata: dict[str, object] = {"corners": corners}
                else:
                    sheet_gds_layer = port.sheet_gds_layer
                    if sheet_gds_layer is None:
                        raise ValueError(
                            f"CPW port '{port.name}' requested authored sheets, but "
                            "its gdsfactory port layer was not recorded."
                        )
                    polygon = select_authored_sheet_polygon(
                        authored_polygons,
                        gds_layer=sheet_gds_layer,
                        center=(cx, cy),
                        port_name=port.name,
                    )
                    surf = create_authored_sheet_surface(kernel, polygon, z=zmin)
                    element_metadata = {
                        "bbox": polygon.bbox,
                        "sheet_source": "layout-authored",
                        "sheet_layer": sheet_layer.name if sheet_layer else None,
                        "sheet_gds_layer": list(polygon.gds_layer),
                    }
                if surf is None:
                    continue
                surface_idx = len(surfaces)
                surfaces.append(surf)
                elements.append(
                    {
                        "surface_idx": surface_idx,
                        "direction": _direction_to_list(direction),
                        **element_metadata,
                    }
                )

            if not surfaces:
                continue

            port_tags[f"P{port_num}"] = surfaces

            port_info.append(
                {
                    "portnumber": port_num,
                    "name": port.name,
                    "Z0": port.impedance,
                    "type": "cpw",
                    "elements": elements,
                    "width": port.width,
                    "length": sheet_length,
                    "orientation": port.orientation,
                    "zmin": zmin,
                    "zmax": zmin,
                    "sheet_source": "generated"
                    if port.generate_sheet
                    else "layout-authored",
                }
            )

        elif port.geometry == PortGeometry.VIA:
            # Via port: vertical between two layers
            if port.from_layer is None or port.to_layer is None:
                continue
            from_layer = stack.layers.get(port.from_layer)
            to_layer = stack.layers.get(port.to_layer)
            if from_layer is None or to_layer is None:
                continue

            x, y = port.center
            hw = port.width / 2

            if from_layer.zmin < to_layer.zmin:
                zmin = from_layer.zmax
                zmax = to_layer.zmin
            else:
                zmin = to_layer.zmax
                zmax = from_layer.zmin

            # Create vertical port surface
            angle = port.orientation % 360
            is_y_axis = 45 <= angle < 135 or 225 <= angle < 315
            if not is_y_axis:
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x, y - hw, zmin, x, y + hw, zmax
                )
            else:
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hw, y, zmin, x + hw, y, zmax
                )

            direction = port.direction
            if direction is None:
                raise ValueError(
                    f"Via port '{port.name}' is missing direction metadata"
                )
            port_tags[f"P{port_num}"] = [surfacetag]
            port_info.append(
                {
                    "portnumber": port_num,
                    "name": port.name,
                    "Z0": port.impedance,
                    "type": "via",
                    "direction": _direction_to_list(direction),
                    "length": zmax - zmin,
                    "width": port.width,
                    "xmin": x if not is_y_axis else x - hw,
                    "xmax": x if not is_y_axis else x + hw,
                    "ymin": y - hw if not is_y_axis else y,
                    "ymax": y + hw if not is_y_axis else y,
                    "zmin": zmin,
                    "zmax": zmax,
                }
            )

        else:
            # Inplane port: horizontal on single layer
            if port.layer is None:
                continue
            target_layer = stack.layers.get(port.layer)
            if target_layer is None:
                continue

            x, y = port.center
            hw = port.width / 2
            zmin = target_layer.zmin
            zmax = target_layer.zmax

            if port.port_type == PortType.LUMPED:
                length = port.length or port.width
                width = port.width
                sheet_layer = None
                if port.generate_sheet:
                    surfacetag, corners = _create_horizontal_port_sheet(
                        kernel,
                        center=(x, y),
                        length=length,
                        width=width,
                        orientation=port.orientation,
                        z=zmin,
                    )
                    xmin, ymin, xmax, ymax = _horizontal_port_bbox(corners)
                    sheet_metadata: dict[str, object] = {
                        "corners": corners,
                        "sheet_source": "generated",
                    }
                else:
                    if simulation_layers is None:
                        raise ValueError(
                            f"Port '{port.name}' requested an authored sheet; call "
                            "set_simulation_layers() before meshing."
                        )
                    if port.sheet_gds_layer is None:
                        raise ValueError(
                            f"Port '{port.name}' requested an authored sheet, but "
                            "its gdsfactory port layer was not recorded."
                        )
                    sheet_gds_layer = port.sheet_gds_layer
                    sheet_layer = simulation_layers.for_gds_layer(sheet_gds_layer)
                    if sheet_layer is None:
                        raise ValueError(
                            f"Port '{port.name}' uses GDS layer "
                            f"{sheet_gds_layer}, which is not registered as a "
                            "simulation solver sheet layer."
                        )
                    zmin = sheet_layer.resolve_z(stack)
                    polygon = select_authored_sheet_polygon(
                        authored_polygons,
                        gds_layer=sheet_gds_layer,
                        center=(x, y),
                        port_name=port.name,
                    )
                    surfacetag = create_authored_sheet_surface(kernel, polygon, z=zmin)
                    xmin, ymin, xmax, ymax = polygon.bbox
                    sheet_metadata = {
                        "bbox": polygon.bbox,
                        "sheet_source": "layout-authored",
                        "sheet_layer": sheet_layer.name,
                        "sheet_gds_layer": list(polygon.gds_layer),
                    }
                if surfacetag is None:
                    continue

                direction = port.direction
                if direction is None:
                    raise ValueError(
                        f"Lumped port '{port.name}' is missing direction metadata"
                    )
                port_tags[f"P{port_num}"] = [surfacetag]
                port_info.append(
                    {
                        "portnumber": port_num,
                        "name": port.name,
                        "Z0": port.impedance,
                        "type": "lumped",
                        "direction": _direction_to_list(direction),
                        "length": length,
                        "width": width,
                        "orientation": port.orientation,
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "zmin": zmin,
                        "zmax": zmin,
                        **sheet_metadata,
                    }
                )
            else:
                # Build a robust z-envelope for waveports. When synthetic
                # dielectric boxes are disabled, stack.get_z_range() can miss
                # patterned core levels; include layer extents as well.
                layer_zmin, layer_zmax = stack.get_z_range()
                if stack.layers:
                    zmin_layers = min(layer.zmin for layer in stack.layers.values())
                    zmax_layers = max(layer.zmax for layer in stack.layers.values())
                    layer_zmin = min(layer_zmin, zmin_layers)
                    layer_zmax = max(layer_zmax, zmax_layers)

                if port.max_size and domain_bounds is not None:
                    # Fill the full 3D simulation domain.
                    _, _, zmin, _, _, zmax = domain_bounds
                elif port.max_size:
                    # Backward-compatible fallback when 3D bounds are unavailable.
                    zmin = layer_zmin
                    zmax = layer_zmax
                else:
                    zmin = zmin - port.z_margin
                    zmax = zmax + port.z_margin
                    zmin = max(zmin, layer_zmin)
                    zmax = min(zmax, layer_zmax)

                # Guard against inverted/degenerate z extents.
                if zmax <= zmin:
                    z_center = target_layer.zmin + 0.5 * target_layer.thickness
                    z_half = max(port.z_margin, 0.01)
                    zmin = z_center - z_half
                    zmax = z_center + z_half

                angle = port.orientation % 360
                is_y_axis = 45 <= angle < 135 or 225 <= angle < 315

                if port.max_size:
                    if domain_bounds is not None:
                        dom_xmin, dom_ymin, _, dom_xmax, dom_ymax, _ = domain_bounds
                    elif domain_bbox is not None:
                        dom_xmin, dom_ymin, dom_xmax, dom_ymax = domain_bbox
                    else:
                        raise ValueError(
                            f"Port '{port.name}' has max_size=True but "
                            "domain bounds were not provided to add_ports()"
                        )
                    if is_y_axis:
                        xmin = dom_xmin
                        xmax = dom_xmax
                        ymin = y
                        ymax = y
                    else:
                        xmin = x
                        xmax = x
                        ymin = dom_ymin
                        ymax = dom_ymax
                elif is_y_axis:
                    xmin = x - hw - port.lateral_margin
                    xmax = x + hw + port.lateral_margin
                    ymin = y
                    ymax = y
                else:
                    xmin = x
                    xmax = x
                    ymin = y - hw - port.lateral_margin
                    ymax = y + hw + port.lateral_margin
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, xmin, ymin, zmin, xmax, ymax, zmax
                )
                port_tags[f"P{port_num}"] = [surfacetag]

                effective_width = xmax - xmin if is_y_axis else ymax - ymin

                port_info.append(
                    {
                        "portnumber": port_num,
                        "type": "waveport",
                        "width": effective_width,
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "zmin": zmin,
                        "zmax": zmax,
                    }
                )
        port_num += 1

    kernel.synchronize()

    return port_tags, port_info


__all__ = [
    "GeometryData",
    "MetalGeometryResult",
    "add_dielectrics",
    "add_metals",
    "add_pec_blocks",
    "add_ports",
    "build_entities",
    "extract_geometry",
    "extract_pec_polygons",
    "get_layer_info",
]
