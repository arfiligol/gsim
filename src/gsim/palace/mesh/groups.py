"""Physical group assignment for Palace mesh generation.

This module builds the ``groups`` dict consumed by the config generator
from the ``pg_map`` produced by ``run_boolean_pipeline``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import gmsh

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def assign_physical_groups(
    kernel,
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    entities: list[gmsh_utils.Entity],
    pg_map: dict[str, int],
    _stack: LayerStack,
    pec_block_tags: dict | None = None,
) -> dict:
    """Build the ``groups`` dict from the boolean-pipeline result.

    Args:
        kernel: gmsh OCC kernel
        metal_tags: Metal layer tags from add_metals()
        dielectric_tags: Dielectric material tags from add_dielectrics()
        port_tags: Port surface tags (may have multiple surfaces for CPW)
        port_info: Port metadata including type info
        entities: Entity list used in run_boolean_pipeline
        pg_map: name -> physical-group tag returned by run_boolean_pipeline
        _stack: Layer stack used to identify via layers

    Returns:
        Dict with the same schema as before::

            {
                "volumes": {name: {"phys_group": int, "tags": [int]}},
                "conductor_surfaces": {name: {"phys_group": int, "tags": [int]}},
                "pec_surfaces": {name: {"phys_group": int, "tags": [int]}},
                "port_surfaces": {name: ...},
                "boundary_surfaces": {name: {"phys_group": int, "tags": [int]}},
            }
    """
    groups: dict[str, dict] = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    # Helper: entity name -> (phys_group, surface_tags)
    entity_by_name: dict[str, gmsh_utils.Entity] = {e.name: e for e in entities}

    def _surface_bbox(
        tags: list[int],
    ) -> tuple[float, float, float, float, float, float] | None:
        bboxes: list[tuple[float, float, float, float, float, float]] = []
        for tag in tags:
            with contextlib.suppress(Exception):
                bboxes.append(kernel.getBoundingBox(2, tag))
        if not bboxes:
            return None
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            min(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
            max(bbox[4] for bbox in bboxes),
            max(bbox[5] for bbox in bboxes),
        )

    def _surface_tags_for_physical_group(pg_tag: int) -> list[int]:
        with contextlib.suppress(Exception):
            return list(gmsh.model.getEntitiesForPhysicalGroup(2, pg_tag))
        return []

    def _append_pec_surface(
        *,
        key: str,
        entity_name: str,
        entity: gmsh_utils.Entity,
        layer_name: str,
        island_index: int | None = None,
    ) -> None:
        pg = pg_map.get(entity_name)
        if pg is None:
            return
        surf_tags = [tag for dim, tag in entity.dimtags if dim == 2]
        if not surf_tags:
            return
        surface_info = {
            "phys_group": pg,
            "tags": surf_tags,
            "layer": layer_name,
            "physical_name": key,
        }
        bbox = _surface_bbox(surf_tags)
        if bbox is not None:
            surface_info["bbox"] = bbox
        if island_index is not None:
            surface_info["island_index"] = island_index
        groups["pec_surfaces"][key] = surface_info

    # Build set of via and shaped-dielectric layer names
    via_layers: set[str] = set()
    shaped_dielectric_layers: set[str] = set()
    if _stack:
        via_layers = {
            n for n, layer in _stack.layers.items() if layer.layer_type == "via"
        }

    # Auto-detected shaped dielectrics from add_metals() metadata
    _shaped_meta = metal_tags.get("__shaped_dielectrics__")
    if isinstance(_shaped_meta, set):
        shaped_dielectric_layers |= _shaped_meta

    # --- Volumes (dielectrics + airbox) ---
    for material in dielectric_tags:
        entity_name = "air" if material == "airbox" else material
        group_name = "air" if material == "airbox" else material
        entity = entity_by_name.get(entity_name)
        pg = pg_map.get(entity_name)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                groups["volumes"][group_name] = {
                    "phys_group": pg,
                    "tags": vol_tags,
                }

    # --- Via volumes (3D material regions with conductivity) ---
    for layer_name in via_layers:
        entity = entity_by_name.get(layer_name)
        pg = pg_map.get(layer_name)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                groups["volumes"][layer_name] = {
                    "phys_group": pg,
                    "tags": vol_tags,
                    "is_via": True,
                }

    # --- Shaped dielectric volumes (3D polygon-extruded dielectrics) ---
    for layer_name in shaped_dielectric_layers:
        entity = entity_by_name.get(layer_name)
        pg = pg_map.get(layer_name)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                groups["volumes"][layer_name] = {
                    "phys_group": pg,
                    "tags": vol_tags,
                    "is_shaped_dielectric": True,
                }

    # --- PEC surfaces (planar conductors) ---
    for layer_name, tag_info in metal_tags.items():
        if layer_name.startswith("__") and layer_name.endswith("__"):
            continue
        if tag_info.get("surfaces_xy"):
            pec_name = f"{layer_name}_pec"
            entity = entity_by_name.get(pec_name)
            if entity:
                _append_pec_surface(
                    key=layer_name,
                    entity_name=pec_name,
                    entity=entity,
                    layer_name=layer_name,
                )
            else:
                split_entities = sorted(
                    (
                        name,
                        split_entity,
                    )
                    for name, split_entity in entity_by_name.items()
                    if name.startswith(f"{layer_name}_pec_")
                )
                for index, (split_name, split_entity) in enumerate(split_entities):
                    _append_pec_surface(
                        key=split_name,
                        entity_name=split_name,
                        entity=split_entity,
                        layer_name=layer_name,
                        island_index=index,
                    )

            # Always collect refinement lines for planar conductors — either
            # from explicit refinement_lines (if they survived boolean) or
            # from the PEC surface boundary curves (fallback).
            pec_entity = entity_by_name.get(pec_name)
            refinement_lines = tag_info.get("refinement_lines", [])
            if refinement_lines:
                valid_lines: list[int] = []
                for ltag in refinement_lines:
                    try:
                        kernel.getBoundingBox(1, ltag)
                        valid_lines.append(ltag)
                    except Exception:
                        pass  # Curve was merged / renumbered
                if len(valid_lines) < len(refinement_lines):
                    # Some curves were merged — find replacements by bbox.
                    all_curves = list(kernel.getEntities(1))
                    curve_bboxes: dict[int, tuple] = {}
                    for _, ctag in all_curves:
                        with contextlib.suppress(Exception):
                            curve_bboxes[ctag] = kernel.getBoundingBox(1, ctag)
                    for ltag in refinement_lines:
                        if ltag in valid_lines:
                            continue
                        try:
                            old_bbox = kernel.getBoundingBox(1, ltag)
                        except Exception:
                            continue
                        for ctag, bbox in curve_bboxes.items():
                            if ctag in valid_lines:
                                continue
                            if all(
                                abs(a - b) < 0.01
                                for a, b in zip(bbox, old_bbox, strict=True)
                            ):
                                valid_lines.append(ctag)
                                break
                if valid_lines:
                    groups.setdefault("refinement_lines", {})[layer_name] = {
                        "tags": sorted(set(valid_lines)),
                    }
                elif pec_entity:
                    # Fallback to PEC surface boundary curves.
                    surf_tags = [t for d, t in pec_entity.dimtags if d == 2]
                    if surf_tags:
                        fallback_lines = []
                        for stag in surf_tags:
                            try:
                                b = gmsh.model.getBoundary(
                                    [(2, stag)],
                                    combined=False,
                                    oriented=False,
                                    recursive=False,
                                )
                                for bdim, btag in b:
                                    if bdim == 1:
                                        fallback_lines.append(btag)
                            except Exception:
                                pass
                        if fallback_lines:
                            groups.setdefault("refinement_lines", {})[layer_name] = {
                                "tags": sorted(set(fallback_lines)),
                            }
            elif pec_entity:
                # No explicit refinement_lines — fallback to PEC surface boundary.
                surf_tags = [t for d, t in pec_entity.dimtags if d == 2]
                if surf_tags:
                    fallback_lines = []
                    for stag in surf_tags:
                        try:
                            b = gmsh.model.getBoundary(
                                [(2, stag)],
                                combined=False,
                                oriented=False,
                                recursive=False,
                            )
                            for bdim, btag in b:
                                if bdim == 1:
                                    fallback_lines.append(btag)
                        except Exception:
                            pass
                    if fallback_lines:
                        groups.setdefault("refinement_lines", {})[layer_name] = {
                            "tags": sorted(set(fallback_lines)),
                        }

            # Ultimate fallback: when the _pec entity was merged into a
            # dielectric boundary by the boolean pipeline, query the live model
            # for dim=2 surfaces whose bboxes match the pre-boolean PEC
            # surfaces and harvest their boundary curves.
            if not groups.get("refinement_lines", {}).get(layer_name, {}).get("tags"):
                _pec_bboxes = (
                    metal_tags.get("__pec_surface_bboxes__", {}).get(layer_name, [])
                    if isinstance(metal_tags.get("__pec_surface_bboxes__"), dict)
                    else []
                )
                if _pec_bboxes:
                    all_surfaces = gmsh.model.getEntities(2)
                    found_lines: set[int] = set()
                    for _, stag in all_surfaces:
                        try:
                            actual_bbox = gmsh.model.getBoundingBox(2, stag)
                        except Exception:
                            continue
                        for expected_bbox in _pec_bboxes:
                            if all(
                                abs(a - b) < 0.01
                                for a, b in zip(actual_bbox, expected_bbox, strict=True)
                            ):
                                try:
                                    b = gmsh.model.getBoundary(
                                        [(2, stag)],
                                        combined=False,
                                        oriented=False,
                                        recursive=False,
                                    )
                                    for bdim, btag in b:
                                        if bdim == 1:
                                            found_lines.add(btag)
                                except Exception:
                                    pass
                                break
                    if found_lines:
                        groups.setdefault("refinement_lines", {})[layer_name] = {
                            "tags": sorted(found_lines),
                        }
        else:
            # No surfaces_xy but may have refinement_lines
            refinement_lines = tag_info.get("refinement_lines", [])
            if refinement_lines:
                valid_lines = []
                for ltag in refinement_lines:
                    try:
                        kernel.getBoundingBox(1, ltag)
                        valid_lines.append(ltag)
                    except Exception:
                        pass
                if valid_lines:
                    groups.setdefault("refinement_lines", {})[layer_name] = {
                        "tags": sorted(set(valid_lines)),
                    }

    # --- PEC block surfaces ---
    if pec_block_tags:
        for block_name in pec_block_tags:
            for suffix in ("_xy", "_z"):
                name = f"{block_name}{suffix}"
                entity = entity_by_name.get(name)
                pg = pg_map.get(name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        groups["pec_surfaces"][name] = {
                            "phys_group": pg,
                            "tags": surf_tags,
                        }

    # --- Volumetric conductor surfaces (finite thickness) ---
    for layer_name, tag_info in metal_tags.items():
        if layer_name.startswith("__") and layer_name.endswith("__"):
            continue
        if tag_info.get("volumes"):
            for suffix in ("_xy", "_z"):
                name = f"{layer_name}{suffix}"
                entity = entity_by_name.get(name)
                pg = pg_map.get(name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        groups["conductor_surfaces"][name] = {
                            "phys_group": pg,
                            "tags": surf_tags,
                        }

    # --- Port surfaces ---
    for port_name, tags in port_tags.items():
        port_num = int(port_name[1:])
        info = next(
            (p for p in port_info if p["portnumber"] == port_num),
            None,
        )

        if info and info.get("type") == "cpw":
            element_phys_groups = []
            for i in range(len(tags)):
                elem_name = f"{port_name}_E{i}"
                entity = entity_by_name.get(elem_name)
                pg = pg_map.get(elem_name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        element_phys_groups.append(
                            {
                                "phys_group": pg,
                                "tags": surf_tags,
                                "direction": info["elements"][i].get("direction"),
                            }
                        )
            groups["port_surfaces"][port_name] = {
                "type": "cpw",
                "elements": element_phys_groups,
            }
        else:
            pg = pg_map.get(port_name)
            entity = entity_by_name.get(port_name)
            if entity and pg is not None:
                surf_tags = [t for d, t in entity.dimtags if d == 2]
                if surf_tags:
                    groups["port_surfaces"][port_name] = {
                        "phys_group": pg,
                        "tags": surf_tags,
                    }

    # --- Via boundary surfaces (via volume faces exposed to dielectric) ---
    #
    # Palace's electrostatic solver treats Conductivity-bearing material domains
    # as plain dielectrics; current does NOT flow through such regions to drag
    # terminal potential into them. To make a via behave as a true conductive
    # extension of the terminal it connects to, we need its dielectric-facing
    # boundary surfaces (bottom + sides — the top has already been merged with
    # the touching conductor shell) to be Dirichlet boundaries on that terminal.
    #
    # The boolean pipeline labels these surfaces "<material>__<via>" (parts are
    # sorted, joined with "__"). Collect them here by via layer so the config
    # generator can attach them to the right terminal.
    if _stack:
        cond_via_names = {
            n
            for n, layer in _stack.layers.items()
            if layer.layer_type in ("conductor", "via")
        }
        via_boundary: dict[str, list[int]] = {}
        for pg_name, pg_tag in pg_map.items():
            parts = gmsh_utils.split_interface_physical_name(pg_name)
            via_parts = [p for p in parts if p in via_layers]
            if len(via_parts) != 1:
                continue
            others = [p for p in parts if p != via_parts[0]]
            # Skip outer-boundary side ("___None") and via<->conductor interfaces
            if not others or "None" in others:
                continue
            if any(o in cond_via_names for o in others):
                continue
            via_boundary.setdefault(via_parts[0], []).append(pg_tag)
        if via_boundary:
            groups["via_boundary_surfaces"] = via_boundary

    # --- Internal material-interface surfaces ---
    #
    # The boolean pipeline labels unassigned surfaces by the two volume names
    # they separate, e.g. "metal___substrate" or "substrate___vacuum". Preserve
    # those names in the public groups dict so manifests, postprocessing
    # builders, and report loaders can keep Palace indices tied to CAD/mesh
    # identity instead of requiring private mesh readers.
    for pg_name, pg_tag in pg_map.items():
        if pg_name in groups["boundary_surfaces"]:
            continue
        if gmsh_utils.is_exterior_physical_name(pg_name):
            continue
        interface_parts = gmsh_utils.split_interface_physical_name(pg_name)
        if len(interface_parts) != 2:
            continue
        groups["boundary_surfaces"][pg_name] = {
            "phys_group": pg_tag,
            "tags": _surface_tags_for_physical_group(pg_tag),
            "physical_name": pg_name,
        }

    # --- Boundary surfaces (outer faces labelled *___None by the pipeline) ---
    boundary_pgs: list[int] = [
        pg for name, pg in pg_map.items() if gmsh_utils.is_exterior_physical_name(name)
    ]
    if boundary_pgs:
        groups["boundary_surfaces"]["absorbing"] = {
            "phys_group": boundary_pgs,
            "tags": [],  # tags not needed; pg_map is authoritative
        }

    kernel.synchronize()
    return groups


__all__ = ["assign_physical_groups"]
