"""Physical group assignment for Palace mesh generation.

This module builds the ``groups`` dict consumed by the config generator
from the ``pg_map`` produced by ``run_boolean_pipeline``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import gmsh

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import ActivatedRegion

logger = logging.getLogger(__name__)


def _surface_volume_owners(
    entities: Sequence[gmsh_utils.Entity],
) -> dict[int, tuple[str, ...]]:
    owners: dict[int, list[str]] = {}
    for entity in entities:
        if entity.dim != 3:
            continue
        for dimtag in entity.dimtags:
            with contextlib.suppress(Exception):
                boundary = gmsh.model.getBoundary(
                    [dimtag],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
                for bdim, btag in boundary:
                    if bdim != 2:
                        continue
                    names = owners.setdefault(btag, [])
                    if entity.name not in names:
                        names.append(entity.name)
    return {tag: tuple(names) for tag, names in owners.items()}


def _is_vacuum_like_name(name: str | None) -> bool:
    if not name:
        return False
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"air", "vacuum"}:
        return True
    tokens = [token for token in normalized.split("_") if token]
    return "air" in tokens or "vacuum" in tokens


def _is_vacuum_like_material(name: str, stack: LayerStack | None) -> bool:
    if _is_vacuum_like_name(name):
        return True
    if stack is None:
        return False
    material = stack.materials.get(name)
    eps = material.get("permittivity") if isinstance(material, Mapping) else None
    try:
        return eps is not None and abs(float(eps) - 1.0) <= 1e-9
    except (TypeError, ValueError):
        return False


def _volume_material_name(
    owner_name: str,
    volume_info: Mapping[str, object],
    stack: LayerStack | None,
) -> str:
    material = volume_info.get("material")
    if isinstance(material, str) and material:
        return material
    layer = stack.layers.get(owner_name) if stack is not None else None
    if layer is not None:
        return str(layer.material)
    return owner_name


def _surface_epr_interface_metadata(
    *,
    layer_name: str,
    neutral_name: str,
    metadata: Mapping[str, object],
    surface_tags: Sequence[int],
    stack: LayerStack | None,
    surface_owners: Mapping[int, tuple[str, ...]],
    volume_material_by_name: Mapping[str, str],
) -> tuple[str, dict[str, object]]:
    tag_owner_names = {
        int(surface_tag): tuple(surface_owners.get(int(surface_tag), ()))
        for surface_tag in surface_tags
    }
    missing_owner_tags = [
        surface_tag
        for surface_tag, owner_names in tag_owner_names.items()
        if not owner_names
    ]
    if missing_owner_tags:
        raise ValueError(
            "Surface EPR interface "
            f"{neutral_name!r} contains non-boundary surface tags "
            f"{missing_owner_tags!r}. Build Surface EPR interfaces from "
            "post-fragment full-3D volume boundaries."
        )
    owner_names = tuple(
        dict.fromkeys(
            owner
            for owners in tag_owner_names.values()
            for owner in owners
        )
    )
    face_kind = str(metadata.get("face_kind", "unsupported"))
    owner_materials = tuple(
        volume_material_by_name.get(owner_name, owner_name)
        for owner_name in owner_names
    )
    vacuum_owners = [
        owner_name
        for owner_name, material_name in zip(
            owner_names,
            owner_materials,
            strict=True,
        )
        if _is_vacuum_like_material(material_name, stack)
        or _is_vacuum_like_name(owner_name)
    ]
    dielectric_owners = [
        owner_name for owner_name in owner_names if owner_name not in vacuum_owners
    ]
    if face_kind == "bottom" and dielectric_owners:
        interface_type = "MS"
        selected_owners = tuple(dielectric_owners)
    elif vacuum_owners:
        interface_type = "MA"
        selected_owners = tuple(vacuum_owners)
    else:
        interface_type = "MS"
        selected_owners = owner_names

    layer = stack.layers.get(layer_name) if stack is not None else None
    metal_material = str(layer.material) if layer is not None else layer_name
    metal_body_id = str(metadata.get("metal_body_id") or layer_name)
    selected_materials = tuple(
        volume_material_by_name.get(owner_name, owner_name)
        for owner_name in selected_owners
    )
    label = str(metadata.get("surface_epr_band_label") or "TOTAL")
    final_name = f"{metal_body_id}__{interface_type}__{face_kind.upper()}__{label}"
    resolved = {
        **dict(metadata),
        "interface_id": final_name,
        "interface_type": interface_type,
        "adjacency_source": "gmsh_volume_boundary",
        "adjacent_volume_owner_names": owner_names,
        "surface_epr_tag_owner_names": tag_owner_names,
        "adjacent_body_ids": (metal_body_id, *selected_owners),
        "adjacent_materials": (metal_material, *selected_materials),
    }
    if label != "TOTAL":
        resolved["parent_interface_id"] = (
            f"{metal_body_id}__{interface_type}__{face_kind.upper()}__TOTAL"
        )
    return final_name, resolved


def assign_physical_groups(
    kernel,
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    entities: list[gmsh_utils.Entity],
    pg_map: dict[str, int],
    _stack: LayerStack,
    activated_regions: Sequence[ActivatedRegion] | None = None,
    pec_block_tags: dict | None = None,
    shaped_dielectric_names: set[str] | None = None,
    pec_surface_bboxes: dict | None = None,
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
        activated_regions: Explicit region declarations from the public
            simulation API. Group metadata preserves their role, margins, die
            names, and material override provenance.

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
    surface_volume_owners = _surface_volume_owners(entities)

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
        metadata: Mapping[str, object] | None = None,
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
        if metadata:
            surface_info.update(dict(metadata))
        groups["pec_surfaces"][key] = surface_info

    # Build set of via and shaped-dielectric layer names
    via_layers: set[str] = set()
    shaped_dielectric_layers = set(shaped_dielectric_names or ())
    activated_region_by_layer = {
        region.layer: region for region in activated_regions or ()
    }
    if _stack:
        via_layers = {
            n for n, layer in _stack.layers.items() if layer.layer_type == "via"
        }

    # --- Volumes (dielectrics + airbox) ---
    for material in dielectric_tags:
        entity_name = "air" if material == "airbox" else material
        group_name = "air" if material == "airbox" else material
        entity = entity_by_name.get(entity_name)
        pg = pg_map.get(entity_name)
        if entity and pg is not None:
            vol_tags = [t for d, t in entity.dimtags if d == 3]
            if vol_tags:
                volume_info = {
                    "phys_group": pg,
                    "tags": vol_tags,
                }
                layer = _stack.layers.get(group_name) if _stack else None
                if layer is not None and layer.layer_type in {
                    "dielectric",
                    "substrate",
                }:
                    activated_region = activated_region_by_layer.get(group_name)
                    material_name = (
                        activated_region.material or layer.material
                        if activated_region is not None
                        else layer.material
                    )
                    volume_info["stack_layer"] = group_name
                    volume_info["material"] = material_name
                    if activated_region is not None:
                        volume_info["activated_region"] = True
                        volume_info["activated_region_role"] = activated_region.role
                        volume_info["margin_x"] = activated_region.margin_x
                        volume_info["margin_y"] = activated_region.margin_y
                        volume_info["z_above"] = activated_region.z_above
                        volume_info["z_below"] = activated_region.z_below
                        volume_info["material_source"] = (
                            "activated_region_override"
                            if activated_region.material is not None
                            else "stack_layer"
                        )
                        if activated_region.material is not None:
                            volume_info["material_override"] = activated_region.material
                        if activated_region.die is not None:
                            volume_info["die"] = activated_region.die
                        if activated_region.lower_die is not None:
                            volume_info["lower_die"] = activated_region.lower_die
                        if activated_region.upper_die is not None:
                            volume_info["upper_die"] = activated_region.upper_die
                groups["volumes"][group_name] = volume_info

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

    volume_material_by_name = {
        name: _volume_material_name(name, info, _stack)
        for name, info in groups["volumes"].items()
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
                _pec_bboxes = (pec_surface_bboxes or {}).get(layer_name, [])
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
            for shell_name, metadata in (
                tag_info.get("terminal_shells") or {}
            ).items():
                raw_tags = metadata.get("tags", ())
                surf_tags: list[int] = []
                for tag in raw_tags:
                    with contextlib.suppress(Exception):
                        kernel.getBoundingBox(2, int(tag))
                        surf_tags.append(int(tag))
                if not surf_tags:
                    continue
                unique_tags = sorted(set(surf_tags))
                clean_metadata = dict(metadata)
                clean_metadata.pop("tags", None)
                surface_info = {
                    "tags": unique_tags,
                    "layer": layer_name,
                    "physical_name": str(shell_name),
                    **clean_metadata,
                    "logical_only": True,
                    "palace_attributes": [],
                }
                bbox = _surface_bbox(unique_tags)
                if bbox is not None:
                    surface_info["bbox"] = bbox
                groups["conductor_surfaces"][str(shell_name)] = surface_info

            aggregate_totals: dict[str, dict[str, object]] = {}
            interface_surfaces = tag_info.get("surface_epr_interfaces") or {}
            for name, metadata in interface_surfaces.items():
                neutral_name = str(name)
                entity = entity_by_name.get(neutral_name)
                pg = pg_map.get(neutral_name)
                if entity and pg is not None:
                    surf_tags = [t for d, t in entity.dimtags if d == 2]
                    if surf_tags:
                        final_name, resolved_metadata = (
                            _surface_epr_interface_metadata(
                                layer_name=layer_name,
                                neutral_name=neutral_name,
                                metadata=metadata,
                                surface_tags=surf_tags,
                                stack=_stack,
                                surface_owners=surface_volume_owners,
                                volume_material_by_name=volume_material_by_name,
                            )
                        )
                        with contextlib.suppress(Exception):
                            gmsh.model.removePhysicalGroups([(2, pg)])
                        try:
                            pg = gmsh.model.addPhysicalGroup(
                                2,
                                surf_tags,
                                tag=pg,
                                name=final_name,
                            )
                        except Exception:
                            pg = gmsh.model.addPhysicalGroup(
                                2,
                                surf_tags,
                                name=final_name,
                            )
                        surface_info = {
                            "phys_group": pg,
                            "tags": surf_tags,
                            "layer": layer_name,
                            "physical_name": final_name,
                            **resolved_metadata,
                            "palace_attribute": pg,
                            "palace_attributes": [pg],
                        }
                        bbox = _surface_bbox(surf_tags)
                        if bbox is not None:
                            surface_info["bbox"] = bbox
                        groups["conductor_surfaces"][final_name] = surface_info
                        parent_name = resolved_metadata.get("parent_interface_id")
                        if isinstance(parent_name, str) and parent_name:
                            aggregate = aggregate_totals.setdefault(
                                parent_name,
                                {
                                    "tags": [],
                                    "children": [],
                                    "tag_owner_names": {},
                                    "metadata": dict(resolved_metadata),
                                },
                            )
                            aggregate["tags"].extend(surf_tags)  # type: ignore[union-attr]
                            aggregate["children"].append(final_name)  # type: ignore[union-attr]
                            aggregate["tag_owner_names"].update(  # type: ignore[union-attr]
                                resolved_metadata.get("surface_epr_tag_owner_names", {})
                            )
            for parent_name, aggregate in aggregate_totals.items():
                total_tags = sorted(set(aggregate["tags"]))  # type: ignore[arg-type]
                if not total_tags:
                    continue
                total_metadata = dict(aggregate["metadata"])  # type: ignore[arg-type]
                total_metadata.pop("parent_interface_id", None)
                total_metadata.pop("surface_epr_band_label", None)
                total_metadata.pop("tags", None)
                total_metadata["postprocessing_only"] = True
                total_metadata["interface_id"] = parent_name
                total_metadata["physical_name"] = parent_name
                total_metadata["surface_epr_band_min_um"] = 0.0
                total_metadata["surface_epr_band_max_um"] = None
                total_metadata["surface_epr_exclude_below_um"] = 0.0
                total_metadata["surface_epr_summary_kind"] = "total"
                total_metadata["surface_epr_child_interfaces"] = tuple(
                    aggregate["children"]  # type: ignore[arg-type]
                )
                total_metadata["surface_epr_tag_owner_names"] = dict(
                    aggregate["tag_owner_names"]  # type: ignore[arg-type]
                )
                total_metadata["logical_only"] = True
                total_metadata["palace_attributes"] = []
                surface_info = {
                    "tags": total_tags,
                    "layer": layer_name,
                    "physical_name": parent_name,
                    **total_metadata,
                }
                bbox = _surface_bbox(total_tags)
                if bbox is not None:
                    surface_info["bbox"] = bbox
                groups["conductor_surfaces"][parent_name] = surface_info
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
                                "sheet_source": info["elements"][i].get("sheet_source"),
                                "sheet_layer": info["elements"][i].get("sheet_layer"),
                                "sheet_gds_layer": info["elements"][i].get(
                                    "sheet_gds_layer"
                                ),
                            }
                        )
            groups["port_surfaces"][port_name] = {
                "type": "cpw",
                "port_name": info.get("name") if info else None,
                "sheet_source": info.get("sheet_source") if info else None,
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
                        "port_name": info.get("name") if info else None,
                        "sheet_source": info.get("sheet_source") if info else None,
                        "sheet_layer": info.get("sheet_layer") if info else None,
                        "sheet_gds_layer": info.get("sheet_gds_layer")
                        if info
                        else None,
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
    assigned_surface_names = {
        *groups["conductor_surfaces"],
        *groups["pec_surfaces"],
        *groups["port_surfaces"],
    }
    for pg_name, pg_tag in pg_map.items():
        if pg_name in assigned_surface_names:
            continue
        if pg_name in groups["boundary_surfaces"]:
            continue
        if gmsh_utils.is_exterior_physical_name(pg_name):
            continue
        interface_parts = gmsh_utils.split_interface_physical_name(pg_name)
        if len(interface_parts) != 2:
            continue
        surface_info = {
            "phys_group": pg_tag,
            "tags": _surface_tags_for_physical_group(pg_tag),
            "physical_name": pg_name,
        }
        interface_materials = {
            part: volume_material_by_name[part]
            for part in interface_parts
            if part in volume_material_by_name
        }
        if interface_materials:
            surface_info["interface_materials"] = interface_materials
        groups["boundary_surfaces"][pg_name] = surface_info

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
