"""Palace-specific material resolution with frequency-dependent dispersion.

Evaluates dispersion models from the materials database at a target frequency
and produces a materials dict suitable for Palace config generation.

This implements the RFC's external frequency loop strategy: since Palace
does not natively support epsilon(f), gsim evaluates each material's
dispersion model at the target frequency and writes scalar properties
to the Palace config JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gsim.common.stack.materials import (
    MaterialProperties,
    get_material_properties,
)
from gsim.common.stack.overlays import load_overlay, load_overlay_data, merge_overlay

MaterialOverlayInput = str | Path | Mapping[str, Any] | None


def resolve_palace_materials_at_frequency(
    materials: dict[str, dict],
    frequency_hz: float,
    material_overlay: MaterialOverlayInput = None,
) -> dict[str, dict]:
    """Evaluate material dispersion at a given frequency for Palace config.

    For each material in the dict, looks up the corresponding entry in
    MATERIALS_DB, optionally merged with a PDK material overlay, and evaluates
    its dispersion model at the target frequency. The resolved scalar values
    replace the original values in the dict.

    Materials without dispersion models or without matching database entries
    are left unchanged.

    Args:
        materials: Dict of material name -> properties dict (from LayerStack)
        frequency_hz: Target frequency in Hz
        material_overlay: Optional PDK overlay path, raw overlay mapping, or
            already-loaded material overlay dict. Overlay entries are merged
            with the built-in material database for Palace config generation
            without mutating ``materials`` or the source ``LayerStack``.

    Returns:
        New materials dict with evaluated scalar properties
    """
    resolved: dict[str, dict] = {}
    material_db = _material_database(material_overlay)

    for name, props in materials.items():
        db_props = _lookup_material(name, material_db)
        if db_props is None:
            resolved[name] = dict(props)
            continue

        evaluated = db_props.evaluate_at_frequency(frequency_hz)

        new_props: dict[str, object] = dict(props)

        if evaluated.permittivity is not None:
            new_props["permittivity"] = evaluated.permittivity

        if evaluated.loss_tangent is not None:
            new_props["loss_tangent"] = evaluated.loss_tangent

        if evaluated.conductivity is not None:
            new_props["conductivity"] = evaluated.conductivity
        else:
            new_props.pop("conductivity", None)

        if evaluated.permeability is not None:
            new_props["permeability"] = evaluated.permeability

        resolved[name] = new_props

    return resolved


def _material_database(
    material_overlay: MaterialOverlayInput,
) -> dict[str, MaterialProperties] | None:
    if material_overlay is None:
        return None
    return merge_overlay(_load_material_overlay(material_overlay))


def _load_material_overlay(
    material_overlay: MaterialOverlayInput,
) -> dict[str, MaterialProperties]:
    if material_overlay is None:
        return {}

    if isinstance(material_overlay, str | Path):
        return load_overlay(material_overlay)

    if isinstance(material_overlay, Mapping):
        if all(
            isinstance(value, MaterialProperties) for value in material_overlay.values()
        ):
            return dict(material_overlay)

        if "materials" in material_overlay:
            return load_overlay_data(material_overlay)

        return load_overlay_data({"materials": material_overlay})

    raise TypeError(
        "material_overlay must be a path, overlay mapping, loaded material "
        f"mapping, or None; got {type(material_overlay).__name__}."
    )


def _lookup_material(
    material_name: str,
    material_db: dict[str, MaterialProperties] | None,
) -> MaterialProperties | None:
    if material_db is not None:
        if material_name in material_db:
            return material_db[material_name]

        material_name_lower = material_name.lower()
        for db_name, props in material_db.items():
            if db_name.lower() == material_name_lower:
                return props

    return get_material_properties(material_name)
