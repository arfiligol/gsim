"""Palace material resolution with optional PDK-overlay provenance."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gsim.common.stack.materials import (
    MATERIAL_ALIASES,
    MATERIALS_DB,
    MaterialProperties,
    get_material_properties,
)
from gsim.common.stack.overlays import load_overlay, load_overlay_data, merge_overlay

MaterialOverlayInput = str | Path | Mapping[str, Any] | None
MATERIAL_RESOLUTION_SCHEMA_VERSION = 1


def resolve_palace_materials_at_frequency(
    materials: dict[str, dict],
    frequency_hz: float,
    material_overlay: MaterialOverlayInput = None,
) -> dict[str, dict]:
    """Resolve stack material properties at one Palace evaluation frequency."""
    resolved, _ = resolve_palace_materials_with_report(
        materials, frequency_hz, material_overlay=material_overlay
    )
    return resolved


def resolve_palace_materials_with_report(
    materials: dict[str, dict],
    frequency_hz: float,
    material_overlay: MaterialOverlayInput = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Resolve material properties and retain the selected source provenance."""
    resolved: dict[str, dict] = {}
    rows: list[dict[str, Any]] = []
    material_db = _material_database(material_overlay)
    for name, props in materials.items():
        material_lookup = _lookup_material_with_name(name, material_db)
        if material_lookup is None:
            effective = dict(props)
            evaluated = None
            matched_name = None
        else:
            matched_name, db_props = material_lookup
            evaluated = db_props.evaluate_at_frequency(frequency_hz)
            effective = dict(props)
            for field in (
                "permittivity",
                "loss_tangent",
                "conductivity",
                "permeability",
            ):
                value = getattr(evaluated, field)
                if value is not None:
                    effective[field] = value
            if evaluated.conductivity is None:
                effective.pop("conductivity", None)
        resolved[name] = effective
        rows.append(
            _material_resolution_row(
                stack_material_name=name,
                matched_material_name=matched_name,
                input_material=props,
                effective_material=effective,
                frequency_hz=frequency_hz,
                evaluated=evaluated,
            )
        )
    return resolved, {
        "schema_version": MATERIAL_RESOLUTION_SCHEMA_VERSION,
        "evaluation_frequency_hz": float(frequency_hz),
        "materials": rows,
    }


def _material_database(
    material_overlay: MaterialOverlayInput,
) -> dict[str, MaterialProperties] | None:
    """Return the merged material database when an overlay is supplied."""
    if material_overlay is None:
        return None
    return merge_overlay(_load_material_overlay(material_overlay))


def _load_material_overlay(
    material_overlay: MaterialOverlayInput,
) -> dict[str, MaterialProperties]:
    """Normalize supported overlay inputs into material properties."""
    if isinstance(material_overlay, str | Path):
        return load_overlay(material_overlay)
    if isinstance(material_overlay, Mapping):
        if all(
            isinstance(value, MaterialProperties) for value in material_overlay.values()
        ):
            return dict(material_overlay)
        return load_overlay_data(
            material_overlay
            if "materials" in material_overlay
            else {"materials": material_overlay}
        )
    raise TypeError(
        "material_overlay must be a path, overlay mapping, loaded material "
        f"mapping, or None; got {type(material_overlay).__name__}."
    )


def _lookup_material_with_name(
    material_name: str,
    material_db: dict[str, MaterialProperties] | None,
) -> tuple[str, MaterialProperties] | None:
    """Return the resolved material name and properties when available."""
    if material_db is not None:
        if material_name in material_db:
            return material_name, material_db[material_name]
        material_name_lower = material_name.lower()
        for db_name, props in material_db.items():
            if db_name.lower() == material_name_lower:
                return db_name, props
    props = get_material_properties(material_name)
    if props is None:
        return None
    normalized = material_name.lower().strip()
    if normalized in MATERIAL_ALIASES:
        return MATERIAL_ALIASES[normalized], props
    for db_name, db_props in MATERIALS_DB.items():
        if db_props is props or db_name.lower() == normalized:
            return db_name, props
    return material_name, props


def _material_resolution_row(
    *,
    stack_material_name: str,
    matched_material_name: str | None,
    input_material: Mapping[str, Any],
    effective_material: Mapping[str, Any],
    frequency_hz: float,
    evaluated: Any | None,
) -> dict[str, Any]:
    """Build one material-resolution provenance row."""
    row: dict[str, Any] = {
        "stack_material_name": stack_material_name,
        "matched_material_name": matched_material_name,
        "evaluation_frequency_hz": float(frequency_hz),
        "evaluation_frequency_ghz": float(frequency_hz) / 1e9,
        "input_material": dict(input_material),
        "effective_material": dict(effective_material),
    }
    if evaluated is None:
        row.update(
            model_type=None,
            model_source=None,
            within_validity=None,
            validity_note="material not found in gsim material database",
        )
    else:
        row.update(
            model_type=evaluated.model_type or None,
            model_source=evaluated.model_source or None,
            within_validity=evaluated.within_validity,
            validity_note=evaluated.validity_note or None,
            resolved_permittivity=evaluated.permittivity,
            resolved_loss_tangent=evaluated.loss_tangent,
            resolved_conductivity=evaluated.conductivity,
            resolved_permeability=evaluated.permeability,
            resolved_material_axes=evaluated.material_axes,
        )
    return row
