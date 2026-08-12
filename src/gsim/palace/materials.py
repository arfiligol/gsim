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
    resolved, _report = resolve_palace_materials_with_report(
        materials,
        frequency_hz,
        material_overlay=material_overlay,
    )
    return resolved


def resolve_palace_materials_with_report(
    materials: dict[str, dict],
    frequency_hz: float,
    material_overlay: MaterialOverlayInput = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Evaluate Palace materials and return an audit report.

    The first return value matches :func:`resolve_palace_materials_at_frequency`.
    The second value is a JSON-friendly sidecar document that records which
    material database/overlay entry was used, which model source was selected,
    and whether the evaluation was within the model validity range.
    """
    resolved: dict[str, dict] = {}
    rows: list[dict[str, Any]] = []
    material_db = _material_database(material_overlay)

    for name, props in materials.items():
        material_lookup = _lookup_material_with_name(name, material_db)
        if material_lookup is None:
            resolved[name] = dict(props)
            rows.append(
                _material_resolution_row(
                    stack_material_name=name,
                    matched_material_name=None,
                    input_material=props,
                    effective_material=resolved[name],
                    frequency_hz=frequency_hz,
                    evaluated=None,
                )
            )
            continue

        matched_name, db_props = material_lookup
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
        rows.append(
            _material_resolution_row(
                stack_material_name=name,
                matched_material_name=matched_name,
                input_material=props,
                effective_material=new_props,
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
    """Return the merged material database for an optional overlay."""
    if material_overlay is None:
        return None
    return merge_overlay(_load_material_overlay(material_overlay))


def _load_material_overlay(
    material_overlay: MaterialOverlayInput,
) -> dict[str, MaterialProperties]:
    """Load one accepted material-overlay representation."""
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
    """Resolve material properties by name from overlay or base data."""
    lookup = _lookup_material_with_name(material_name, material_db)
    return None if lookup is None else lookup[1]


def _lookup_material_with_name(
    material_name: str,
    material_db: dict[str, MaterialProperties] | None,
) -> tuple[str, MaterialProperties] | None:
    """Resolve material properties together with their matched database name."""
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

    material_name_lower = material_name.lower().strip()
    if material_name_lower in MATERIAL_ALIASES:
        canonical_name = MATERIAL_ALIASES[material_name_lower]
        return canonical_name, props
    for db_name, db_props in MATERIALS_DB.items():
        if db_props is props or db_name.lower() == material_name_lower:
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
    row = {
        "stack_material_name": stack_material_name,
        "matched_material_name": matched_material_name,
        "evaluation_frequency_hz": float(frequency_hz),
        "evaluation_frequency_ghz": float(frequency_hz) / 1.0e9,
        "input_material": dict(input_material),
        "effective_material": dict(effective_material),
    }
    if evaluated is None:
        row.update(
            {
                "model_type": None,
                "model_source": None,
                "within_validity": None,
                "validity_note": "material not found in gsim material database",
            }
        )
        return row

    row.update(
        {
            "model_type": evaluated.model_type or None,
            "model_source": evaluated.model_source or None,
            "within_validity": evaluated.within_validity,
            "validity_note": evaluated.validity_note or None,
            "resolved_permittivity": evaluated.permittivity,
            "resolved_loss_tangent": evaluated.loss_tangent,
            "resolved_conductivity": evaluated.conductivity,
            "resolved_permeability": evaluated.permeability,
            "resolved_material_axes": evaluated.material_axes,
        }
    )
    return row
