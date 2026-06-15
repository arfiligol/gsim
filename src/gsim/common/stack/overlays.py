"""PDK overlay system for augmenting the built-in materials database.

PDK overlays are small YAML/JSON files that add foundry-specific measurements
to the built-in MATERIALS_DB. For example, ``gsim/pdk_overlays/ihp.yaml``
would contain IHP's measured eps=4.1 for SiO2 at RF, tagged with the process name.

When a PDK overlay is active, its values are merged into MATERIALS_DB at
resolution time, taking priority over the built-in values (but lower
priority than user overrides).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from gsim.common.stack.materials import (
    DispersionModel,
    MaterialProperties,
    ValidityRange,
)

_FIELD_ALIASES = {
    "permittivity": ("permittivity", "relative_permittivity", "epsilon_r", "eps_r"),
    "conductivity": ("conductivity",),
    "loss_tangent": ("loss_tangent",),
    "permeability": ("permeability", "relative_permeability", "mu_r"),
}


def _parse_tensor_or_scalar(
    entry: Mapping[str, Any], key: str
) -> float | list[float] | None:
    """Parse a key that can be scalar or list-of-3 from an overlay dict."""
    val = None
    for alias in _FIELD_ALIASES.get(key, (key,)):
        if alias in entry:
            val = entry[alias]
            break
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return float(val)


def load_overlay_data(data: Mapping[str, Any] | None) -> dict[str, MaterialProperties]:
    """Load a PDK overlay from an in-memory mapping.

    The mapping should have the format:

    ```python
    materials:
      SiO2:
        permittivity: 4.1
        loss_tangent: 0.0
        validity_frequency: [0, 10e9]
        source: "IHP SG13G2 PDK"
    material_aliases:
      silicon: Si
    ```

    Args:
        data: Mapping containing a ``materials`` section and optional
            ``material_aliases`` section mapping external names to overlay
            material records.

    Returns:
        Dict of material name -> MaterialProperties from the overlay mapping.
    """
    if not data or "materials" not in data:
        return {}

    overlay_materials: dict[str, MaterialProperties] = {}
    for name, entry in data["materials"].items():
        if not isinstance(entry, dict):
            continue

        mat_type = entry.get("type", "dielectric")
        props_kwargs: dict = {"type": mat_type}

        for key in ("permittivity", "conductivity", "loss_tangent", "permeability"):
            val = _parse_tensor_or_scalar(entry, key)
            if val is not None:
                props_kwargs[key] = val

        if "material_axes" in entry:
            props_kwargs["material_axes"] = entry["material_axes"]

        dispersion_models = []
        if "dispersion_models" in entry:
            for model_data in entry["dispersion_models"]:
                if not isinstance(model_data, dict):
                    continue
                validity = ValidityRange()
                if "validity_frequency" in model_data:
                    vf = model_data["validity_frequency"]
                    validity = ValidityRange(valid_frequency=(vf[0], vf[1]))
                elif "validity_wavelength" in model_data:
                    vw = model_data["validity_wavelength"]
                    validity = ValidityRange(valid_wavelength=(vw[0], vw[1]))

                dm = DispersionModel(
                    type=model_data.get("type", "constant"),
                    permittivity=model_data.get("permittivity"),
                    validity=validity,
                    source=model_data.get("source", ""),
                )
                dispersion_models.append(dm)

        if dispersion_models:
            props_kwargs["dispersion_models"] = dispersion_models

        overlay_materials[name] = MaterialProperties(**props_kwargs)

    return _expand_material_aliases(overlay_materials, data.get("material_aliases"))


def _expand_material_aliases(
    overlay_materials: dict[str, MaterialProperties],
    material_aliases: Any,
) -> dict[str, MaterialProperties]:
    """Return overlay materials with explicit overlay-local aliases expanded."""
    if material_aliases is None:
        return overlay_materials

    if not isinstance(material_aliases, Mapping):
        msg = "material_aliases must be a mapping of alias name to material name."
        raise TypeError(msg)

    expanded = dict(overlay_materials)
    for alias, target in material_aliases.items():
        alias_name = _material_alias_name(alias)
        target_name = _material_alias_name(target)
        if alias_name in overlay_materials:
            msg = (
                f"material_aliases entry {alias_name!r} collides with an "
                "explicit overlay material."
            )
            raise ValueError(msg)
        if target_name not in overlay_materials:
            msg = (
                f"material_aliases entry {alias_name!r} targets unknown "
                f"overlay material {target_name!r}."
            )
            raise KeyError(msg)
        expanded[alias_name] = overlay_materials[target_name]
    return expanded


def _material_alias_name(name: Any) -> str:
    if isinstance(name, bool) or not isinstance(name, str) or not name:
        msg = "material_aliases entries must be non-empty strings."
        raise ValueError(msg)
    return name


def load_overlay(path: str | Path) -> dict[str, MaterialProperties]:
    """Load a PDK overlay from a YAML/JSON file.

    The file should have the format accepted by :func:`load_overlay_data`.

    Args:
        path: Path to the overlay YAML/JSON file.

    Returns:
        Dict of material name -> MaterialProperties from the overlay.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Overlay file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return load_overlay_data(data)


def merge_overlay(
    overlay: dict[str, MaterialProperties],
    base: dict[str, MaterialProperties] | None = None,
) -> dict[str, MaterialProperties]:
    """Merge a PDK overlay into the materials database.

    Overlay values take priority over base values for matching material names.
    Materials not in the overlay are kept from the base.

    Args:
        overlay: Material properties from the PDK overlay.
        base: Base materials dict. Defaults to MATERIALS_DB.

    Returns:
        Merged materials dict (new dict, does not mutate inputs).
    """
    from gsim.common.stack.materials import MATERIALS_DB

    if base is None:
        base = dict(MATERIALS_DB)

    merged = dict(base)
    for name, props in overlay.items():
        canonical_name = _canonical_material_name(name, merged)
        if canonical_name in merged:
            existing = merged[canonical_name]
            merged[canonical_name] = _merge_material(existing, props)
            if canonical_name != name:
                merged[name] = merged[canonical_name]
        else:
            merged[name] = props

    return merged


def _canonical_material_name(
    name: str,
    materials: dict[str, MaterialProperties],
) -> str:
    from gsim.common.stack.materials import MATERIAL_ALIASES

    if name in materials:
        return name

    normalized = name.lower().strip()
    alias = MATERIAL_ALIASES.get(normalized)
    if alias and alias in materials:
        return alias

    for material_name in materials:
        if material_name.lower() == normalized:
            return material_name

    return name


def _merge_material(
    base: MaterialProperties, overlay: MaterialProperties
) -> MaterialProperties:
    """Merge overlay properties into base, keeping unset fields from base."""
    kwargs: dict = {}

    for field_name in (
        "conductivity",
        "permittivity",
        "loss_tangent",
        "permeability",
        "material_axes",
    ):
        overlay_val = getattr(overlay, field_name)
        base_val = getattr(base, field_name)
        kwargs[field_name] = overlay_val if overlay_val is not None else base_val

    if overlay.dispersion_models:
        kwargs["dispersion_models"] = overlay.dispersion_models
    else:
        kwargs["dispersion_models"] = base.dispersion_models

    return MaterialProperties(**kwargs)
