"""Resolve Palace material and dielectric-interface sidecars.

This module owns the material-derived table boundary inside Resolve. It keeps
config sidecar discovery, material row construction, and dielectric-interface
row construction together because they all describe one semantic artifact
family: material provenance for postprocessing tables. Loss and participation
derivations consume these tables from this module instead of reaching into
sidecar-specific helper files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from gsim.palace.mesh.postprocessing import PostprocessingIndexMap

from gsim.palace._shared import optional_int, optional_str
from gsim.palace.resolve.loaders.index_maps import (
    load_optional_postprocessing_index_map,
)
from gsim.palace.resolve.sources.run_artifacts import (
    find_config_json,
    find_material_resolution_json,
)

DOMAIN_MATERIAL_COLUMNS = (
    "material_row_index",
    "material_attribute",
    "material_attributes",
    "domain_index",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "metadata",
    "material_name",
    "permittivity",
    "loss_tangent",
    "conductivity",
    "permeability",
    "material_axes",
    "volume_name",
    "stack_material_name",
    "matched_material_name",
    "material_model_type",
    "material_model_source",
    "material_within_validity",
    "material_validity_note",
    "material_frequency_hz",
    "material_frequency_ghz",
    "raw_material_resolution",
    "raw_material",
)
DIELECTRIC_INTERFACE_COLUMNS = (
    "interface_row_index",
    "surface_index",
    "surface_attribute",
    "surface_attributes",
    "section",
    "source_name",
    "physical_name",
    "entry_name",
    "role",
    "attributes",
    "metadata",
    "interface_type",
    "preset_name",
    "preset_source",
    "loss_channel",
    "source_entry_name",
    "thickness",
    "permittivity",
    "loss_tangent",
    "interface_material_name",
    "matched_material_name",
    "material_model_type",
    "material_model_source",
    "material_within_validity",
    "material_validity_note",
    "material_frequency_hz",
    "material_frequency_ghz",
    "raw_material_resolution",
    "raw_interface",
)


def load_domain_material_summary(
    source: str | Path | dict,
    *,
    config_path: str | Path | None = None,
    index_map_path: str | Path | None = None,
    material_resolution_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Palace ``Domains.Materials`` with physical-name provenance.

    The returned table interprets the effective material rows from
    ``config.json`` and, when available, joins each material attribute back to
    the domain postprocessing entries in ``palace_index_map.json``. When
    available, ``palace_material_resolution.json`` is joined by material
    attribute so reports can explain which material overlay/model source was
    applied during config generation.
    """
    import pandas as pd

    resolved_config_path = find_optional_config_path(
        source,
        config_path=config_path,
    )
    if resolved_config_path is None or not resolved_config_path.exists():
        msg = "config.json not found"
        raise FileNotFoundError(msg)

    data = json.loads(resolved_config_path.read_text())
    materials = domain_material_entries(data)
    index_map = load_optional_postprocessing_index_map(
        source,
        index_map_path=index_map_path,
    )
    material_resolution_rows = load_optional_material_resolution_rows(
        source,
        material_resolution_path=material_resolution_path,
    )

    rows: list[dict[str, Any]] = []
    for material_row_index, material in enumerate(materials, start=1):
        attributes = material_attributes(material)
        if not attributes:
            rows.append(
                domain_material_row(
                    material_row_index=material_row_index,
                    material_attribute=None,
                    material_attributes=attributes,
                    material=material,
                    index_entry=None,
                    material_resolution=matching_material_resolution_row(
                        material_resolution_rows,
                        material_row_index=material_row_index,
                        material_attribute=None,
                    ),
                )
            )
            continue

        for attribute in attributes:
            material_resolution = matching_material_resolution_row(
                material_resolution_rows,
                material_row_index=material_row_index,
                material_attribute=attribute,
            )
            matches = (
                index_map.entries_for_attribute(
                    attribute,
                    section="Domains.Postprocessing.Energy",
                )
                if index_map is not None
                else ()
            )
            if not matches:
                rows.append(
                    domain_material_row(
                        material_row_index=material_row_index,
                        material_attribute=attribute,
                        material_attributes=attributes,
                        material=material,
                        index_entry=None,
                        material_resolution=material_resolution,
                    )
                )
                continue
            rows.extend(
                domain_material_row(
                    material_row_index=material_row_index,
                    material_attribute=attribute,
                    material_attributes=attributes,
                    material=material,
                    index_entry=index_entry,
                    material_resolution=material_resolution,
                )
                for index_entry in matches
            )

    if not rows:
        return empty_domain_material_summary()
    return pd.DataFrame.from_records(rows, columns=DOMAIN_MATERIAL_COLUMNS)


def load_dielectric_interface_summary(
    source: str | Path | dict,
    *,
    config_path: str | Path | None = None,
    index_map_path: str | Path | None = None,
    material_resolution_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Palace dielectric postprocessing interfaces with provenance.

    The returned table interprets ``Boundaries.Postprocessing.Dielectric`` rows
    from ``config.json`` and joins their Palace indices back to
    ``palace_index_map.json`` physical names when that map is available.
    """
    import pandas as pd

    resolved_config_path = find_optional_config_path(
        source,
        config_path=config_path,
    )
    if resolved_config_path is None or not resolved_config_path.exists():
        msg = "config.json not found"
        raise FileNotFoundError(msg)

    data = json.loads(resolved_config_path.read_text())
    interfaces = dielectric_interface_entries(data)
    index_map = load_optional_postprocessing_index_map(
        source,
        index_map_path=index_map_path,
    )
    material_resolution_rows = load_optional_interface_material_resolution_rows(
        source,
        material_resolution_path=material_resolution_path,
    )

    rows: list[dict[str, Any]] = []
    for interface_row_index, interface in enumerate(interfaces, start=1):
        surface_index = optional_int(config_material_value(interface, "Index"))
        attributes = material_attributes(interface)
        interface_resolution = matching_interface_material_resolution_row(
            material_resolution_rows,
            interface_row_index=interface_row_index,
            surface_index=surface_index,
        )
        matches = dielectric_interface_matches(
            index_map,
            surface_index=surface_index,
            attributes=attributes,
        )
        if not matches:
            rows.append(
                dielectric_interface_row(
                    interface_row_index=interface_row_index,
                    surface_index=surface_index,
                    surface_attribute=attributes[0] if attributes else None,
                    surface_attributes=attributes,
                    interface=interface,
                    index_entry=None,
                    material_resolution=interface_resolution,
                )
            )
            continue
        rows.extend(
            dielectric_interface_row(
                interface_row_index=interface_row_index,
                surface_index=surface_index,
                surface_attribute=matching_surface_attribute(index_entry, attributes),
                surface_attributes=attributes,
                interface=interface,
                index_entry=index_entry,
                material_resolution=interface_resolution,
            )
            for index_entry in matches
        )

    if not rows:
        return empty_dielectric_interface_summary()
    return pd.DataFrame.from_records(rows, columns=DIELECTRIC_INTERFACE_COLUMNS)


def domain_material_entries(data: Any) -> tuple[dict[str, Any], ...]:
    """Return Palace ``Domains.Materials`` entries from decoded config data."""
    if not isinstance(data, dict):
        return ()
    domains = data.get("Domains", {})
    if not isinstance(domains, dict):
        return ()
    materials = domains.get("Materials", ())
    if not isinstance(materials, (list, tuple)):
        return ()
    return tuple(entry for entry in materials if isinstance(entry, dict))


def dielectric_interface_entries(data: Any) -> tuple[dict[str, Any], ...]:
    """Return Palace dielectric postprocessing entries from decoded config data."""
    if not isinstance(data, dict):
        return ()
    boundaries = data.get("Boundaries", {})
    if not isinstance(boundaries, dict):
        return ()
    postprocessing = boundaries.get("Postprocessing", {})
    if not isinstance(postprocessing, dict):
        return ()
    interfaces = postprocessing.get("Dielectric", ())
    if not isinstance(interfaces, (list, tuple)):
        return ()
    return tuple(entry for entry in interfaces if isinstance(entry, dict))


def load_optional_material_resolution_rows(
    source: str | Path | dict,
    *,
    material_resolution_path: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    """Load domain material-resolution rows when the sidecar is available."""
    resolved_path = find_optional_material_resolution_path(
        source,
        material_resolution_path=material_resolution_path,
    )
    if resolved_path is None or not resolved_path.exists():
        return ()
    data = json.loads(resolved_path.read_text())
    rows = data.get("materials", ()) if isinstance(data, dict) else ()
    if not isinstance(rows, (list, tuple)):
        return ()
    return tuple(dict(row) for row in rows if isinstance(row, dict))


def load_optional_interface_material_resolution_rows(
    source: str | Path | dict,
    *,
    material_resolution_path: str | Path | None,
) -> tuple[dict[str, Any], ...]:
    """Load interface material-resolution rows when the sidecar is available."""
    resolved_path = find_optional_material_resolution_path(
        source,
        material_resolution_path=material_resolution_path,
    )
    if resolved_path is None or not resolved_path.exists():
        return ()
    data = json.loads(resolved_path.read_text())
    rows = data.get("interfaces", ()) if isinstance(data, dict) else ()
    if not isinstance(rows, (list, tuple)):
        return ()
    return tuple(dict(row) for row in rows if isinstance(row, dict))


def domain_material_row(
    *,
    material_row_index: int,
    material_attribute: int | None,
    material_attributes: tuple[int, ...],
    material: dict[str, Any],
    index_entry: Any | None,
    material_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return one domain-material audit row with optional index-map provenance."""
    physical_name = None
    entry_name = None
    role = None
    section = None
    domain_index = None
    attributes = (material_attribute,) if material_attribute is not None else ()
    metadata: dict[str, Any] = {}
    source_name = (
        f"Attribute {material_attribute}"
        if material_attribute is not None
        else f"Material row {material_row_index}"
    )

    if index_entry is not None:
        domain_index = index_entry.index
        section = index_entry.section
        entry_name = index_entry.entry_name
        role = index_entry.role
        attributes = index_entry.attributes
        physical_name = (
            index_entry.physical_names[0] if index_entry.physical_names else None
        )
        source_name = index_entry.primary_physical_name
        metadata = dict(index_entry.metadata)

    return {
        "material_row_index": material_row_index,
        "material_attribute": material_attribute,
        "material_attributes": material_attributes,
        "domain_index": domain_index,
        "section": section,
        "source_name": source_name,
        "physical_name": physical_name,
        "entry_name": entry_name,
        "role": role,
        "attributes": attributes,
        "metadata": metadata,
        "material_name": config_material_value(material, "Name", "Material"),
        "permittivity": optional_numeric(
            config_material_value(
                material,
                "Permittivity",
                "RelativePermittivity",
                "epsilon_r",
                "eps_r",
            )
        ),
        "loss_tangent": optional_numeric(
            config_material_value(
                material,
                "LossTan",
                "LossTangent",
                "loss_tangent",
                "tan_delta",
            )
        ),
        "conductivity": optional_numeric(
            config_material_value(
                material,
                "Conductivity",
                "conductivity",
            )
        ),
        "permeability": optional_numeric(
            config_material_value(
                material,
                "Permeability",
                "RelativePermeability",
                "mu_r",
            )
        ),
        "material_axes": config_material_value(
            material,
            "MaterialAxes",
            "Axes",
        ),
        "volume_name": material_resolution_value(material_resolution, "volume_name"),
        "stack_material_name": material_resolution_value(
            material_resolution,
            "stack_material_name",
        ),
        "matched_material_name": material_resolution_value(
            material_resolution,
            "matched_material_name",
        ),
        "material_model_type": material_resolution_value(
            material_resolution,
            "model_type",
        ),
        "material_model_source": material_resolution_value(
            material_resolution,
            "model_source",
        ),
        "material_within_validity": material_resolution_value(
            material_resolution,
            "within_validity",
        ),
        "material_validity_note": material_resolution_value(
            material_resolution,
            "validity_note",
        ),
        "material_frequency_hz": optional_numeric(
            material_resolution_value(material_resolution, "evaluation_frequency_hz")
        ),
        "material_frequency_ghz": optional_numeric(
            material_resolution_value(material_resolution, "evaluation_frequency_ghz")
        ),
        "raw_material_resolution": (
            {} if material_resolution is None else dict(material_resolution)
        ),
        "raw_material": dict(material),
    }


def dielectric_interface_row(
    *,
    interface_row_index: int,
    surface_index: int | None,
    surface_attribute: int | None,
    surface_attributes: tuple[int, ...],
    interface: dict[str, Any],
    index_entry: Any | None,
    material_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return one dielectric-interface audit row with physical-name provenance."""
    physical_name = None
    entry_name = None
    role = None
    section = (
        "Boundaries.Postprocessing.Dielectric" if surface_index is not None else None
    )
    attributes = surface_attributes
    metadata: dict[str, Any] = {}
    preset_name = None
    preset_source = None
    source_name = (
        f"Surface {surface_index}"
        if surface_index is not None
        else f"Interface row {interface_row_index}"
    )

    if index_entry is not None:
        surface_index = index_entry.index
        section = index_entry.section
        entry_name = index_entry.entry_name
        role = index_entry.role
        attributes = index_entry.attributes
        physical_name = (
            index_entry.physical_names[0] if index_entry.physical_names else None
        )
        source_name = index_entry.primary_physical_name
        metadata = dict(index_entry.metadata)
        preset_name = optional_str(index_entry.extra.get("preset_name"))
        preset_source = optional_str(index_entry.extra.get("preset_source"))
        source_name = optional_str(metadata.get("source_entry_name")) or source_name

    return {
        "interface_row_index": interface_row_index,
        "surface_index": surface_index,
        "surface_attribute": surface_attribute,
        "surface_attributes": surface_attributes,
        "section": section,
        "source_name": source_name,
        "physical_name": physical_name,
        "entry_name": entry_name,
        "role": role,
        "attributes": attributes,
        "metadata": metadata,
        "interface_type": config_material_value(interface, "Type", "interface_type"),
        "preset_name": preset_name,
        "preset_source": preset_source,
        "loss_channel": optional_str(metadata.get("loss_channel")),
        "source_entry_name": optional_str(metadata.get("source_entry_name")),
        "thickness": optional_numeric(
            config_material_value(interface, "Thickness", "thickness")
        ),
        "permittivity": optional_numeric(
            config_material_value(interface, "Permittivity", "permittivity")
        ),
        "loss_tangent": optional_numeric(
            config_material_value(
                interface,
                "LossTan",
                "LossTangent",
                "loss_tangent",
                "tan_delta",
            )
        ),
        "interface_material_name": material_resolution_value(
            material_resolution,
            "interface_material_name",
        ),
        "matched_material_name": material_resolution_value(
            material_resolution,
            "matched_material_name",
        ),
        "material_model_type": material_resolution_value(
            material_resolution,
            "model_type",
        ),
        "material_model_source": material_resolution_value(
            material_resolution,
            "model_source",
        ),
        "material_within_validity": material_resolution_value(
            material_resolution,
            "within_validity",
        ),
        "material_validity_note": material_resolution_value(
            material_resolution,
            "validity_note",
        ),
        "material_frequency_hz": optional_numeric(
            material_resolution_value(material_resolution, "evaluation_frequency_hz")
        ),
        "material_frequency_ghz": optional_numeric(
            material_resolution_value(material_resolution, "evaluation_frequency_ghz")
        ),
        "raw_material_resolution": (
            {} if material_resolution is None else dict(material_resolution)
        ),
        "raw_interface": dict(interface),
    }


def matching_material_resolution_row(
    rows: tuple[dict[str, Any], ...],
    *,
    material_row_index: int,
    material_attribute: int | None,
) -> dict[str, Any] | None:
    """Return the material-resolution row matching a config material entry."""
    if material_attribute is not None:
        for row in rows:
            if optional_int(row.get("material_attribute")) == material_attribute:
                return row

    for row in rows:
        if optional_int(row.get("material_row_index")) == material_row_index:
            return row
    return None


def matching_interface_material_resolution_row(
    rows: tuple[dict[str, Any], ...],
    *,
    interface_row_index: int,
    surface_index: int | None,
) -> dict[str, Any] | None:
    """Return the material-resolution row matching a dielectric interface entry."""
    if surface_index is not None:
        for row in rows:
            if optional_int(row.get("surface_index")) == surface_index:
                return row

    for row in rows:
        if optional_int(row.get("interface_row_index")) == interface_row_index:
            return row
    return None


def dielectric_interface_matches(
    index_map: PostprocessingIndexMap | None,
    *,
    surface_index: int | None,
    attributes: tuple[int, ...],
) -> tuple[Any, ...]:
    """Return index-map entries matching one dielectric-interface config entry."""
    if index_map is None:
        return ()

    section = "Boundaries.Postprocessing.Dielectric"
    if surface_index is not None:
        entry = index_map.entry_for_index(section, surface_index)
        if entry is not None:
            return (entry,)

    matched_entries = []
    seen: set[tuple[str, int]] = set()
    for attribute in attributes:
        for entry in index_map.entries_for_attribute(attribute, section=section):
            key = (entry.section, entry.index)
            if key in seen:
                continue
            seen.add(key)
            matched_entries.append(entry)
    return tuple(matched_entries)


def matching_surface_attribute(
    index_entry: Any, attributes: tuple[int, ...]
) -> int | None:
    """Return the surface attribute associated with one index-map entry."""
    for attribute in attributes:
        if attribute in index_entry.attributes:
            return attribute
    return attributes[0] if attributes else None


def material_resolution_value(
    material_resolution: dict[str, Any] | None,
    key: str,
) -> Any:
    """Return one material-resolution field, preserving missing values as ``None``."""
    if material_resolution is None:
        return None
    return material_resolution.get(key)


def material_attributes(material: dict[str, Any]) -> tuple[int, ...]:
    """Return Palace material/interface attributes from one config entry."""
    value = config_material_value(material, "Attributes", "attributes")
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (int(value),)
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(attribute) for attribute in value)


def config_material_value(material: dict[str, Any], *keys: str) -> Any:
    """Return a material config value using case-insensitive key aliases."""
    for key in keys:
        if key in material:
            return material[key]
    lower_keys = {str(key).lower(): value for key, value in material.items()}
    for key in keys:
        value = lower_keys.get(key.lower())
        if value is not None:
            return value
    return None


def optional_numeric(value: Any) -> Any:
    """Return a float when conversion succeeds, otherwise preserve the value."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def find_optional_config_path(
    source: str | Path | dict,
    *,
    config_path: str | Path | None,
) -> Path | None:
    """Return an explicit or discovered Palace config path when available."""
    if config_path is not None:
        return Path(config_path)
    return find_config_json(source)


def find_optional_material_resolution_path(
    source: str | Path | dict,
    *,
    material_resolution_path: str | Path | None,
) -> Path | None:
    """Return an explicit or discovered material-resolution path when available."""
    if material_resolution_path is not None:
        return Path(material_resolution_path)
    return find_material_resolution_json(source)


def empty_domain_material_summary() -> pd.DataFrame:
    """Return an empty domain-material summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(columns=DOMAIN_MATERIAL_COLUMNS)


def empty_dielectric_interface_summary() -> pd.DataFrame:
    """Return an empty dielectric-interface summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(columns=DIELECTRIC_INTERFACE_COLUMNS)


__all__ = [
    "DIELECTRIC_INTERFACE_COLUMNS",
    "DOMAIN_MATERIAL_COLUMNS",
    "empty_dielectric_interface_summary",
    "empty_domain_material_summary",
    "find_optional_config_path",
    "load_dielectric_interface_summary",
    "load_domain_material_summary",
]
