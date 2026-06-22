"""Typed Palace postprocessing builders derived from mesh roles.

Responsibility:
Owns Palace postprocessing specs, config fragments, and index-map rows derived
from mesh manifests or interface catalogs. It does not discover geometry,
assign Gmsh physical groups, parse Palace result CSVs, or render reports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from gsim.palace._shared import int_tuple

from .manifest import MeshManifest, MeshPhysicalGroup, MeshRole

SurfaceFluxType = Literal["Electric", "Magnetic", "Power"]
DielectricInterfaceType = Literal["Default", "MA", "MS", "SA"]
DielectricInterfaceSelector = str | tuple[str, str]
DielectricMaterialKind = Literal[
    "conductor",
    "dielectric",
    "vacuum",
    "superconductor",
    "mixed",
    "conductive",
]
_DIELECTRIC_INTERFACE_TYPES = {"Default", "MA", "MS", "SA"}
_DIELECTRIC_MATERIAL_KIND_ALIASES = {
    "conductor": "conductor",
    "superconductor": "conductor",
    "mixed": "conductor",
    "conductive": "conductor",
    "dielectric": "dielectric",
    "vacuum": "vacuum",
}
_DEFAULT_INTERFACE_TYPES_BY_KIND_PAIR: dict[frozenset[str], tuple[str, ...]] = {
    frozenset(("conductor", "vacuum")): ("MA",),
    frozenset(("conductor", "dielectric")): ("MS",),
    frozenset(("dielectric", "vacuum")): ("SA",),
}


@dataclass(frozen=True)
class PostprocessingIndexEntry:
    """Bidirectional link between a Palace postprocessing index and mesh identity."""

    section: str
    index: int
    entry_name: str
    role: str
    attributes: tuple[int, ...] = ()
    entity_tags: tuple[int, ...] = ()
    physical_names: tuple[str, ...] = ()
    dimension: int | None = None
    source: str | None = None
    interface_of: tuple[str, str] | None = None
    exterior_of: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_physical_name(self) -> str:
        """Return the first physical name, falling back to the manifest name."""
        if self.physical_names:
            return self.physical_names[0]
        return self.entry_name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row for audit artifacts and reports."""
        row: dict[str, Any] = {
            "section": self.section,
            "index": self.index,
            "entry_name": self.entry_name,
            "role": self.role,
            "attributes": list(self.attributes),
            "physical_names": list(self.physical_names or (self.entry_name,)),
        }
        if self.entity_tags:
            row["entity_tags"] = list(self.entity_tags)
        if self.dimension is not None:
            row["dimension"] = self.dimension
        if self.source is not None:
            row["source"] = self.source
        if self.interface_of is not None:
            row["interface_of"] = list(self.interface_of)
        if self.exterior_of is not None:
            row["exterior_of"] = self.exterior_of
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        if self.extra:
            row.update(dict(self.extra))
        return row


@dataclass(frozen=True)
class PostprocessingIndexMap:
    """Lookup table for Palace postprocessing indices and mesh physical names."""

    entries: tuple[PostprocessingIndexEntry, ...] = ()
    schema_version: int = 1

    def to_rows(self) -> tuple[dict[str, Any], ...]:
        """Return JSON-friendly index-map rows."""
        return tuple(entry.to_dict() for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly index-map document."""
        return {
            "schema_version": self.schema_version,
            "entries": list(self.to_rows()),
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the index map to a JSON artifact and return the path."""
        artifact_path = Path(path)
        artifact_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return artifact_path

    def entry_for_index(
        self, section: str, index: int
    ) -> PostprocessingIndexEntry | None:
        """Return the entry for a Palace ``section`` and ``Index``."""
        for entry in self.entries:
            if entry.section == section and entry.index == index:
                return entry
        return None

    def physical_name_for_index(self, section: str, index: int) -> str | None:
        """Return the mesh physical name for a Palace ``section`` and ``Index``."""
        entry = self.entry_for_index(section, index)
        return None if entry is None else entry.primary_physical_name

    def entries_for_name(self, entry_name: str) -> tuple[PostprocessingIndexEntry, ...]:
        """Return all Palace indices derived from one mesh physical name."""
        return tuple(entry for entry in self.entries if entry.entry_name == entry_name)

    def entries_for_physical_name(
        self, physical_name: str
    ) -> tuple[PostprocessingIndexEntry, ...]:
        """Return all Palace indices derived from one physical group name."""
        return tuple(
            entry for entry in self.entries if physical_name in entry.physical_names
        )

    def entries_for_attribute(
        self, attribute: int, section: str | None = None
    ) -> tuple[PostprocessingIndexEntry, ...]:
        """Return entries that reference a Palace physical-group attribute."""
        return tuple(
            entry
            for entry in self.entries
            if attribute in entry.attributes
            and (section is None or entry.section == section)
        )

    def indices_for_name(
        self, entry_name: str, section: str | None = None
    ) -> tuple[int, ...]:
        """Return Palace indices for one mesh physical name."""
        return tuple(
            entry.index
            for entry in self.entries_for_name(entry_name)
            if section is None or entry.section == section
        )

    def indices_for_physical_name(
        self, physical_name: str, section: str | None = None
    ) -> tuple[int, ...]:
        """Return Palace indices for one physical group name."""
        return tuple(
            entry.index
            for entry in self.entries_for_physical_name(physical_name)
            if section is None or entry.section == section
        )


@dataclass(frozen=True)
class SurfaceFluxSpec:
    """Select manifest entries for Palace ``Boundaries.Postprocessing.SurfaceFlux``."""

    role: MeshRole | str = "conductor_surface"
    entry_names: tuple[str, ...] = ()
    flux_type: SurfaceFluxType = "Electric"
    two_sided: bool | None = True
    center: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DielectricInterfaceSpec:
    """Select one or more manifest entries for Palace interface EPR postprocessing."""

    interface_type: DielectricInterfaceType
    thickness: float
    permittivity: float | None = None
    loss_tangent: float = 0.0
    material_name: str | None = None
    role: MeshRole | str = "boundary_surface"
    entry_names: tuple[str, ...] = ()
    preset_name: str | None = None
    preset_source: str | None = None
    combine_entries: bool = False
    entry_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PostprocessingConfig:
    """Palace domain/boundary postprocessing config with a stable index map."""

    domains: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    boundaries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    index_map: PostprocessingIndexMap = field(default_factory=PostprocessingIndexMap)

    def to_config(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Return the config fragments accepted by ``generate_palace_config``."""
        return {
            "domains": self.domains,
            "boundaries": self.boundaries,
        }


def build_postprocessing_config_from_manifest(
    manifest: MeshManifest,
    *,
    energy_roles: tuple[MeshRole | str, ...] = (
        "dielectric_volume",
        "shaped_dielectric_volume",
    ),
    surface_flux: tuple[SurfaceFluxSpec, ...] = (),
    dielectric_interfaces: tuple[DielectricInterfaceSpec, ...] = (),
    include_empty_sections: bool = True,
) -> PostprocessingConfig:
    """Build Palace postprocessing config from manifest roles.

    This builder keeps public workflow notebooks out of private naming schemes:
    callers select physical roles or public manifest entry names, and the
    builder emits Palace ``Index`` values plus an audit-friendly index map.
    It emits Palace's existing domain and boundary postprocessing fragments
    only; masked surface EPR output remains a Palace-side capability rather
    than a Python replay.

    Args:
        manifest: Mesh manifest with role-tagged physical groups.
        energy_roles: Domain roles that should receive Palace energy
            postprocessing indices.
        surface_flux: Boundary surface-flux requests.
        dielectric_interfaces: Boundary dielectric-interface requests.
        include_empty_sections: Keep empty Palace postprocessing sections in
            the returned config. Disable this when the config will be merged
            with problem-generated boundary postprocessing and empty sections
            should not override existing solver-owned entries.
    """
    domains: dict[str, list[dict[str, Any]]] = {"Energy": [], "Probe": []}
    boundaries: dict[str, list[dict[str, Any]]] = {
        "SurfaceFlux": [],
        "Dielectric": [],
    }
    index_entries: list[PostprocessingIndexEntry] = []

    domain_index = 1
    for role in energy_roles:
        for entry in manifest.entries_for_role(role):
            if not entry.attributes:
                continue
            domains["Energy"].append(
                {"Index": domain_index, "Attributes": list(entry.attributes)}
            )
            index_entries.append(
                _index_entry(
                    section="Domains.Postprocessing.Energy",
                    index=domain_index,
                    entry=entry,
                )
            )
            domain_index += 1

    surface_index = 1
    for spec in surface_flux:
        for entry in _selected_entries(
            manifest=manifest,
            role=spec.role,
            entry_names=spec.entry_names,
        ):
            if not entry.attributes:
                continue
            surface_entry: dict[str, Any] = {
                "Index": surface_index,
                "Attributes": list(entry.attributes),
                "Type": spec.flux_type,
            }
            if spec.two_sided is not None:
                surface_entry["TwoSided"] = spec.two_sided
            if spec.center is not None:
                surface_entry["Center"] = list(spec.center)
            boundaries["SurfaceFlux"].append(surface_entry)
            index_entries.append(
                _index_entry(
                    section="Boundaries.Postprocessing.SurfaceFlux",
                    index=surface_index,
                    entry=entry,
                    extra={"Type": spec.flux_type},
                )
            )
            surface_index += 1

    dielectric_index = 1
    for spec in dielectric_interfaces:
        if spec.permittivity is None and spec.material_name is None:
            msg = (
                "DielectricInterfaceSpec requires either permittivity or material_name."
            )
            raise ValueError(msg)
        selected_entries = _selected_entries(
            manifest=manifest,
            role=spec.role,
            entry_names=spec.entry_names,
        )
        if spec.combine_entries:
            if not selected_entries:
                continue
            attributes = _combined_attributes(selected_entries)
            if not attributes:
                continue
            boundaries["Dielectric"].append(
                _dielectric_entry(
                    index=dielectric_index,
                    spec=spec,
                    attributes=attributes,
                )
            )
            index_entries.append(
                _combined_index_entry(
                    section="Boundaries.Postprocessing.Dielectric",
                    index=dielectric_index,
                    spec=spec,
                    entries=selected_entries,
                    extra=_dielectric_interface_index_extra(spec),
                )
            )
            dielectric_index += 1
            continue

        for entry in selected_entries:
            if not entry.attributes:
                continue
            boundaries["Dielectric"].append(
                _dielectric_entry(
                    index=dielectric_index,
                    spec=spec,
                    attributes=entry.attributes,
                )
            )
            index_entries.append(
                _index_entry(
                    section="Boundaries.Postprocessing.Dielectric",
                    index=dielectric_index,
                    entry=entry,
                    extra=_dielectric_interface_index_extra(spec),
                )
            )
            dielectric_index += 1

    if not include_empty_sections:
        domains = _drop_empty_postprocessing_sections(domains)
        boundaries = _drop_empty_postprocessing_sections(boundaries)

    return PostprocessingConfig(
        domains=domains,
        boundaries=boundaries,
        index_map=PostprocessingIndexMap(entries=tuple(index_entries)),
    )


def _drop_empty_postprocessing_sections(
    sections: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return only non-empty Palace postprocessing sections."""
    return {name: entries for name, entries in sections.items() if entries}


def build_surface_epr_dielectric_specs(
    surfaces: Iterable[Any],
    *,
    preset_name: str,
    preset: Mapping[str, Any],
    face_kind: str | None = None,
    role: MeshRole | str = "conductor_surface",
) -> tuple[DielectricInterfaceSpec, ...]:
    """Build total and split Surface EPR dielectric specs from catalog surfaces."""
    base_spec = _dielectric_interface_spec_from_preset(
        preset_name=preset_name,
        preset=preset,
        role=role,
        entry_name="",
    )
    selected_surfaces = tuple(
        surface
        for surface in surfaces
        if getattr(surface, "interface_type", None) == base_spec.interface_type
        and (face_kind is None or getattr(surface, "face_kind", None) == face_kind)
    )
    split_surfaces = tuple(
        surface for surface in selected_surfaces if not _surface_epr_is_total(surface)
    )
    if not split_surfaces:
        suffix = "" if face_kind is None else f" {face_kind}"
        raise ValueError(
            f"No generated {base_spec.interface_type}{suffix} Surface EPR groups found."
        )

    source_ids = tuple(
        dict.fromkeys(
            source_id
            for surface in split_surfaces
            if (source_id := _surface_epr_source_id(surface)) is not None
        )
    )
    total_specs = tuple(
        replace(
            base_spec,
            entry_names=tuple(
                _surface_epr_entry_name(surface)
                for surface in split_surfaces
                if _surface_epr_source_id(surface) == source_id
            ),
            combine_entries=True,
            entry_name=next(
                (
                    _surface_epr_entry_name(surface)
                    for surface in selected_surfaces
                    if _surface_epr_source_id(surface) == source_id
                    and _surface_epr_is_total(surface)
                ),
                _surface_epr_total_name(
                    source_id=source_id,
                    interface_type=base_spec.interface_type,
                    face_kind=face_kind,
                ),
            ),
            metadata={
                "loss_channel": base_spec.interface_type,
                "surface_epr_summary_kind": "total",
                "surface_epr_exclude_below_um": 0.0,
            },
        )
        for source_id in source_ids
    )
    split_specs = tuple(
        replace(
            base_spec,
            entry_names=(_surface_epr_entry_name(surface),),
            combine_entries=True,
            entry_name=_surface_epr_entry_name(surface),
            metadata={
                "loss_channel": base_spec.interface_type,
                "surface_epr_summary_kind": "core"
                if getattr(surface, "band_max_um", None) is None
                else "band",
                "surface_epr_exclude_below_um": float(
                    getattr(surface, "band_min_um", 0.0)
                ),
            },
        )
        for surface in split_surfaces
    )
    return total_specs + split_specs


def build_dielectric_interface_specs_from_assignments(
    manifest: MeshManifest,
    *,
    presets: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[DielectricInterfaceSelector, str | Iterable[str]],
    role: MeshRole | str = "boundary_surface",
    require_interface: bool = True,
) -> tuple[DielectricInterfaceSpec, ...]:
    """Build interface specs from caller-supplied preset assignments.

    This helper is intentionally assignment-driven: public PDKs or private
    overlays provide preset records and choose exact mesh entries, physical
    names, or parsed interface pairs. ``gsim`` only resolves those selectors
    against the manifest and emits Palace postprocessing specs.
    """
    specs: list[DielectricInterfaceSpec] = []
    for selector, preset_names in assignments.items():
        entry = _entry_for_dielectric_interface_assignment(
            manifest=manifest,
            selector=selector,
            role=role,
            require_interface=require_interface,
        )
        specs.extend(
            _dielectric_interface_spec_from_preset(
                preset_name=preset_name,
                preset=_preset_record(presets=presets, preset_name=preset_name),
                role=role,
                entry_name=entry.name,
            )
            for preset_name in _preset_names(preset_names)
        )
    return tuple(specs)


def build_dielectric_interface_specs_from_material_kinds(
    manifest: MeshManifest,
    *,
    material_kind_by_name: Mapping[str, DielectricMaterialKind | str],
    material_name_aliases: Mapping[str, str] | None = None,
    presets: Mapping[str, Mapping[str, Any]],
    preset_by_interface_type: Mapping[str, str | Iterable[str]],
    role: MeshRole | str = "boundary_surface",
    interface_types_by_kind_pair: Mapping[tuple[str, str], str | Iterable[str]]
    | None = None,
) -> tuple[DielectricInterfaceSpec, ...]:
    """Build interface specs by classifying manifest material-kind pairs.

    Callers own material naming, material-kind assignment, and preset records.
    ``gsim`` only classifies parsed manifest interfaces into interface types,
    using material provenance metadata when semantic mesh names such as
    activated regions are present. It skips non-loss kind pairs and emits
    ordered Palace postprocessing specs.
    """
    kind_map = {
        str(name): _normalized_material_kind(kind=kind, material_name=str(name))
        for name, kind in material_kind_by_name.items()
    }
    alias_map = (
        {
            str(name): str(target_name)
            for name, target_name in material_name_aliases.items()
        }
        if material_name_aliases is not None
        else {}
    )
    interface_type_map = _interface_type_map(interface_types_by_kind_pair)
    specs: list[DielectricInterfaceSpec] = []

    for entry in manifest.entries_for_role(role):
        if entry.interface_of is None:
            continue
        left, right = entry.interface_of
        left_material = _interface_part_material_name(entry=entry, part=left)
        right_material = _interface_part_material_name(entry=entry, part=right)
        left_kind = _kind_for_interface_part(
            kind_map=kind_map,
            material_name=left_material,
            material_name_aliases=alias_map,
        )
        right_kind = _kind_for_interface_part(
            kind_map=kind_map,
            material_name=right_material,
            material_name_aliases=alias_map,
        )
        interface_types = interface_type_map.get(frozenset((left_kind, right_kind)), ())
        for interface_type in interface_types:
            preset_names = _preset_names_for_interface_type(
                preset_by_interface_type=preset_by_interface_type,
                interface_type=interface_type,
            )
            for preset_name in preset_names:
                spec = _dielectric_interface_spec_from_preset(
                    preset_name=preset_name,
                    preset=_preset_record(presets=presets, preset_name=preset_name),
                    role=role,
                    entry_name=entry.name,
                )
                if spec.interface_type != interface_type:
                    msg = (
                        f"Dielectric interface preset {preset_name!r} has "
                        f"interface_type {spec.interface_type!r}, but "
                        f"classified interface {entry.name!r} requires "
                        f"{interface_type!r}."
                    )
                    raise ValueError(msg)
                specs.append(spec)

    return tuple(specs)


def build_terminal_index_map_from_manifest(
    manifest: MeshManifest,
    terminal_entries: Iterable[Mapping[str, Any]],
    *,
    terminal_names: tuple[str, ...] = (),
) -> PostprocessingIndexMap:
    """Build an index map for Palace ``Boundaries.Terminal`` entries.

    Electrostatic capacitance matrices refer to Palace terminal indices. This
    builder links those indices back to the manifest physical groups selected
    by generated ``config.json`` terminal attributes.
    """
    manifest_entries = {
        attribute: entry
        for entry in manifest.entries
        if entry.role
        in {
            "conductor_surface",
            "pec_surface",
            "via_boundary_surface",
        }
        for attribute in entry.attributes
    }
    index_entries: list[PostprocessingIndexEntry] = []

    for terminal_entry in terminal_entries:
        index = terminal_entry.get("Index")
        if not isinstance(index, int):
            continue
        terminal_name = (
            terminal_names[index - 1]
            if 0 <= index - 1 < len(terminal_names)
            else f"T{index}"
        )
        attributes = int_tuple(terminal_entry.get("Attributes", ()))
        for attribute in attributes:
            entry = manifest_entries.get(attribute)
            if entry is None:
                continue
            index_entries.append(
                _index_entry(
                    section="Boundaries.Terminal",
                    index=index,
                    entry=entry,
                    extra={
                        "terminal_name": terminal_name,
                        "terminal_attributes": list(attributes),
                    },
                )
            )

    return PostprocessingIndexMap(entries=tuple(index_entries))


def build_surface_current_index_map_from_manifest(
    manifest: MeshManifest,
    surface_current_entries: Iterable[Mapping[str, Any]],
    *,
    current_source_names: tuple[str, ...] = (),
) -> PostprocessingIndexMap:
    """Build an index map for magnetostatic surface-current boundaries."""
    manifest_entries = {
        attribute: entry
        for entry in manifest.entries
        if entry.role
        in {
            "conductor_surface",
            "pec_surface",
            "via_boundary_surface",
        }
        for attribute in entry.attributes
    }
    index_entries: list[PostprocessingIndexEntry] = []

    for current_entry in surface_current_entries:
        index = current_entry.get("Index")
        if not isinstance(index, int):
            continue
        source_name = (
            current_source_names[index - 1]
            if 0 <= index - 1 < len(current_source_names)
            else f"I{index}"
        )
        attribute_extras = _surface_current_attribute_extras(current_entry)
        attributes = tuple(attribute for attribute, _ in attribute_extras)
        direction = current_entry.get("Direction")
        coordinate_system = current_entry.get("CoordinateSystem")
        elements = current_entry.get("Elements")
        extra: dict[str, Any] = {
            "current_source_name": source_name,
            "current_source_attributes": list(attributes),
        }
        if isinstance(direction, (str, list, tuple)):
            extra["Direction"] = direction
        if isinstance(coordinate_system, str):
            extra["CoordinateSystem"] = coordinate_system
        if isinstance(elements, list):
            extra["current_source_element_count"] = len(elements)
        for attribute, attribute_extra in attribute_extras:
            entry = manifest_entries.get(attribute)
            if entry is None:
                continue
            row_extra = {**extra, **attribute_extra}
            index_entries.append(
                _index_entry(
                    section="Boundaries.SurfaceCurrent",
                    index=index,
                    entry=entry,
                    extra=row_extra,
                )
            )
            index_entries.append(
                _index_entry(
                    section="Boundaries.Postprocessing.SurfaceFlux",
                    index=index,
                    entry=entry,
                    extra={
                        **row_extra,
                        "Type": "Magnetic",
                    },
                )
            )

    return PostprocessingIndexMap(entries=tuple(index_entries))


def _surface_current_attribute_extras(
    current_entry: dict[str, Any],
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Return SurfaceCurrent attributes with row-level element metadata."""
    rows: list[tuple[int, dict[str, Any]]] = [
        (attribute, {}) for attribute in int_tuple(current_entry.get("Attributes", ()))
    ]
    elements = current_entry.get("Elements")
    if isinstance(elements, list):
        for element_index, element in enumerate(elements, start=1):
            if isinstance(element, dict):
                element_extra: dict[str, Any] = {
                    "current_source_element_index": element_index,
                }
                direction = element.get("Direction")
                if isinstance(direction, (str, list, tuple)):
                    element_extra["Direction"] = direction
                coordinate_system = element.get("CoordinateSystem")
                if isinstance(coordinate_system, str):
                    element_extra["CoordinateSystem"] = coordinate_system
                rows.extend(
                    (attribute, element_extra)
                    for attribute in int_tuple(element.get("Attributes", ()))
                )

    # Preserve one row per physical attribute while carrying the element-local
    # metadata for multielement source review tables.
    deduped: dict[int, dict[str, Any]] = {}
    for attribute, extra in rows:
        deduped.setdefault(attribute, extra)
    return tuple((attribute, deduped[attribute]) for attribute in sorted(deduped))


def _selected_entries(
    *,
    manifest: MeshManifest,
    role: MeshRole | str,
    entry_names: tuple[str, ...],
):
    entries = manifest.entries_for_role(role)
    if not entry_names:
        return entries
    selected = set(entry_names)
    return tuple(entry for entry in entries if entry.name in selected)


def _entry_for_dielectric_interface_assignment(
    *,
    manifest: MeshManifest,
    selector: DielectricInterfaceSelector,
    role: MeshRole | str,
    require_interface: bool,
) -> MeshPhysicalGroup:
    matches = tuple(
        entry
        for entry in manifest.entries_for_role(role)
        if _matches_dielectric_interface_selector(entry=entry, selector=selector)
    )
    if not matches:
        msg = (
            "No mesh manifest entry matches dielectric interface selector "
            f"{selector!r}."
        )
        raise KeyError(msg)
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        msg = f"Dielectric interface selector {selector!r} is ambiguous: {names}."
        raise ValueError(msg)

    entry = matches[0]
    if require_interface and entry.interface_of is None:
        msg = (
            f"Dielectric interface selector {selector!r} matched {entry.name!r}, "
            "but that manifest entry is not a parsed material interface."
        )
        raise ValueError(msg)
    return entry


def _matches_dielectric_interface_selector(
    *,
    entry: MeshPhysicalGroup,
    selector: DielectricInterfaceSelector,
) -> bool:
    if isinstance(selector, str):
        return selector == entry.name or selector in entry.physical_names
    if len(selector) != 2 or entry.interface_of is None:
        return False
    return frozenset(str(part) for part in selector) == frozenset(entry.interface_of)


def _preset_names(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    names = tuple(str(item) for item in value)
    if not names:
        msg = "Dielectric interface assignment must reference at least one preset."
        raise ValueError(msg)
    return names


def _preset_record(
    *,
    presets: Mapping[str, Mapping[str, Any]],
    preset_name: str,
) -> Mapping[str, Any]:
    try:
        return presets[preset_name]
    except KeyError as error:
        msg = f"Unknown dielectric interface preset {preset_name!r}."
        raise KeyError(msg) from error


def _normalized_material_kind(
    *,
    kind: DielectricMaterialKind | str,
    material_name: str,
) -> str:
    normalized = str(kind).lower()
    normalized_kind = _DIELECTRIC_MATERIAL_KIND_ALIASES.get(normalized)
    if normalized_kind is None:
        msg = (
            f"Material {material_name!r} has unsupported dielectric material "
            f"kind {kind!r}."
        )
        raise ValueError(msg)
    return normalized_kind


def _kind_for_interface_part(
    *,
    kind_map: Mapping[str, str],
    material_name: str,
    material_name_aliases: Mapping[str, str] | None = None,
) -> str:
    if material_name in kind_map:
        return kind_map[material_name]

    alias_target = (
        None
        if material_name_aliases is None
        else material_name_aliases.get(material_name)
    )
    if alias_target is not None and alias_target in kind_map:
        return kind_map[alias_target]

    if alias_target is not None:
        msg = (
            "Missing dielectric material kind for interface material "
            f"{material_name!r} aliased to {alias_target!r}."
        )
        raise KeyError(msg)

    msg = f"Missing dielectric material kind for interface material {material_name!r}."
    raise KeyError(msg)


def _interface_part_material_name(*, entry: MeshPhysicalGroup, part: str) -> str:
    interface_materials = entry.metadata.get("interface_materials")
    if isinstance(interface_materials, Mapping):
        material_name = interface_materials.get(part)
        if isinstance(material_name, str) and material_name:
            return material_name
    return part


def _interface_type_map(
    interface_types_by_kind_pair: Mapping[tuple[str, str], str | Iterable[str]] | None,
) -> dict[frozenset[str], tuple[str, ...]]:
    if interface_types_by_kind_pair is None:
        return dict(_DEFAULT_INTERFACE_TYPES_BY_KIND_PAIR)
    return {
        frozenset(
            (
                _normalized_material_kind(kind=left, material_name="kind-pair"),
                _normalized_material_kind(kind=right, material_name="kind-pair"),
            )
        ): _interface_types(interface_types)
        for (left, right), interface_types in interface_types_by_kind_pair.items()
    }


def _interface_types(value: str | Iterable[str]) -> tuple[str, ...]:
    interface_types = _preset_names(value)
    unsupported = [
        interface_type
        for interface_type in interface_types
        if interface_type not in _DIELECTRIC_INTERFACE_TYPES
    ]
    if unsupported:
        msg = f"Unsupported dielectric interface types: {unsupported}."
        raise ValueError(msg)
    return interface_types


def _preset_names_for_interface_type(
    *,
    preset_by_interface_type: Mapping[str, str | Iterable[str]],
    interface_type: str,
) -> tuple[str, ...]:
    try:
        return _preset_names(preset_by_interface_type[interface_type])
    except KeyError as error:
        msg = f"Missing preset assignment for interface type {interface_type!r}."
        raise KeyError(msg) from error


def _dielectric_interface_spec_from_preset(
    *,
    preset_name: str,
    preset: Mapping[str, Any],
    role: MeshRole | str,
    entry_name: str,
) -> DielectricInterfaceSpec:
    try:
        interface_type = preset["interface_type"]
        thickness = preset["thickness"]
    except KeyError as error:
        msg = (
            f"Dielectric interface preset {preset_name!r} is missing {error.args[0]!r}."
        )
        raise KeyError(msg) from error

    if interface_type not in _DIELECTRIC_INTERFACE_TYPES:
        msg = (
            f"Dielectric interface preset {preset_name!r} has unsupported "
            f"interface_type {interface_type!r}."
        )
        raise ValueError(msg)

    material_name = preset.get("material_name")
    permittivity = preset.get("permittivity")
    has_material_name = isinstance(material_name, str) and bool(material_name)
    has_permittivity = permittivity is not None
    if has_material_name == has_permittivity:
        msg = (
            f"Dielectric interface preset {preset_name!r} must set exactly one of "
            "material_name or permittivity."
        )
        raise ValueError(msg)

    kwargs: dict[str, Any] = {
        "interface_type": interface_type,
        "thickness": _positive_float(thickness, preset_name, "thickness"),
        "loss_tangent": _nonnegative_float(
            preset.get("loss_tangent", 0.0),
            preset_name,
            "loss_tangent",
        ),
        "role": role,
        "entry_names": (entry_name,),
        "preset_name": preset_name,
        "preset_source": _optional_preset_source(preset.get("source")),
    }
    if has_material_name:
        kwargs["material_name"] = material_name
    else:
        kwargs["permittivity"] = _positive_float(
            permittivity,
            preset_name,
            "permittivity",
        )
    return DielectricInterfaceSpec(**kwargs)


def _dielectric_interface_index_extra(
    spec: DielectricInterfaceSpec,
) -> dict[str, Any]:
    extra: dict[str, Any] = {"Type": spec.interface_type}
    if spec.preset_name is not None:
        extra["preset_name"] = spec.preset_name
    if spec.preset_source is not None:
        extra["preset_source"] = spec.preset_source
    return extra


def _dielectric_entry(
    *,
    index: int,
    spec: DielectricInterfaceSpec,
    attributes: tuple[int, ...],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "Index": index,
        "Attributes": list(attributes),
        "Type": spec.interface_type,
        "Thickness": spec.thickness,
        "LossTan": spec.loss_tangent,
    }
    if spec.permittivity is not None:
        entry["Permittivity"] = spec.permittivity
    if spec.material_name is not None:
        entry["_MaterialName"] = spec.material_name
    return entry


def _combined_attributes(entries: tuple[MeshPhysicalGroup, ...]) -> tuple[int, ...]:
    attributes: list[int] = []
    for entry in entries:
        attributes.extend(entry.attributes)
    return tuple(dict.fromkeys(attributes))


def _surface_epr_entry_name(surface: Any) -> str:
    return str(getattr(surface, "physical_group_name", None) or surface.interface_id)


def _surface_epr_is_total(surface: Any) -> bool:
    return (
        float(getattr(surface, "band_min_um", 0.0)) == 0.0
        and getattr(surface, "band_max_um", None) is None
    )


def _surface_epr_source_id(surface: Any) -> str | None:
    source_id = getattr(surface, "source_id", None) or getattr(
        surface,
        "metal_body_id",
        None,
    )
    return source_id if isinstance(source_id, str) and source_id else None


def _surface_epr_total_name(
    *,
    source_id: str,
    interface_type: str,
    face_kind: str | None,
) -> str:
    if face_kind is None:
        return f"{source_id}__{interface_type}__TOTAL"
    return f"{source_id}__{interface_type}__{face_kind.upper()}__TOTAL"


def _combined_index_entry(
    *,
    section: str,
    index: int,
    spec: DielectricInterfaceSpec,
    entries: tuple[MeshPhysicalGroup, ...],
    extra: Mapping[str, Any],
) -> PostprocessingIndexEntry:
    metadata: dict[str, Any] = {"entry_names": [entry.name for entry in entries]}
    entry_metadata = {
        entry.name: dict(entry.metadata) for entry in entries if entry.metadata
    }
    if entry_metadata:
        metadata["entries"] = entry_metadata
    metadata.update(dict(spec.metadata))
    return PostprocessingIndexEntry(
        section=section,
        index=index,
        entry_name=spec.entry_name or "+".join(entry.name for entry in entries),
        role=str(spec.role),
        attributes=_combined_attributes(entries),
        entity_tags=tuple(
            dict.fromkeys(tag for entry in entries for tag in entry.entity_tags)
        ),
        physical_names=tuple(
            dict.fromkeys(name for entry in entries for name in entry.physical_names)
        ),
        dimension=entries[0].dimension,
        source="gsim_postprocessing",
        metadata=metadata,
        extra=dict(extra),
    )


def _optional_preset_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _positive_float(value: Any, preset_name: str, field_name: str) -> float:
    if not _is_finite_number(value) or float(value) <= 0.0:
        msg = (
            f"Dielectric interface preset {preset_name!r} field {field_name!r} "
            "must be > 0."
        )
        raise ValueError(msg)
    return float(value)


def _nonnegative_float(value: Any, preset_name: str, field_name: str) -> float:
    if not _is_finite_number(value) or float(value) < 0.0:
        msg = (
            f"Dielectric interface preset {preset_name!r} field {field_name!r} "
            "must be >= 0."
        )
        raise ValueError(msg)
    return float(value)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _index_entry(
    *,
    section: str,
    index: int,
    entry: MeshPhysicalGroup,
    extra: Mapping[str, Any] | None = None,
) -> PostprocessingIndexEntry:
    return PostprocessingIndexEntry(
        section=section,
        index=index,
        entry_name=entry.name,
        role=str(entry.role),
        attributes=entry.attributes,
        entity_tags=entry.entity_tags,
        physical_names=entry.physical_names,
        dimension=entry.dimension,
        source=entry.source,
        interface_of=entry.interface_of,
        exterior_of=entry.exterior_of,
        metadata=entry.metadata,
        extra={} if extra is None else dict(extra),
    )


__all__ = [
    "DielectricInterfaceSelector",
    "DielectricInterfaceSpec",
    "DielectricInterfaceType",
    "DielectricMaterialKind",
    "PostprocessingConfig",
    "PostprocessingIndexEntry",
    "PostprocessingIndexMap",
    "SurfaceFluxSpec",
    "SurfaceFluxType",
    "build_dielectric_interface_specs_from_assignments",
    "build_dielectric_interface_specs_from_material_kinds",
    "build_postprocessing_config_from_manifest",
    "build_surface_current_index_map_from_manifest",
    "build_surface_epr_dielectric_specs",
    "build_terminal_index_map_from_manifest",
]
