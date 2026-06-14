"""Typed Palace postprocessing builders derived from mesh roles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .manifest import MeshManifest, MeshPhysicalGroup, MeshRole

SurfaceFluxType = Literal["Electric", "Magnetic", "Power"]
DielectricInterfaceType = Literal["Default", "MA", "MS", "SA"]


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
    permittivity: float
    loss_tangent: float = 0.0
    role: MeshRole | str = "boundary_surface"
    entry_names: tuple[str, ...] = ()


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
) -> PostprocessingConfig:
    """Build Palace postprocessing config from manifest roles.

    This builder keeps public workflow notebooks out of private naming schemes:
    callers select physical roles or public manifest entry names, and the
    builder emits Palace ``Index`` values plus an audit-friendly index map.
    It configures Palace's existing postprocessing domains only; masked surface
    EPR output remains a Palace-side capability rather than a Python replay.
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
        for entry in _selected_entries(
            manifest=manifest,
            role=spec.role,
            entry_names=spec.entry_names,
        ):
            if not entry.attributes:
                continue
            boundaries["Dielectric"].append(
                {
                    "Index": dielectric_index,
                    "Attributes": list(entry.attributes),
                    "Type": spec.interface_type,
                    "Thickness": spec.thickness,
                    "Permittivity": spec.permittivity,
                    "LossTan": spec.loss_tangent,
                }
            )
            index_entries.append(
                _index_entry(
                    section="Boundaries.Postprocessing.Dielectric",
                    index=dielectric_index,
                    entry=entry,
                    extra={"Type": spec.interface_type},
                )
            )
            dielectric_index += 1

    return PostprocessingConfig(
        domains=domains,
        boundaries=boundaries,
        index_map=PostprocessingIndexMap(entries=tuple(index_entries)),
    )


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
        attributes = _as_int_tuple(terminal_entry.get("Attributes", ()))
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


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return ()
    return tuple(
        item for item in value if isinstance(item, int) and not isinstance(item, bool)
    )


__all__ = [
    "DielectricInterfaceSpec",
    "DielectricInterfaceType",
    "PostprocessingConfig",
    "PostprocessingIndexEntry",
    "PostprocessingIndexMap",
    "SurfaceFluxSpec",
    "SurfaceFluxType",
    "build_postprocessing_config_from_manifest",
    "build_terminal_index_map_from_manifest",
]
