"""Role-aware mesh manifest helpers for Palace physical groups."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

from gsim.palace._shared import int_tuple

MeshRole = Literal[
    "dielectric_volume",
    "via_volume",
    "shaped_dielectric_volume",
    "conductor_surface",
    "pec_surface",
    "port_surface",
    "boundary_surface",
    "refinement_line",
    "via_boundary_surface",
]

_ROLE_DIMENSION: dict[str, int] = {
    "dielectric_volume": 3,
    "via_volume": 3,
    "shaped_dielectric_volume": 3,
    "conductor_surface": 2,
    "pec_surface": 2,
    "port_surface": 2,
    "boundary_surface": 2,
    "refinement_line": 1,
    "via_boundary_surface": 2,
}
_INTERFACE_DELIMITERS = ("___", "__")
_EXTERIOR_SIDE_NAMES = {"none", "boundary"}


@dataclass(frozen=True)
class MeshPhysicalGroup:
    """Role metadata for one mesh group or group-like mesh entity."""

    name: str
    role: MeshRole
    attributes: tuple[int, ...] = ()
    entity_tags: tuple[int, ...] = ()
    physical_names: tuple[str, ...] = ()
    dimension: int | None = None
    source: str = "gsim_gmsh"
    interface_of: tuple[str, str] | None = None
    exterior_of: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_physical_name(self) -> str:
        """Return the first physical name, falling back to the manifest name."""
        if self.physical_names:
            return self.physical_names[0]
        return self.name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        data: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "attributes": list(self.attributes),
            "entity_tags": list(self.entity_tags),
            "physical_names": list(self.physical_names or (self.name,)),
            "source": self.source,
        }
        if self.dimension is not None:
            data["dimension"] = self.dimension
        if self.interface_of is not None:
            data["interface_of"] = list(self.interface_of)
        if self.exterior_of is not None:
            data["exterior_of"] = self.exterior_of
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class MeshManifest:
    """Lightweight manifest of generated mesh groups and their roles."""

    entries: tuple[MeshPhysicalGroup, ...] = ()
    schema_version: int = 1

    def entries_for_role(self, role: MeshRole | str) -> tuple[MeshPhysicalGroup, ...]:
        """Return manifest entries with the requested role."""
        return tuple(entry for entry in self.entries if entry.role == role)

    def attributes_for_role(self, role: MeshRole | str) -> tuple[int, ...]:
        """Return de-duplicated Palace physical-group attributes for a role."""
        attributes: list[int] = []
        for entry in self.entries_for_role(role):
            attributes.extend(entry.attributes)
        return tuple(dict.fromkeys(attributes))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the manifest to a JSON artifact and return the path."""
        artifact_path = Path(path)
        artifact_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return artifact_path


def build_mesh_manifest(groups: Mapping[str, Any]) -> MeshManifest:
    """Build a role-aware manifest from the existing ``MeshResult.groups`` dict."""
    entries: list[MeshPhysicalGroup] = []

    # Keep this translation role-oriented. Downstream Palace config,
    # postprocessing, and EPR builders need stable physical meaning without
    # depending on private layer names or gsim's internal group layout.
    for name, info in _iter_group_items(groups.get("volumes")):
        if info.get("is_via"):
            role: MeshRole = "via_volume"
        elif info.get("is_shaped_dielectric"):
            role = "shaped_dielectric_volume"
        else:
            role = "dielectric_volume"
        _append_entry(entries, name=name, role=role, info=info)

    for name, info in _iter_group_items(groups.get("conductor_surfaces")):
        _append_entry(entries, name=name, role="conductor_surface", info=info)

    for name, info in _iter_group_items(groups.get("pec_surfaces")):
        _append_entry(entries, name=name, role="pec_surface", info=info)

    for port_name, info in _iter_group_items(groups.get("port_surfaces")):
        if info.get("type") == "cpw":
            elements = info.get("elements", ())
            if isinstance(elements, Iterable) and not isinstance(
                elements, (str, bytes, Mapping)
            ):
                for index, element in enumerate(elements):
                    if not isinstance(element, Mapping):
                        continue
                    _append_entry(
                        entries,
                        name=f"{port_name}_E{index}",
                        role="port_surface",
                        info=cast(Mapping[str, Any], element),
                        metadata={
                            "port": port_name,
                            "port_type": "cpw",
                            "element_index": index,
                        },
                    )
            continue

        _append_entry(
            entries,
            name=port_name,
            role="port_surface",
            info=info,
            metadata={
                "port": port_name,
                "port_type": str(info.get("type", "single")),
            },
        )

    for name, info in _iter_group_items(groups.get("boundary_surfaces")):
        _append_entry(entries, name=name, role="boundary_surface", info=info)

    for name, info in _iter_group_items(groups.get("refinement_lines")):
        _append_entry(entries, name=name, role="refinement_line", info=info)

    via_boundary_surfaces = groups.get("via_boundary_surfaces")
    if isinstance(via_boundary_surfaces, Mapping):
        for name, attributes in via_boundary_surfaces.items():
            entries.append(
                MeshPhysicalGroup(
                    name=str(name),
                    role="via_boundary_surface",
                    attributes=int_tuple(attributes),
                    physical_names=(str(name),),
                    dimension=_ROLE_DIMENSION["via_boundary_surface"],
                )
            )

    return MeshManifest(entries=tuple(entries))


def _append_entry(
    entries: list[MeshPhysicalGroup],
    *,
    name: str,
    role: MeshRole,
    info: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Append one role-aware physical-group manifest entry."""
    entry_metadata = _metadata(info)
    if metadata:
        entry_metadata.update(metadata)
    physical_names = _physical_names(name=name, info=info)
    if _is_structured_sgb_record(info):
        # Route A/B records carry exact ownership and adjacency in metadata.
        # Display physical names are deliberately not a second topology authority.
        interface_of, exterior_of = _structured_relation(info)
    else:
        interface_of, exterior_of = _parse_physical_relation(physical_names)
    entries.append(
        MeshPhysicalGroup(
            name=str(name),
            role=role,
            attributes=int_tuple(info.get("phys_group")),
            entity_tags=int_tuple(info.get("tags")),
            physical_names=physical_names,
            dimension=_dimension(role=role, info=info),
            source=str(info.get("source", "gsim_gmsh")),
            interface_of=interface_of,
            exterior_of=exterior_of,
            metadata=entry_metadata,
        )
    )


def _iter_group_items(value: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Iterate mapping entries that contain group metadata mappings."""
    if not isinstance(value, Mapping):
        return ()
    return (
        (str(name), info) for name, info in value.items() if isinstance(info, Mapping)
    )


def _metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    """Extract manifest metadata excluding transport fields."""
    return {
        str(key): value
        for key, value in info.items()
        if key
        not in {"phys_group", "tags", "elements", "physical_name", "physical_names"}
    }


def _physical_names(*, name: str, info: Mapping[str, Any]) -> tuple[str, ...]:
    """Normalize singular or plural physical-group names."""
    value = info.get("physical_names", info.get("physical_name", name))
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return tuple(str(item) for item in value)
    return (str(name),)


def _is_structured_sgb_record(info: Mapping[str, Any]) -> bool:
    """Return whether a group carries current SGB structured provenance."""
    return (
        info.get("sgb_record") == "final_physical_group"
        and info.get("geometry_kind") == "sgb_occ"
    )


def _structured_relation(
    info: Mapping[str, Any],
) -> tuple[tuple[str, str] | None, str | None]:
    """Derive a manifest relation only from structured SGB adjacency."""
    adjacent = info.get("adjacent_solution_volume_ids")
    if isinstance(adjacent, Iterable) and not isinstance(
        adjacent, (str, bytes, Mapping)
    ):
        values = tuple(str(value) for value in adjacent)
        if len(values) == 2 and all(values):
            return (values[0], values[1]), None
    return None, None


def _dimension(*, role: MeshRole, info: Mapping[str, Any]) -> int | None:
    """Resolve a group dimension from explicit metadata or its role."""
    dim = info.get("dim")
    if isinstance(dim, Integral) and not isinstance(dim, bool):
        return int(dim)
    return _ROLE_DIMENSION.get(str(role))


def _parse_physical_relation(
    physical_names: tuple[str, ...],
) -> tuple[tuple[str, str] | None, str | None]:
    """Parse an interface pair or exterior relation from physical names."""
    for physical_name in physical_names:
        for delimiter in _INTERFACE_DELIMITERS:
            if delimiter not in physical_name:
                continue
            left, right = physical_name.split(delimiter, maxsplit=1)
            if not left or not right:
                continue
            if _is_exterior_side(right):
                return None, left
            if _is_exterior_side(left):
                return None, right
            return (left, right), None
    return None, None


def _is_exterior_side(name: str) -> bool:
    """Return whether a physical-name side denotes an exterior medium."""
    return name.lower() in _EXTERIOR_SIDE_NAMES


__all__ = [
    "MeshManifest",
    "MeshPhysicalGroup",
    "MeshRole",
    "build_mesh_manifest",
]
