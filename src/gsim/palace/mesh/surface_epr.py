"""Surface EPR interface catalog.

Responsibility:
Owns: manifest-facing Surface EPR interface records.
Does not own: discovering CAD adjacency, assigning physical groups, partitioned
physical groups/config rows, or reports.
Inputs: Gmsh-derived mesh groups and already-discovered planar face polygons.
Outputs: audit metadata for Palace dielectric-interface specs.
Pipeline position: after final topology/group assignment exposes sheet or shell
interfaces, before Palace config rows. Representation tags select A/B/C route
materialization families; this module records them but does not generate the
geometry behind each family.
Source of Truth: gsim Full-3D Interface Surface EPR goal context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

InterfaceType = Literal["MS", "MA", "SA"]
FaceKind = Literal["top", "bottom", "sidewall"]
GeometryKind = Literal["planar_xy", "vertical_ruled", "unsupported_3d"]


@dataclass(frozen=True)
class InterfaceSurface:
    """One physical CAD interface surface selected for Surface EPR."""

    interface_id: str
    surface_tags: tuple[int, ...]
    interface_type: InterfaceType
    adjacent_materials: tuple[str, ...] = ()
    adjacent_body_ids: tuple[str, ...] = ()
    metal_body_id: str | None = None
    source_id: str | None = None
    terminal_id: str | None = None
    face_kind: FaceKind | str | None = None
    geometry_kind: GeometryKind | str = "unsupported_3d"
    representation: str = "B"
    physical_group_name: str | None = None
    physical_group_attribute: int | None = None


@dataclass(frozen=True)
class InterfaceSurfaceCatalog:
    """Small catalog of Gmsh-derived interface surfaces."""

    surfaces: tuple[InterfaceSurface, ...] = ()

    def entry_names(
        self,
        *,
        interface_type: InterfaceType | str | None = None,
        face_kind: str | None = None,
    ) -> tuple[str, ...]:
        """Return manifest entry names for selected interface surfaces."""
        names = []
        for surface in self.surfaces:
            if interface_type is not None and surface.interface_type != interface_type:
                continue
            if face_kind is not None and surface.face_kind != face_kind:
                continue
            names.append(surface.physical_group_name or surface.interface_id)
        return tuple(names)


def build_interface_surface_catalog(
    groups: Mapping[str, Any],
) -> InterfaceSurfaceCatalog:
    """Build an InterfaceSurfaceCatalog from Gmsh-derived mesh groups."""
    surfaces: list[InterfaceSurface] = []
    for name, info in _iter_mapping(groups.get("conductor_surfaces")):
        surface = _interface_surface_from_group(name, info)
        if surface is not None:
            surfaces.append(surface)
    for name, info in _iter_mapping(groups.get("boundary_surfaces")):
        surface = _interface_surface_from_group(name, info)
        if surface is not None:
            surfaces.append(surface)
    return InterfaceSurfaceCatalog(tuple(surfaces))


def _iter_mapping(value: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Iterate mapping entries whose values are group metadata mappings."""
    if not isinstance(value, Mapping):
        return ()
    return (
        (str(name), info) for name, info in value.items() if isinstance(info, Mapping)
    )


def _interface_surface_from_group(
    name: str,
    info: Mapping[str, Any],
) -> InterfaceSurface | None:
    """Build one Surface EPR catalog record from compatible group metadata."""
    interface_type = info.get("interface_type")
    if interface_type not in {"MS", "MA", "SA"}:
        return None
    attrs = _int_tuple(info.get("phys_group"))
    interface_id = str(info.get("interface_id", name))
    return InterfaceSurface(
        interface_id=interface_id,
        surface_tags=_int_tuple(info.get("tags")),
        interface_type=interface_type,  # type: ignore[arg-type]
        adjacent_materials=tuple(
            str(item) for item in info.get("adjacent_materials", ())
        ),
        adjacent_body_ids=tuple(
            str(item) for item in info.get("adjacent_body_ids", ())
        ),
        metal_body_id=_optional_string(info.get("metal_body_id")),
        source_id=_optional_string(info.get("source_id")),
        terminal_id=_optional_string(info.get("terminal_id")),
        face_kind=_optional_string(info.get("face_kind")),
        geometry_kind=str(info.get("geometry_kind", "unsupported_3d")),
        representation=str(info.get("representation", "B")),
        physical_group_name=str(info.get("physical_name", name)),
        physical_group_attribute=attrs[0] if attrs else None,
    )


def _int_tuple(value: Any) -> tuple[int, ...]:
    """Normalize integer-valued metadata to a tuple of tags."""
    if isinstance(value, bool) or value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return tuple(int(item) for item in value if not isinstance(item, bool))
    return ()


def _optional_string(value: Any) -> str | None:
    """Return a non-empty string value when present."""
    return value if isinstance(value, str) and value else None


__all__ = [
    "InterfaceSurface",
    "InterfaceSurfaceCatalog",
    "build_interface_surface_catalog",
]
