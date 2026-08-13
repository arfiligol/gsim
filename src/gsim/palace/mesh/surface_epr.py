"""Structured Route A/B Surface-EPR catalog derived from mesh groups."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

InterfaceType = Literal["MS", "MA", "SA"]


@dataclass(frozen=True)
class InterfaceSurface:
    """One structured solver-exposed Surface-EPR interface."""

    interface_id: str
    surface_tags: tuple[int, ...]
    interface_type: InterfaceType
    source_id: str | None = None
    face_kind: str | None = None
    representation: Literal["A", "B"] = "B"
    physical_group_name: str | None = None


@dataclass(frozen=True)
class InterfaceSurfaceCatalog:
    """Small typed interface catalog; names are display-only."""

    surfaces: tuple[InterfaceSurface, ...] = ()


def build_interface_surface_catalog(
    groups: Mapping[str, Any],
) -> InterfaceSurfaceCatalog:
    """Read explicit structured A/B interface metadata from final groups."""
    surfaces: list[InterfaceSurface] = []
    for collection in ("conductor_surfaces", "boundary_surfaces"):
        value = groups.get(collection, {})
        if not isinstance(value, Mapping):
            continue
        for name, info in value.items():
            if not isinstance(info, Mapping):
                continue
            interface_type = info.get("interface_type")
            route = str(info.get("route", info.get("representation", "B"))).upper()
            if interface_type not in {"MS", "MA", "SA"} or route not in {"A", "B"}:
                continue
            tags = info.get("tags", ())
            if isinstance(tags, int) and not isinstance(tags, bool):
                tags = (tags,)
            surfaces.append(
                InterfaceSurface(
                    interface_id=str(info.get("surface_id", name)),
                    surface_tags=tuple(
                        int(tag) for tag in tags if isinstance(tag, int)
                    ),
                    interface_type=interface_type,
                    source_id=_optional_string(info.get("source_id")),
                    face_kind=_optional_string(info.get("face_kind")),
                    representation=route,
                    physical_group_name=str(info.get("physical_name", name)),
                )
            )
    return InterfaceSurfaceCatalog(tuple(surfaces))


def _optional_string(value: Any) -> str | None:
    """Return a non-empty string or ``None``."""
    return value if isinstance(value, str) and value else None


__all__ = [
    "InterfaceSurface",
    "InterfaceSurfaceCatalog",
    "build_interface_surface_catalog",
]
