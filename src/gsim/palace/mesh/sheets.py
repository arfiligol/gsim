"""Layout-authored solver sheet selection for Palace mesh generation.

This module extracts polygons from PDK-declared simulation layers, selects the
unique horizontal solver sheet requested by a port, and converts selected
simulation-only polygons into 2D Gmsh surfaces.

Ports still have to be declared explicitly, and the PDK owns the catalog that
marks simulation-only layers. Material extraction, physical group assignment,
and Palace configuration generation happen elsewhere in the mesh pipeline. The
authored-sheet flow starts with ``add_port(generate_sheet=False)`` and rejoins
the normal port physical-group to ``Boundaries.LumpedPort`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely import Point
from shapely import Polygon as ShapelyPolygon

from gsim.palace.models.simulation_layers import (
    SimulationLayerCatalog,
    normalize_gds_layer,
)

from . import gmsh_utils


@dataclass(frozen=True)
class AuthoredSheetPolygon:
    """Polygon on a registered simulation layer."""

    gds_layer: tuple[int, int]
    pts_x: list[float]
    pts_y: list[float]
    holes: list[tuple[list[float], list[float]]]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return the polygon XY bounding box."""
        return min(self.pts_x), min(self.pts_y), max(self.pts_x), max(self.pts_y)

    def covers(self, point: tuple[float, float]) -> bool:
        """Return whether the polygon covers the point, including boundary."""
        shell = list(zip(self.pts_x, self.pts_y, strict=True))
        holes = [
            list(zip(hole_x, hole_y, strict=True)) for hole_x, hole_y in self.holes
        ]
        return bool(ShapelyPolygon(shell, holes).covers(Point(point)))


def extract_authored_sheet_polygons(
    component: Any,
    catalog: SimulationLayerCatalog,
) -> dict[tuple[int, int], list[AuthoredSheetPolygon]]:
    """Extract polygons from registered simulation-only GDS layers."""
    if component is None:
        raise ValueError("Authored sheet selection requires a component.")

    registered_layers = catalog.gds_layers
    polygons_by_index = component.get_polygons()
    layout = component.kcl.layout

    index_to_gds: dict[int, tuple[int, int]] = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    result: dict[tuple[int, int], list[AuthoredSheetPolygon]] = {}
    for layer_index, polygons in polygons_by_index.items():
        gds_layer = index_to_gds.get(layer_index)
        if gds_layer not in registered_layers:
            continue

        for polygon in polygons:
            points = list(polygon.each_point_hull())
            if len(points) < 3:
                continue
            pts_x = [point.x / 1000.0 for point in points]
            pts_y = [point.y / 1000.0 for point in points]

            holes: list[tuple[list[float], list[float]]] = []
            for hole_index in range(polygon.holes()):
                hole_points = list(polygon.each_point_hole(hole_index))
                if len(hole_points) >= 3:
                    holes.append(
                        (
                            [point.x / 1000.0 for point in hole_points],
                            [point.y / 1000.0 for point in hole_points],
                        )
                    )

            result.setdefault(gds_layer, []).append(
                AuthoredSheetPolygon(
                    gds_layer=gds_layer,
                    pts_x=pts_x,
                    pts_y=pts_y,
                    holes=holes,
                )
            )

    return result


def select_authored_sheet_polygon(
    polygons_by_layer: dict[tuple[int, int], list[AuthoredSheetPolygon]],
    *,
    gds_layer: tuple[int, int],
    center: tuple[float, float],
    port_name: str,
) -> AuthoredSheetPolygon:
    """Select exactly one authored sheet polygon for a port center."""
    candidates = [
        polygon
        for polygon in polygons_by_layer.get(gds_layer, [])
        if polygon.covers(center)
    ]
    if not candidates:
        raise ValueError(
            f"Port '{port_name}' requested authored sheet on GDS layer "
            f"{gds_layer}, but no polygon covers center {center}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Port '{port_name}' requested authored sheet on GDS layer "
            f"{gds_layer}, but {len(candidates)} polygons cover center {center}."
        )
    return candidates[0]


def create_authored_sheet_surface(
    kernel: Any,
    polygon: AuthoredSheetPolygon,
    *,
    z: float,
) -> int:
    """Create a Gmsh surface for a selected authored sheet polygon."""
    surface = gmsh_utils.create_polygon_surface(
        kernel,
        polygon.pts_x,
        polygon.pts_y,
        z,
        holes=polygon.holes,
    )
    if surface is None:
        raise ValueError(
            f"Could not create authored solver sheet on GDS layer {polygon.gds_layer}."
        )
    return surface


def authored_sheet_gds_layer(value: Any) -> tuple[int, int]:
    """Normalize the GDS layer recorded on a gdsfactory port."""
    return normalize_gds_layer(value)


__all__ = [
    "AuthoredSheetPolygon",
    "authored_sheet_gds_layer",
    "create_authored_sheet_surface",
    "extract_authored_sheet_polygons",
    "select_authored_sheet_polygon",
]
