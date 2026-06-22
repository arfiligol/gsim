"""Simulation-only layer catalog for Palace mesh inputs.

This module defines the PDK/project supplied catalog that identifies GDS layers
used only for solver geometry, such as layout-authored horizontal port sheets.
It validates those catalog entries and gives mesh code a stable lookup by full
GDS layer/datatype tuple.

The catalog does not declare ports, extract materials, create Gmsh surfaces, or
write Palace JSON. Callers pass catalog data into
``PalaceSimBase.set_simulation_layers()``, and mesh generation uses it only for
declared solver sheets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gsim.palace._shared import int_tuple

SimulationLayerRole = Literal["solver_boundary_sheet"]


def normalize_gds_layer(value: Any) -> tuple[int, int]:
    """Return a two-item GDS layer/datatype tuple."""
    layer = int_tuple(value)
    if len(layer) != 2:
        raise ValueError("Simulation layer GDS layer must be a two-item tuple.")
    return layer[0], layer[1]


class SimulationLayer(BaseModel):
    """PDK-declared simulation-only GDS layer."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    gds_layer: tuple[int, int]
    role: SimulationLayerRole = "solver_boundary_sheet"
    stack_layer: str | None = None
    z: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gds_layer", mode="before")
    @classmethod
    def _validate_gds_layer(cls, value: Any) -> tuple[int, int]:
        return normalize_gds_layer(value)

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, value: Any) -> Any:
        if value == "surface_epr_band":
            raise ValueError(
                "Source-polygon Surface EPR bands are not supported as "
                "simulation layers; use Gmsh-derived interface surfaces."
            )
        return value

    @field_validator("z")
    @classmethod
    def _validate_z(cls, value: float | None) -> float | None:
        if value is None:
            return None
        z = float(value)
        if not math.isfinite(z):
            raise ValueError("Simulation layer z must be finite.")
        return z

    def resolve_z(self, stack: Any) -> float:
        """Resolve the authored sheet z coordinate from catalog or stack."""
        if self.z is not None:
            return self.z

        if self.stack_layer is not None:
            try:
                return float(stack.layers[self.stack_layer].zmin)
            except KeyError as exc:
                raise ValueError(
                    f"Simulation layer '{self.name}' references missing stack layer "
                    f"'{self.stack_layer}'."
                ) from exc

        matches = [
            layer
            for layer in stack.layers.values()
            if normalize_gds_layer(layer.gds_layer) == self.gds_layer
        ]
        if len(matches) == 1:
            return float(matches[0].zmin)
        if not matches:
            raise ValueError(
                f"Simulation layer '{self.name}' needs 'z' or 'stack_layer'; "
                f"no stack layer uses GDS layer {self.gds_layer}."
            )
        raise ValueError(
            f"Simulation layer '{self.name}' maps to multiple stack layers; "
            "provide 'stack_layer' or 'z'."
        )


class SimulationLayerCatalog(BaseModel):
    """Catalog of simulation-only layers passed from a PDK/project."""

    model_config = ConfigDict(validate_assignment=True)

    layers: dict[str, SimulationLayer] = Field(default_factory=dict)

    def __init__(
        self,
        catalog: Mapping[str, Any] | None = None,
        **data: Any,
    ) -> None:
        """Accept either ``SimulationLayerCatalog({...})`` or keyword fields."""
        if catalog is not None:
            if "layers" in data:
                raise TypeError("Pass catalog data either positionally or as layers=.")
            data["layers"] = catalog
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> Any:
        if isinstance(value, SimulationLayerCatalog):
            return value
        if isinstance(value, Mapping) and "layers" not in value:
            value = {"layers": value}
        if not isinstance(value, Mapping):
            return value

        raw_layers = value.get("layers", {})
        if not isinstance(raw_layers, Mapping):
            return value

        layers: dict[str, Any] = {}
        for name, entry in raw_layers.items():
            if isinstance(entry, SimulationLayer):
                payload = entry.model_dump()
            elif isinstance(entry, Mapping):
                payload = dict(entry)
            else:
                payload = {"gds_layer": entry}
            payload.setdefault("name", str(name))
            layers[str(name)] = payload

        data = dict(value)
        data["layers"] = layers
        return data

    @model_validator(mode="after")
    def _validate_unique_gds_layers(self) -> Self:
        seen: dict[tuple[int, int], str] = {}
        for name, layer in self.layers.items():
            if layer.gds_layer in seen:
                raise ValueError(
                    f"Simulation layers '{seen[layer.gds_layer]}' and '{name}' "
                    f"share GDS layer {layer.gds_layer}."
                )
            seen[layer.gds_layer] = name
        return self

    @property
    def gds_layers(self) -> set[tuple[int, int]]:
        """Return the registered full GDS layer tuples."""
        return {layer.gds_layer for layer in self.layers.values()}

    def for_gds_layer(self, value: Any) -> SimulationLayer | None:
        """Return the catalog entry for a full GDS layer tuple."""
        gds_layer = normalize_gds_layer(value)
        for layer in self.layers.values():
            if layer.gds_layer == gds_layer:
                return layer
        return None


__all__ = [
    "SimulationLayer",
    "SimulationLayerCatalog",
    "SimulationLayerRole",
    "normalize_gds_layer",
]
