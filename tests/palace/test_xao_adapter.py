"""Tests for semantic-geometry-builder XAO physical-group adaptation."""

from __future__ import annotations

from gsim.palace.mesh import surface_epr as surface_epr_catalog
from gsim.palace.mesh import xao_adapter


def test_domain_boundary_surface_records_map_to_boundary_surfaces(monkeypatch):
    """Domain-boundary records should not become Surface EPR metadata."""
    records = [
        {
            "physical_name": "BOUNDARY__outer",
            "dimension": 2,
            "role": "domain_boundary",
            "route": "B",
            "solver_use": "solver_active",
        }
    ]
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {(2, "BOUNDARY__outer"): (7, (101, 102))},
    )
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, 0.0, 10.0, 20.0, 30.0),
    )
    groups = xao_adapter._groups_from_sgb_records(
        records=records,
        semantic_layer_map={},
        semantic_stack_layer_map={},
    )

    # Injected live physical groups are validated in this unit under monkeypatch.
    assert "BOUNDARY__outer" in groups["boundary_surfaces"]
    boundary_info = groups["boundary_surfaces"]["BOUNDARY__outer"]

    assert boundary_info["surface_epr"] is False
    assert boundary_info["source"] == "domain_boundary"
    assert boundary_info["bbox"] == [0.0, 0.0, 0.0, 10.0, 20.0, 30.0]
    assert "interface_type" not in boundary_info

    catalog = surface_epr_catalog.build_interface_surface_catalog(groups)
    assert not catalog.surfaces
