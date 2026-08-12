"""Tests for semantic-geometry-builder XAO physical-group adaptation."""

from __future__ import annotations

import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh import surface_epr as surface_epr_catalog
from gsim.palace.mesh import xao_adapter
from gsim.palace.mesh.postprocessing import build_surface_epr_dielectric_specs
from gsim.palace.models import ActivatedRegion


def test_domain_boundary_surface_records_map_to_boundary_surfaces(monkeypatch):
    """Domain-boundary records should not become Surface EPR metadata."""
    records = [
        {
            "physical_name": "BOUNDARY__AIR_ABOVE__TOP",
            "dimension": 2,
            "role": "domain_boundary",
            "route": "B",
            "solver_use": "solver_active",
            "metadata": {"source_record_ids": ("SURF__BOUNDARY__AIR_ABOVE__TOP",)},
        },
        {
            "physical_name": "BOUNDARY__AIR_BELOW__BOTTOM",
            "dimension": 2,
            "role": "domain_boundary",
            "route": "B",
            "solver_use": "solver_active",
            "metadata": {"source_record_ids": ("SURF__BOUNDARY__AIR_BELOW__BOTTOM",)},
        },
    ]
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {
            (2, "BOUNDARY__AIR_ABOVE__TOP"): (7, (101,)),
            (2, "BOUNDARY__AIR_BELOW__BOTTOM"): (8, (102,)),
        },
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
    assert set(groups["boundary_surfaces"]) == {
        "BOUNDARY__AIR_ABOVE__TOP",
        "BOUNDARY__AIR_BELOW__BOTTOM",
    }
    boundary_info = groups["boundary_surfaces"]["BOUNDARY__AIR_ABOVE__TOP"]

    assert boundary_info["surface_epr"] is False
    assert boundary_info["source"] == "domain_boundary"
    assert boundary_info["source_record_ids"] == ("SURF__BOUNDARY__AIR_ABOVE__TOP",)
    assert boundary_info["bbox"] == [0.0, 0.0, 0.0, 10.0, 20.0, 30.0]
    assert "interface_type" not in boundary_info

    catalog = surface_epr_catalog.build_interface_surface_catalog(groups)
    assert not catalog.surfaces


def test_sgb_lowering_creates_symmetric_outer_vacuum_solution_regions():
    """Outer vacuum lowers above and below the full activated substrate stack."""
    stack = LayerStack(
        layers={
            "D0": Layer(
                name="D0",
                gds_layer=(1, 0),
                zmin=-500.0,
                zmax=0.0,
                thickness=500.0,
                material="silicon",
                layer_type="substrate",
            ),
            "GAP": Layer(
                name="GAP",
                gds_layer=(2, 0),
                zmin=0.0,
                zmax=10.0,
                thickness=10.0,
                material="vacuum",
                layer_type="dielectric",
            ),
            "D1": Layer(
                name="D1",
                gds_layer=(3, 0),
                zmin=10.0,
                zmax=510.0,
                thickness=500.0,
                material="silicon",
                layer_type="substrate",
            ),
            "OUTER": Layer(
                name="OUTER",
                gds_layer=(4, 0),
                zmin=-500.0,
                zmax=510.0,
                thickness=1010.0,
                material="vacuum",
                layer_type="dielectric",
            ),
        },
        materials={"silicon": {"permittivity": 11.45}, "vacuum": {"permittivity": 1.0}},
    )

    regions, stack_layers, host_void_regions = (
        xao_adapter._solution_regions_from_activated(
            stack=stack,
            activated_regions=(
                ActivatedRegion(layer="D0", role="substrate"),
                ActivatedRegion(
                    layer="GAP", role="inter_die_vacuum", lower_die="D0", upper_die="D1"
                ),
                ActivatedRegion(layer="D1", role="substrate"),
                ActivatedRegion(
                    layer="OUTER",
                    role="outer_vacuum",
                    z_above=1000.0,
                    z_below=1000.0,
                ),
            ),
            component_bounds=(0.0, 0.0, 20.0, 30.0),
        )
    )

    assert regions["AIR_ABOVE"]["geometry"]["z_min_um"] == 510.0
    assert regions["AIR_ABOVE"]["geometry"]["z_max_um"] == 1510.0
    assert regions["AIR_BELOW"]["geometry"]["z_min_um"] == -1500.0
    assert regions["AIR_BELOW"]["geometry"]["z_max_um"] == -500.0
    assert stack_layers["AIR_ABOVE"] == "OUTER"
    assert stack_layers["AIR_BELOW"] == "OUTER"
    assert ("AIR_ABOVE", 510.0, 1510.0) in host_void_regions
    assert ("AIR_BELOW", -1500.0, -500.0) in host_void_regions

    without_lower, _, _ = xao_adapter._solution_regions_from_activated(
        stack=stack,
        activated_regions=(
            ActivatedRegion(layer="D0", role="substrate"),
            ActivatedRegion(layer="D1", role="substrate"),
            ActivatedRegion(layer="OUTER", role="outer_vacuum", z_above=1000.0),
        ),
        component_bounds=(0.0, 0.0, 20.0, 30.0),
    )
    assert "AIR_BELOW" not in without_lower


def test_tokenless_sa_uses_sgb_provenance_and_neighbor_centers(monkeypatch):
    """Tokenless SA records derive the exact face from SGB topology provenance."""
    records = [
        {
            "physical_name": "D0",
            "dimension": 3,
            "role": "material_volume",
            "metadata": {"source_record_ids": ("VOL__D0",)},
        },
        {
            "physical_name": "AIR_ABOVE",
            "dimension": 3,
            "role": "material_volume",
            "metadata": {"source_record_ids": ("VOL__AIR_ABOVE",)},
        },
        {
            "physical_name": "AIR_BELOW",
            "dimension": 3,
            "role": "material_volume",
            "metadata": {"source_record_ids": ("VOL__AIR_BELOW",)},
        },
        {
            "physical_name": "GAP",
            "dimension": 3,
            "role": "material_volume",
            "metadata": {"source_record_ids": ("VOL__GAP",)},
        },
        {
            "physical_name": "D1",
            "dimension": 3,
            "role": "material_volume",
            "metadata": {"source_record_ids": ("VOL__D1",)},
        },
        {
            "physical_name": "SA__D1__AIR_ABOVE__0000",
            "dimension": 2,
            "role": "interface_surface",
            "route": "C",
            "solver_use": "solver_active",
            "metadata": {
                "source_record_ids": ("SURF__SA__D1__AIR_ABOVE__0000",),
            },
        },
        {
            "physical_name": "SA__D0__AIR_BELOW__0000",
            "dimension": 2,
            "role": "interface_surface",
            "route": "C",
            "solver_use": "solver_active",
            "metadata": {
                "source_record_ids": ("SURF__SA__D0__AIR_BELOW__0000",),
            },
        },
        {
            "physical_name": "SA__D0__GAP__0000",
            "dimension": 2,
            "role": "interface_surface",
            "route": "C",
            "solver_use": "solver_active",
            "metadata": {
                "source_record_ids": ("SURF__SA__D0__GAP__0000",),
            },
        },
        {
            "physical_name": "SA__D1__GAP__0000",
            "dimension": 2,
            "role": "interface_surface",
            "route": "C",
            "solver_use": "solver_active",
            "metadata": {
                "source_record_ids": ("SURF__SA__D1__GAP__0000",),
            },
        },
        {
            "physical_name": "SA__D0__AIR_ABOVE__BOTTOM",
            "dimension": 2,
            "role": "interface_surface",
            "route": "C",
            "solver_use": "solver_active",
        },
    ]
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {
            (3, "D0"): (1, (11,)),
            (3, "AIR_ABOVE"): (2, (12,)),
            (3, "AIR_BELOW"): (3, (13,)),
            (3, "GAP"): (4, (14,)),
            (3, "D1"): (5, (15,)),
            (2, "SA__D1__AIR_ABOVE__0000"): (6, (21,)),
            (2, "SA__D0__AIR_BELOW__0000"): (7, (22,)),
            (2, "SA__D0__GAP__0000"): (8, (23,)),
            (2, "SA__D1__GAP__0000"): (9, (24,)),
            (2, "SA__D0__AIR_ABOVE__BOTTOM"): (10, (25,)),
        },
    )
    bounding_boxes = {
        (3, 11): (0.0, 0.0, -500.0, 10.0, 10.0, 0.0),
        (3, 12): (0.0, 0.0, 0.0, 10.0, 10.0, 1000.0),
        (3, 13): (0.0, 0.0, -1500.0, 10.0, 10.0, -500.0),
        (3, 14): (0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
        (3, 15): (0.0, 0.0, 10.0, 10.0, 10.0, 510.0),
        (2, 21): (0.0, 0.0, 0.0, 10.0, 10.0, 0.0),
        (2, 22): (0.0, 0.0, -500.0, 10.0, 10.0, -500.0),
        (2, 23): (0.0, 0.0, 0.0, 10.0, 10.0, 0.0),
        (2, 24): (0.0, 0.0, 10.0, 10.0, 10.0, 10.0),
        (2, 25): (0.0, 0.0, -500.0, 10.0, 10.0, -500.0),
    }
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda dimension, tag: bounding_boxes[(dimension, tag)],
    )

    groups = xao_adapter._groups_from_sgb_records(
        records=records,
        semantic_layer_map={},
        semantic_stack_layer_map={},
    )
    info = groups["boundary_surfaces"]["SA__D1__AIR_ABOVE__0000"]

    assert info["face_kind"] == "top"
    assert info["source_id"] == "D1__AIR_ABOVE"
    assert info["surface_id"] == "SURF__SA__D1__AIR_ABOVE__0000"
    assert info["owner_semantic_ids"] == ("D1", "AIR_ABOVE")
    assert info["source_record_ids"] == ("SURF__SA__D1__AIR_ABOVE__0000",)
    assert (
        groups["boundary_surfaces"]["SA__D0__AIR_BELOW__0000"]["face_kind"] == "bottom"
    )
    assert groups["boundary_surfaces"]["SA__D0__GAP__0000"]["face_kind"] == "top"
    assert groups["boundary_surfaces"]["SA__D1__GAP__0000"]["face_kind"] == "bottom"
    catalog = surface_epr_catalog.build_interface_surface_catalog(groups)
    top_specs = build_surface_epr_dielectric_specs(
        catalog.surfaces,
        preset_name="sa",
        preset={"interface_type": "SA", "thickness": 0.003, "permittivity": 1.0},
        face_kind="top",
    )
    bottom_specs = build_surface_epr_dielectric_specs(
        catalog.surfaces,
        preset_name="sa",
        preset={"interface_type": "SA", "thickness": 0.003, "permittivity": 1.0},
        face_kind="bottom",
    )
    assert {spec.entry_name for spec in top_specs} == {
        "SA__D1__AIR_ABOVE__0000",
        "SA__D0__GAP__0000",
    }
    assert {spec.entry_name for spec in bottom_specs} == {
        "SA__D0__AIR_BELOW__0000",
        "SA__D1__GAP__0000",
        "SA__D0__AIR_ABOVE__BOTTOM",
    }
    assert all(spec.metadata["face_kind"] == "top" for spec in top_specs)
    assert all(spec.metadata["face_kind"] == "bottom" for spec in bottom_specs)


def test_tokenless_sa_rejects_ambiguous_neighbor_centers(monkeypatch):
    """Tokenless SA records fail closed when owner topology has no z direction."""
    records = [
        {
            "physical_name": "D0",
            "dimension": 3,
            "role": "material_volume",
        },
        {
            "physical_name": "AIR_ABOVE",
            "dimension": 3,
            "role": "material_volume",
        },
        {
            "physical_name": "SA__D0__AIR_ABOVE__0000",
            "dimension": 2,
            "role": "interface_surface",
            "route": "C",
            "solver_use": "solver_active",
            "metadata": {
                "source_record_ids": ("SURF__SA__D0__AIR_ABOVE__0000",),
            },
        },
    ]
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {
            (3, "D0"): (1, (11,)),
            (3, "AIR_ABOVE"): (2, (12,)),
            (2, "SA__D0__AIR_ABOVE__0000"): (3, (21,)),
        },
    )
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, 0.0, 10.0, 10.0, 0.0),
    )

    with pytest.raises(ValueError, match="ambiguous owner neighbor z centers"):
        xao_adapter._groups_from_sgb_records(
            records=records,
            semantic_layer_map={},
            semantic_stack_layer_map={},
        )
