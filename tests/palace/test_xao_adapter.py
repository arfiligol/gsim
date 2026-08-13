"""Tests for semantic-geometry-builder XAO physical-group adaptation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from gsim.common.stack import Layer, LayerStack
from gsim.palace.electrostatic import ElectrostaticSim
from gsim.palace.mesh import surface_epr as surface_epr_catalog
from gsim.palace.mesh import xao_adapter
from gsim.palace.mesh.postprocessing import build_surface_epr_dielectric_specs
from gsim.palace.models import ActivatedRegion, TerminalConfig


def _structured_conductor_record(**overrides):
    """Return one explicit current Route A/B SGB conductor surface."""
    record = {
        "physical_name": "display-only-not-a-parser-input",
        "dimension": 2,
        "role": "cutout_boundary_shell",
        "route": "A",
        "solver_use": "solver_active",
        "representation": "cutout_boundary_shell",
        "surface_id": "SURFACE__EXPLICIT",
        "interface_type": "MA",
        "contact_kind": None,
        "face_kind": "top",
        "owner_semantic_ids": ["M1", "AIR_ABOVE"],
        "adjacent_solution_volume_ids": ["AIR_ABOVE"],
        "conductor_component_id": "COMP__SIGNAL",
        "net_id": "M1@signal",
        "equipotential_id": "EQ_SIGNAL",
        "source_provenance": {
            "sources": [
                {
                    "conductor_source_layer_name": "M1",
                    "gds_layer": 40,
                    "gds_datatype": 1,
                }
            ]
        },
        "physical_attribute": {"solver_use": "solver_active"},
        "metadata": {"source_record_ids": ["SURF__EXPLICIT"]},
    }
    record.update(overrides)
    return record


def _structured_solution_volume_record(name: str, material_id: str):
    """Return one current Route-A/B structured solution-volume record."""
    return {
        "physical_name": name,
        "dimension": 3,
        "role": "material_volume",
        "route": "A",
        "solver_use": "solver_active",
        "representation": "solution_volume",
        "source_provenance": {"volume_ids": [f"VOL__{name}"]},
        "physical_attribute": {"material_ids": [material_id]},
    }


def test_domain_boundary_surface_records_map_to_boundary_surfaces(monkeypatch):
    """Domain-boundary records should not become Surface EPR metadata."""
    records = [
        {
            "physical_name": "BOUNDARY__AIR_ABOVE__TOP",
            "dimension": 2,
            "role": "domain_boundary",
            "route": "C",
            "solver_use": "solver_active",
            "metadata": {"source_record_ids": ("SURF__BOUNDARY__AIR_ABOVE__TOP",)},
        },
        {
            "physical_name": "BOUNDARY__AIR_BELOW__BOTTOM",
            "dimension": 2,
            "role": "domain_boundary",
            "route": "C",
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
        route_context="C",
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


@pytest.mark.parametrize(
    ("entity_tags", "bad_bbox"),
    [
        ((101, 102), (0.0, 0.0, float("nan"), 1.0, 1.0, 1.0)),
        ((102, 101), (0.0, 0.0, float("nan"), 1.0, 1.0, 1.0)),
        ((101, 102), (0.0, 0.0, 0.0, 1.0, float("inf"), 1.0)),
        ((102, 101), (0.0, 0.0, 0.0, 1.0, float("inf"), 1.0)),
    ],
)
def test_entities_bbox_rejects_nonfinite_raw_entity_before_aggregation(
    monkeypatch, entity_tags, bad_bbox
):
    """A later malformed Gmsh entity cannot be masked by bbox aggregation."""
    bounding_boxes = {
        101: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        102: bad_bbox,
    }
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, tag: bounding_boxes[tag],
    )

    with pytest.raises(ValueError, match=r"dimension 2 entity 102 is non-finite"):
        xao_adapter._entities_bbox(2, entity_tags)


def test_entities_bbox_rejects_malformed_raw_entity_shape(monkeypatch):
    """Every Gmsh entity bbox must have exactly six coordinates."""
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, 0.0, 1.0, 1.0),
    )

    with pytest.raises(ValueError, match=r"dimension 2 entity 101 is malformed"):
        xao_adapter._entities_bbox(2, (101,))


@pytest.mark.parametrize(
    "bad_bbox",
    [
        ("0.0", 0.0, 0.0, 1.0, 1.0, 1.0),
        (False, 0.0, 0.0, 1.0, 1.0, 1.0),
    ],
)
def test_entities_bbox_rejects_nonreal_raw_coordinates(monkeypatch, bad_bbox):
    """Numeric strings and booleans cannot enter Gmsh bbox aggregation."""
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: bad_bbox,
    )

    with pytest.raises(ValueError, match=r"dimension 2 entity 101 is malformed"):
        xao_adapter._entities_bbox(2, (101,))


def test_surface_z_center_accepts_occ_planar_tolerance(monkeypatch):
    """OCC's +/-1e-7 sheet span remains planar under the shared tolerance."""
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, -1e-7, 1.0, 1.0, 1e-7),
    )

    assert xao_adapter._surface_z_center((101,)) == pytest.approx(0.0)


def test_surface_z_center_rejects_materially_nonplanar_span(monkeypatch):
    """A span just beyond the existing planar tolerance remains nonplanar."""
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, 0.0, 1.0, 1.0, 1.000001e-6),
    )

    assert xao_adapter._surface_z_center((101,)) is None


def test_surface_z_center_rejects_exact_planar_tolerance(monkeypatch):
    """Planarity keeps the shared strict less-than boundary semantics."""
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, 0.0, 1.0, 1.0, 1e-6),
    )

    assert xao_adapter._surface_z_center((101,)) is None


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


def test_route_ab_structured_surface_uses_explicit_authority_not_display_name(
    monkeypatch,
):
    """Route A/B surfaces use structured provenance without name or z inference."""
    record = _structured_conductor_record()
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {(2, record["physical_name"]): (17, (101,))},
    )
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda _dimension, _tag: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )

    groups = xao_adapter._groups_from_sgb_records(
        records=[record],
        semantic_layer_map={"M1": "M1_STACK"},
        semantic_stack_layer_map={},
        route_context="A",
    )

    info = groups["boundary_surfaces"]["SURFACE__EXPLICIT"]
    assert info["layer"] == "M1_STACK"
    assert info["metal_body_id"] == "M1_STACK"
    assert info["conductor_component_id"] == "COMP__SIGNAL"
    assert info["net_id"] == "M1@signal"
    assert info["equipotential_id"] == "EQ_SIGNAL"
    assert info["source_provenance"] == record["source_provenance"]


@pytest.mark.parametrize(
    ("adjacent_ids", "volume_bounds", "expected_faces"),
    [
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": (0.0, 0.0, -10.0, 1.0, 1.0, 0.0),
                "GAP": (0.0, 0.0, 0.0, 1.0, 1.0, 10.0),
            },
            {"MS": "bottom", "MA": "top"},
        ),
        (
            ("GAP", "SUBSTRATE"),
            {
                "SUBSTRATE": (0.0, 0.0, 0.0, 1.0, 1.0, 10.0),
                "GAP": (0.0, 0.0, -10.0, 1.0, 1.0, 0.0),
            },
            {"MS": "top", "MA": "bottom"},
        ),
    ],
)
def test_route_ab_structured_ms_ma_sheet_splits_logical_faces_without_duplication(
    monkeypatch, adjacent_ids, volume_bounds, expected_faces
):
    """One structured sheet has opposite MS/MA children from typed topology."""
    sheet = _structured_conductor_record(
        physical_name="display-only-not-a-parser-input",
        representation="surface_sheet",
        surface_id="SHEET__EXPLICIT",
        interface_type="MS_MA",
        face_kind="interface",
        owner_semantic_ids=["M1", "SUBSTRATE", "GAP"],
        adjacent_solution_volume_ids=list(adjacent_ids),
    )
    volumes = [
        {
            "physical_name": name,
            "dimension": 3,
            "role": "material_volume",
            "route": "A",
            "solver_use": "solver_active",
            "representation": "solution_volume",
            "surface_id": None,
            "interface_type": "material_volume",
            "contact_kind": None,
            "face_kind": "volume",
            "owner_semantic_ids": [name],
            "adjacent_solution_volume_ids": [],
            "conductor_component_id": None,
            "net_id": None,
            "equipotential_id": None,
            "source_provenance": {"volume_ids": [f"VOL__{name}"]},
            "physical_attribute": {
                "material_ids": ["silicon" if name == "SUBSTRATE" else "vacuum"]
            },
        }
        for name in ("SUBSTRATE", "GAP")
    ]
    live = {
        (3, "SUBSTRATE"): (1, (11,)),
        (3, "GAP"): (2, (12,)),
        (2, sheet["physical_name"]): (17, (101,)),
    }
    bounding_boxes = {
        (3, 11): volume_bounds["SUBSTRATE"],
        (3, 12): volume_bounds["GAP"],
        (2, 101): (0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    }
    monkeypatch.setattr(xao_adapter, "_live_physical_groups", lambda: live)
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda dimension, tag: bounding_boxes[(dimension, tag)],
    )

    groups = xao_adapter._groups_from_sgb_records(
        records=[*volumes, sheet],
        semantic_layer_map={"M1": "M1_STACK"},
        semantic_stack_layer_map={"SUBSTRATE": "D0", "GAP": "OUTER"},
        route_context="A",
    )

    interfaces = {
        info["interface_type"]: info for info in groups["boundary_surfaces"].values()
    }
    assert set(interfaces) == {"MS", "MA"}
    assert {
        kind: info["face_kind"] for kind, info in interfaces.items()
    } == expected_faces
    assert {tuple(info["tags"]) for info in interfaces.values()} == {(101,)}
    assert {info["physical_group_attribute"] for info in interfaces.values()} == {17}
    assert {info["source_id"] for info in interfaces.values()} == {"M1"}
    assert {info["layer"] for info in interfaces.values()} == {"M1_STACK"}
    assert all(info["surface_id"] == "SHEET__EXPLICIT" for info in interfaces.values())
    assert all(
        info["source_provenance"] == sheet["source_provenance"]
        for info in interfaces.values()
    )
    assert set(groups["pec_surfaces"]) == {sheet["physical_name"]}
    assert groups["pec_surfaces"][sheet["physical_name"]]["tags"] == [101]

    catalog = surface_epr_catalog.build_interface_surface_catalog(groups)
    specs = {
        interface_type: build_surface_epr_dielectric_specs(
            catalog.surfaces,
            preset_name=interface_type.lower(),
            preset={
                "interface_type": interface_type,
                "thickness": 0.003,
                "permittivity": 1.0,
            },
            face_kind=expected_faces[interface_type],
        )
        for interface_type in ("MS", "MA")
    }
    assert all(len(value) == 1 for value in specs.values())
    assert specs["MS"][0].entry_names != specs["MA"][0].entry_names

    sim = ElectrostaticSim()
    sim._last_mesh_result = SimpleNamespace(
        groups=groups,
        manifest=xao_adapter.build_mesh_manifest(groups),
    )
    sim.set_surface_epr(
        representation="A",
        interfaces={
            interface_type: {
                "preset": {
                    "thickness": 0.003,
                    "permittivity": 1.0,
                },
                "face_kind": expected_faces[interface_type],
            }
            for interface_type in ("MS", "MA")
        },
    )
    postprocessing = sim._build_surface_epr_postprocessing_config()
    selected = {
        entry.metadata["interface_type"]: entry
        for entry in postprocessing.index_map.entries
        if entry.section == "Boundaries.Postprocessing.Dielectric"
    }
    assert {kind: entry.metadata["face_kind"] for kind, entry in selected.items()} == (
        expected_faces
    )
    assert {entry.attributes for entry in selected.values()} == {(17,)}


@pytest.mark.parametrize(
    ("adjacent_ids", "volume_bounds", "expected_face"),
    [
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": (0.0, 0.0, -10.0, 1.0, 1.0, 0.0),
                "GAP": (0.0, 0.0, 0.0, 1.0, 1.0, 10.0),
            },
            "top",
        ),
        (
            ("GAP", "SUBSTRATE"),
            {
                "SUBSTRATE": (0.0, 0.0, 0.0, 1.0, 1.0, 10.0),
                "GAP": (0.0, 0.0, -10.0, 1.0, 1.0, 0.0),
            },
            "bottom",
        ),
    ],
)
def test_route_ab_structured_sa_derives_mirrored_face_from_volume_topology(
    monkeypatch, adjacent_ids, volume_bounds, expected_face
):
    """Structured SA stays one logical face with typed top/bottom authority."""
    surface = _structured_conductor_record(
        physical_name="display-only-not-a-parser-input",
        role="solution_interface",
        representation="solution_surface",
        surface_id="SA__EXPLICIT",
        interface_type="SA",
        conductor_component_id=None,
        net_id=None,
        equipotential_id=None,
        source_provenance={"sources": [{"route": "A"}]},
        face_kind="interface",
        owner_semantic_ids=["SUBSTRATE", "GAP"],
        adjacent_solution_volume_ids=list(adjacent_ids),
    )
    records = [
        _structured_solution_volume_record("SUBSTRATE", "silicon"),
        _structured_solution_volume_record("GAP", "vacuum"),
        surface,
    ]
    live = {
        (3, "SUBSTRATE"): (1, (11,)),
        (3, "GAP"): (2, (12,)),
        (2, surface["physical_name"]): (17, (101,)),
    }
    bounding_boxes = {
        (3, 11): volume_bounds["SUBSTRATE"],
        (3, 12): volume_bounds["GAP"],
        (2, 101): (0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
    }
    monkeypatch.setattr(xao_adapter, "_live_physical_groups", lambda: live)
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda dimension, tag: bounding_boxes[(dimension, tag)],
    )

    groups = xao_adapter._groups_from_sgb_records(
        records=records,
        semantic_layer_map={},
        semantic_stack_layer_map={},
        route_context="A",
    )

    assert set(groups["boundary_surfaces"]) == {"SA__EXPLICIT"}
    info = groups["boundary_surfaces"]["SA__EXPLICIT"]
    assert info["interface_type"] == "SA"
    assert info["face_kind"] == expected_face
    assert info["physical_group_attribute"] == 17
    assert info["tags"] == [101]
    assert info["owner_semantic_ids"] == ("SUBSTRATE", "GAP")
    assert info["source_provenance"] == surface["source_provenance"]
    assert {"layer", "metal_body_id", "metal_volume_id"}.isdisjoint(info)
    assert not groups["pec_surfaces"]

    catalog = surface_epr_catalog.build_interface_surface_catalog(groups)
    [spec] = build_surface_epr_dielectric_specs(
        catalog.surfaces,
        preset_name="sa",
        preset={"interface_type": "SA", "thickness": 0.003, "permittivity": 1.0},
        face_kind=expected_face,
    )
    assert spec.entry_names == ("SA__EXPLICIT",)


def test_route_ab_structured_sa_rejects_ambiguous_volume_topology(monkeypatch):
    """Equal dielectric/vacuum centers cannot decide an SA logical face."""
    surface = _structured_conductor_record(
        role="solution_interface",
        representation="solution_surface",
        surface_id="SA__EXPLICIT",
        interface_type="SA",
        conductor_component_id=None,
        net_id=None,
        equipotential_id=None,
        source_provenance={"sources": [{"route": "A"}]},
        face_kind="interface",
        owner_semantic_ids=["SUBSTRATE", "GAP"],
        adjacent_solution_volume_ids=["SUBSTRATE", "GAP"],
    )
    records = [
        _structured_solution_volume_record("SUBSTRATE", "silicon"),
        _structured_solution_volume_record("GAP", "vacuum"),
        surface,
    ]
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {
            (3, "SUBSTRATE"): (1, (11,)),
            (3, "GAP"): (2, (12,)),
            (2, surface["physical_name"]): (17, (101,)),
        },
    )
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda dimension, tag: {
            (3, 11): (0.0, 0.0, -1.0, 1.0, 1.0, 1.0),
            (3, 12): (0.0, 0.0, -1.0, 1.0, 1.0, 1.0),
            (2, 101): (0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        }[(dimension, tag)],
    )

    with pytest.raises(ValueError, match="ambiguous adjacent-volume topology"):
        xao_adapter._groups_from_sgb_records(
            records=records,
            semantic_layer_map={},
            semantic_stack_layer_map={},
            route_context="A",
        )


@pytest.mark.parametrize(
    ("adjacent_ids", "volumes", "centers", "error"),
    [
        (
            (),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
                "GAP": {"physical_attribute": {"material_ids": ["vacuum"]}},
            },
            {"SUBSTRATE": -1.0, "GAP": 1.0},
            "exactly two distinct solution volumes",
        ),
        (
            ("SUBSTRATE", "SUBSTRATE"),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
            },
            {"SUBSTRATE": -1.0},
            "exactly two distinct solution volumes",
        ),
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
            },
            {"SUBSTRATE": -1.0, "GAP": 1.0},
            "lacks structured volume",
        ),
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
                "GAP": {"physical_attribute": {}},
            },
            {"SUBSTRATE": -1.0, "GAP": 1.0},
            "physical_attribute.material_ids",
        ),
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
                "GAP": {"physical_attribute": {"material_ids": ["vacuum", "air"]}},
            },
            {"SUBSTRATE": -1.0, "GAP": 1.0},
            "one exact material identity",
        ),
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
                "GAP": {"physical_attribute": {"material_ids": ["SiO2"]}},
            },
            {"SUBSTRATE": -1.0, "GAP": 1.0},
            "one dielectric and one vacuum volume",
        ),
        (
            ("SUBSTRATE", "GAP"),
            {
                "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
                "GAP": {"physical_attribute": {"material_ids": ["vacuum"]}},
            },
            {"SUBSTRATE": -1.0, "GAP": float("nan")},
            "finite adjacent-volume topology",
        ),
    ],
)
def test_route_ab_structured_sa_rejects_incomplete_volume_authority(
    adjacent_ids, volumes, centers, error
):
    """Every structured SA material/topology authority branch fails closed."""
    record = {
        "surface_id": "SA__EXPLICIT",
        "adjacent_solution_volume_ids": list(adjacent_ids),
    }

    with pytest.raises(ValueError, match=error):
        xao_adapter._structured_sa_surface_record(
            name="SA__EXPLICIT",
            record=record,
            volume_z_centers=centers,
            structured_volumes=volumes,
        )


def test_route_a_legacy_ms_only_catalog_keeps_ma_clone_fallback():
    """Legacy Route A keeps its MA request synthesized from the MS catalog."""
    groups = {
        "boundary_surfaces": {
            "MS__LEGACY": {
                "phys_group": 17,
                "tags": [101],
                "dim": 2,
                "source": "volume_interface",
                "surface_epr": True,
                "interface_id": "MS__LEGACY",
                "interface_type": "MS",
                "source_id": "M1",
                "face_kind": "bottom",
                "representation": "A",
            }
        }
    }
    sim = ElectrostaticSim()
    sim._last_mesh_result = SimpleNamespace(
        groups=groups,
        manifest=xao_adapter.build_mesh_manifest(groups),
    )
    sim.set_surface_epr(
        representation="A",
        interfaces={
            "MA": {
                "preset": {"thickness": 0.003, "permittivity": 1.0},
                "face_kind": "top",
            }
        },
    )

    postprocessing = sim._build_surface_epr_postprocessing_config()
    [entry] = [
        entry
        for entry in postprocessing.index_map.entries
        if entry.section == "Boundaries.Postprocessing.Dielectric"
    ]
    assert entry.metadata["interface_type"] == "MA"
    assert entry.metadata["face_kind"] == "top"
    assert entry.attributes == (17,)


@pytest.mark.parametrize(
    ("sheet_z_center", "substrate_center", "vacuum_center"),
    [
        (float("nan"), -1.0, 1.0),
        (0.0, float("nan"), 1.0),
        (0.0, -1.0, float("nan")),
        (float("-inf"), -1.0, 1.0),
        (float("inf"), -1.0, 1.0),
    ],
)
def test_route_ab_structured_ms_ma_rejects_nonfinite_topology(
    sheet_z_center, substrate_center, vacuum_center
):
    """Structured sheet orientation cannot compare non-finite topology values."""
    sheet = _structured_conductor_record(
        interface_type="MS_MA",
        face_kind="interface",
        adjacent_solution_volume_ids=["SUBSTRATE", "GAP"],
    )
    volumes = {
        "SUBSTRATE": {"physical_attribute": {"material_ids": ["silicon"]}},
        "GAP": {"physical_attribute": {"material_ids": ["vacuum"]}},
    }

    with pytest.raises(ValueError, match="finite adjacent-volume topology"):
        xao_adapter._structured_ms_ma_surface_records(
            name=sheet["physical_name"],
            record=sheet,
            volume_z_centers={
                "SUBSTRATE": substrate_center,
                "GAP": vacuum_center,
            },
            structured_volumes=volumes,
            sheet_z_center=sheet_z_center,
        )


def test_route_ab_structured_sa_has_no_metal_layer_identity(monkeypatch):
    """Structured solution interfaces never inherit conductor catalog fields."""
    record = _structured_conductor_record(
        interface_type="SA",
        conductor_component_id=None,
        net_id=None,
        equipotential_id=None,
        source_provenance={"sources": [{"route": "A"}]},
        surface_id="SA__EXPLICIT",
        owner_semantic_ids=["SUBSTRATE", "GAP"],
        adjacent_solution_volume_ids=["SUBSTRATE", "GAP"],
    )
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {
            (3, "SUBSTRATE"): (1, (11,)),
            (3, "GAP"): (2, (12,)),
            (2, record["physical_name"]): (17, (101,)),
        },
    )
    monkeypatch.setattr(
        xao_adapter.gmsh.model,
        "getBoundingBox",
        lambda dimension, tag: {
            (3, 11): (0.0, 0.0, -1.0, 1.0, 1.0, 0.0),
            (3, 12): (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (2, 101): (0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        }[(dimension, tag)],
    )

    groups = xao_adapter._groups_from_sgb_records(
        records=[
            _structured_solution_volume_record("SUBSTRATE", "silicon"),
            _structured_solution_volume_record("GAP", "vacuum"),
            record,
        ],
        semantic_layer_map={"M1": "M1_STACK"},
        semantic_stack_layer_map={},
        route_context="A",
    )

    info = groups["boundary_surfaces"]["SA__EXPLICIT"]
    assert {"layer", "metal_body_id", "metal_volume_id"}.isdisjoint(info)


@pytest.mark.parametrize("field", ["representation", "source_provenance", "net_id"])
def test_route_ab_structured_surface_missing_field_fails_closed(monkeypatch, field):
    """Any incomplete current Route A/B record cannot fall through to legacy."""
    record = _structured_conductor_record()
    record.pop(field)
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {(2, record["physical_name"]): (17, (101,))},
    )

    with pytest.raises((TypeError, ValueError), match=r"structured|misses|mapping"):
        xao_adapter._groups_from_sgb_records(
            records=[record],
            semantic_layer_map={},
            semantic_stack_layer_map={},
            route_context="A",
        )


def test_route_ab_structured_surface_requires_explicit_source_layer(monkeypatch):
    """A/B provenance cannot fall back to pre-contract source aliases."""
    record = _structured_conductor_record(
        source_provenance={"sources": [{"source_layer_name": "M1"}]}
    )
    monkeypatch.setattr(
        xao_adapter,
        "_live_physical_groups",
        lambda: {(2, record["physical_name"]): (17, (101,))},
    )

    with pytest.raises(ValueError, match="conductor_source_layer_name"):
        xao_adapter._groups_from_sgb_records(
            records=[record],
            semantic_layer_map={},
            semantic_stack_layer_map={},
            route_context="A",
        )


@pytest.mark.parametrize(
    ("part_role", "expected_a"),
    [
        ("face_metal", "surface_sheet"),
        ("contact_pad", "cutout_boundary_shell"),
        ("bump_body", "cutout_boundary_shell"),
    ],
)
@pytest.mark.parametrize("route", ["A", "B"])
def test_route_ab_typed_layer_lowering_preserves_authored_identity(
    part_role, expected_a, route
):
    """Only explicit typed roles choose A/B representations and provenance."""
    layer = Layer(
        name="M1",
        gds_layer=(40, 1),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role=part_role,
        attached_face_metal_semantic_id="M1_FACE",
        net_id="M1@signal",
        equipotential_id="EQ_SIGNAL",
    )

    record = xao_adapter._sgb_layer_record(
        route=route,
        semantic_id="M1",
        layer_name="M1",
        layer=layer,
        host_void_semantic_id="AIR_ABOVE",
        selector=None,
        is_residual=True,
    )

    assert record["route_representations"]["A"] == expected_a
    assert record["route_representations"]["B"] == "cutout_boundary_shell"
    assert set(record["route_representations"]) == {"A", "B"}
    assert record["geometry"]["route_ab_fused_selector_mode"] is True
    assert record["geometry"]["split_polygons_as_entities"] is True
    assert record["metadata"] == {
        "source_layer_name": "M1",
        "semantic_group_id": "M1",
        "equipotential_id": "EQ_SIGNAL",
    }


def test_route_ab_selector_overrides_only_selected_island_default_net():
    """A selected M1 island is signal while residual and attached UBM stay Ground."""
    m1 = Layer(
        name="M1",
        gds_layer=(40, 0),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role="face_metal",
        net_id="GROUND",
        equipotential_id="GROUND",
    )
    ubm = Layer(
        name="UBM",
        gds_layer=(40, 1),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role="contact_pad",
        attached_face_metal_semantic_id="M1",
        net_id="GROUND",
        equipotential_id="GROUND",
    )
    bump = Layer(
        name="BUMP",
        gds_layer=(40, 2),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="via",
        part_role="bump_body",
        net_id="GROUND",
        equipotential_id="GROUND",
    )
    selector = SimpleNamespace(
        name="signal", physical_label="signal", center=(0.5, 0.5)
    )

    selected = xao_adapter._sgb_layer_record(
        route="A",
        semantic_id="M1@signal",
        layer_name="M1",
        layer=m1,
        host_void_semantic_id="AIR_ABOVE",
        selector=selector,
        is_residual=False,
    )
    residual = xao_adapter._sgb_layer_record(
        route="A",
        semantic_id="M1",
        layer_name="M1",
        layer=m1,
        host_void_semantic_id="AIR_ABOVE",
        selector=None,
        is_residual=True,
        exclude_selector_points_um=[selector.center],
    )
    attached = xao_adapter._sgb_layer_record(
        route="A",
        semantic_id="UBM",
        layer_name="UBM",
        layer=ubm,
        host_void_semantic_id="AIR_ABOVE",
        selector=None,
        is_residual=True,
    )
    bump_record = xao_adapter._sgb_layer_record(
        route="A",
        semantic_id="BUMP",
        layer_name="BUMP",
        layer=bump,
        host_void_semantic_id="AIR_ABOVE",
        selector=None,
        is_residual=True,
    )

    assert selected["net_id"] == "M1@signal"
    assert selected["equipotential_id"] is None
    assert selected["metadata"]["equipotential_id"] is None
    assert residual["net_id"] == "GROUND"
    assert residual["equipotential_id"] == "GROUND"
    assert residual["metadata"]["equipotential_id"] == "GROUND"
    assert residual["geometry"]["exclude_selector_points_um"] == [[0.5, 0.5]]
    assert residual["geometry"]["split_polygons_as_entities"] is True
    assert residual["geometry"]["route_ab_fused_selector_mode"] is True
    assert attached["net_id"] == residual["net_id"] == "GROUND"
    assert attached["attached_face_metal_semantic_id"] == "M1"
    assert attached["equipotential_id"] == residual["equipotential_id"] == "GROUND"
    assert attached["geometry"]["route_ab_fused_selector_mode"] is True
    assert bump_record["net_id"] == "GROUND"
    assert bump_record["equipotential_id"] == "GROUND"


def test_route_ab_lowering_fused_selector_mode_partitions_fractured_hole(
    monkeypatch, tmp_path
):
    """Ordinary Route-A records make SGB select a fused frame, not one shard."""
    import gdstk
    from semantic_geometry_builder.adapter import build_gds_stack_geometry_input

    m1_gds_layer = 40
    gds_path = tmp_path / "fractured.gds"
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    for lower, upper in (
        ((0, 0), (4, 1)),
        ((0, 3), (4, 4)),
        ((0, 1), (1, 3)),
        ((3, 1), (4, 3)),
        ((10, 0), (11, 1)),
    ):
        cell.add(gdstk.rectangle(lower, upper, layer=m1_gds_layer, datatype=0))
    library.write_gds(gds_path)

    expression = object()
    m1 = Layer(
        name="M1",
        gds_layer=(m1_gds_layer, 0),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role="face_metal",
        net_id="GROUND",
        equipotential_id="GROUND",
    )
    m1._source_expression = expression
    stack = LayerStack(
        layers={
            "D0": Layer(
                name="D0",
                gds_layer=(1, 0),
                zmin=-1.0,
                zmax=0.0,
                thickness=1.0,
                material="silicon",
                layer_type="substrate",
            ),
            "M1": m1,
            "OUTER": Layer(
                name="OUTER",
                gds_layer=(2, 0),
                zmin=0.0,
                zmax=2.0,
                thickness=2.0,
                material="vacuum",
                layer_type="dielectric",
            ),
        }
    )
    logical_m1 = MultiPolygon(
        [
            Polygon(
                [(0, 0), (4, 0), (4, 4), (0, 4)],
                holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
            ),
            box(10, 0, 11, 1),
        ]
    )
    monkeypatch.setattr(
        xao_adapter,
        "fuse_polygons",
        lambda _component, source: (
            logical_m1 if source is expression else pytest.fail("unexpected source")
        ),
    )

    mapping, _, _ = xao_adapter._sgb_stack_mapping_from_gsim_inputs(
        component=SimpleNamespace(bbox_np=lambda: ((0.0, 0.0), (11.0, 4.0))),
        stack=stack,
        gds_path=gds_path,
        activated_regions=(
            ActivatedRegion(layer="D0", role="substrate"),
            ActivatedRegion(layer="OUTER", role="outer_vacuum", z_above=2.0),
        ),
        terminals=(
            TerminalConfig(
                name="signal", layer="M1", physical_label="signal", center=(0.5, 0.5)
            ),
        ),
        route="A",
    )

    records = tuple(
        record for record in mapping["layers"] if record["layer"] == m1_gds_layer
    )
    assert tuple(record["semantic_id"] for record in records) == ("M1@signal", "M1")
    assert all(
        record["geometry"]["route_ab_fused_selector_mode"] is True for record in records
    )
    assert records[1]["geometry"]["split_polygons_as_entities"] is True
    assert all(set(record["route_representations"]) == {"A", "B"} for record in records)
    assert records[0]["net_id"] == "M1@signal"
    assert records[0]["equipotential_id"] is None
    assert records[1]["net_id"] == records[1]["equipotential_id"] == "GROUND"

    stack_path = tmp_path / "fractured.stack.json"
    stack_path.write_text(json.dumps(mapping), encoding="utf-8")
    build_input = build_gds_stack_geometry_input(
        gds_file=gds_path, stack_file=stack_path, top_cell_name="TOP"
    )
    metals = tuple(entity for entity in build_input.entities if entity.role == "metal")
    assert tuple(entity.semantic_id for entity in metals) == ("M1@signal", "M1_0000")
    assert metals[0].net_id == "M1@signal"
    assert metals[0].metadata["equipotential_id"] is None
    assert metals[1].net_id == "GROUND"
    assert metals[1].metadata["equipotential_id"] == "GROUND"
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    selected, residual = (polygons[entity.polygon_ids[0]] for entity in metals)
    assert len(selected.holes) == 1
    assert residual.holes == ()
    assert selected.metadata["source_area_um2"] == pytest.approx(12.0)
    assert residual.metadata["source_area_um2"] == pytest.approx(1.0)


@pytest.mark.parametrize("route", ["A", "B"])
def test_route_ab_lowering_omits_excluded_contact_pad_from_sgb_entities(
    monkeypatch, tmp_path, route
):
    """An excluded typed UBM has no Route-A/B record or SGB entity."""
    import gdstk
    from semantic_geometry_builder.adapter import build_gds_stack_geometry_input

    gds_path = tmp_path / "excluded-contact.gds"
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    for layer in (40, 41, 42):
        cell.add(gdstk.rectangle((0, 0), (1, 1), layer=layer, datatype=0))
    library.write_gds(gds_path)

    expression = object()
    m1 = Layer(
        name="M1",
        gds_layer=(40, 0),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role="face_metal",
        net_id="GROUND",
        equipotential_id="GROUND",
    )
    m1._source_expression = expression
    stack = LayerStack(
        layers={
            "D0": Layer(
                name="D0",
                gds_layer=(1, 0),
                zmin=-1.0,
                zmax=0.0,
                thickness=1.0,
                material="silicon",
                layer_type="substrate",
            ),
            "M1": m1,
            "UBM": Layer(
                name="UBM",
                gds_layer=(41, 0),
                zmin=0.2,
                zmax=0.3,
                thickness=0.1,
                material="aluminum",
                layer_type="conductor",
                part_role="contact_pad",
                attached_face_metal_semantic_id="M1",
                net_id="GROUND",
                equipotential_id="GROUND",
                exclude_from_simulation=True,
            ),
            "BUMP": Layer(
                name="BUMP",
                gds_layer=(42, 0),
                zmin=0.3,
                zmax=1.0,
                thickness=0.7,
                material="indium",
                layer_type="via",
                part_role="bump_body",
                net_id="GROUND",
                equipotential_id="GROUND",
            ),
            "OUTER": Layer(
                name="OUTER",
                gds_layer=(2, 0),
                zmin=0.0,
                zmax=2.0,
                thickness=2.0,
                material="vacuum",
                layer_type="dielectric",
            ),
        }
    )
    monkeypatch.setattr(
        xao_adapter,
        "fuse_polygons",
        lambda _component, source: (
            box(0, 0, 1, 1)
            if source is expression
            else pytest.fail("unexpected source")
        ),
    )

    mapping, semantic_layer_map, _ = xao_adapter._sgb_stack_mapping_from_gsim_inputs(
        component=SimpleNamespace(bbox_np=lambda: ((0.0, 0.0), (1.0, 1.0))),
        stack=stack,
        gds_path=gds_path,
        activated_regions=(
            ActivatedRegion(layer="D0", role="substrate"),
            ActivatedRegion(layer="OUTER", role="outer_vacuum", z_above=2.0),
        ),
        terminals=(
            TerminalConfig(
                name="signal", layer="M1", physical_label="signal", center=(0.5, 0.5)
            ),
        ),
        route=route,
    )

    assert tuple(record["semantic_id"] for record in mapping["layers"]) == (
        "BUMP",
        "M1@signal",
        "M1",
    )
    assert "UBM" not in semantic_layer_map
    assert "UBM" not in json.dumps(mapping)

    stack_path = tmp_path / "excluded-contact.stack.json"
    stack_path.write_text(json.dumps(mapping), encoding="utf-8")
    build_input = build_gds_stack_geometry_input(
        gds_file=gds_path, stack_file=stack_path, top_cell_name="TOP"
    )
    assert all("UBM" not in entity.semantic_id for entity in build_input.entities)
    assert {entity.semantic_id.split("_", 1)[0] for entity in build_input.entities} >= {
        "BUMP",
        "M1@signal",
    }


def test_route_c_mapping_keeps_legacy_geometry_without_fused_selector_mode(tmp_path):
    """An ordinary Route-C mapping cannot receive the A/B fused selector flag."""
    import gdstk

    gds_path = tmp_path / "legacy.gds"
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=40, datatype=0))
    library.write_gds(gds_path)
    stack = LayerStack(
        layers={
            "D0": Layer(
                name="D0",
                gds_layer=(1, 0),
                zmin=-1.0,
                zmax=0.0,
                thickness=1.0,
                material="silicon",
                layer_type="substrate",
            ),
            "M1": Layer(
                name="M1",
                gds_layer=(40, 0),
                zmin=0.0,
                zmax=0.2,
                thickness=0.2,
                material="aluminum",
                layer_type="conductor",
                part_role="face_metal",
                net_id="GROUND",
                equipotential_id="GROUND",
            ),
            "OUTER": Layer(
                name="OUTER",
                gds_layer=(2, 0),
                zmin=0.0,
                zmax=2.0,
                thickness=2.0,
                material="vacuum",
                layer_type="dielectric",
            ),
        }
    )

    mapping, _, _ = xao_adapter._sgb_stack_mapping_from_gsim_inputs(
        component=SimpleNamespace(bbox_np=lambda: ((0.0, 0.0), (1.0, 1.0))),
        stack=stack,
        gds_path=gds_path,
        activated_regions=(
            ActivatedRegion(layer="D0", role="substrate"),
            ActivatedRegion(layer="OUTER", role="outer_vacuum", z_above=2.0),
        ),
        terminals=(TerminalConfig(name="signal", layer="M1", center=(0.5, 0.5)),),
        route="C",
    )

    record = mapping["layers"][0]
    assert record["semantic_id"] == "M1@signal"
    assert set(record["route_representations"]) == {"A", "B", "C"}
    assert record["route_representations"]["C"] == "material_volume"
    assert "route_ab_fused_selector_mode" not in record["geometry"]


def test_route_ab_evaluated_gds_replaces_raw_target_with_fused_islands(
    monkeypatch, tmp_path
):
    """Typed Route A/B lowering writes fused source geometry, not raw target GDS."""
    import gdstk

    gds_path = tmp_path / "input.gds"
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((10, 10), (11, 11), layer=40, datatype=1))
    cell.add(gdstk.rectangle((20, 20), (21, 21), layer=9, datatype=0))
    library.write_gds(gds_path)

    expression = object()
    layer = Layer(
        name="M1",
        gds_layer=(40, 1),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role="face_metal",
    )
    layer._source_expression = expression
    stack = LayerStack(layers={"M1": layer})
    monkeypatch.setattr(
        xao_adapter,
        "fuse_polygons",
        lambda _component, source: (
            MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])
            if source is expression
            else pytest.fail("unexpected source expression")
        ),
    )

    xao_adapter._write_sgb_input_gds(
        component=SimpleNamespace(name="TOP"), stack=stack, gds_path=gds_path
    )

    flattened = next(
        cell for cell in gdstk.read_gds(gds_path).cells if isinstance(cell, gdstk.Cell)
    )
    target = [
        polygon
        for polygon in flattened.polygons
        if (polygon.layer, polygon.datatype) == (40, 1)
    ]
    assert len(target) == 2
    assert all(polygon.bounding_box()[0][0] < 5 for polygon in target)
    assert any(
        (polygon.layer, polygon.datatype) == (9, 0) for polygon in flattened.polygons
    )


def test_route_ab_evaluated_gds_preserves_hole_island_layer_and_datatype(
    monkeypatch, tmp_path
):
    """Logical M1 holes lower through GDS booleans without raw coordinate operands."""
    import gdstk

    gds_path = tmp_path / "input.gds"
    library = gdstk.Library()
    library.new_cell("TOP")
    library.write_gds(gds_path)

    expression = object()
    layer = Layer(
        name="M1",
        gds_layer=(40, 1),
        zmin=0.0,
        zmax=0.2,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
        part_role="face_metal",
    )
    layer._source_expression = expression
    stack = LayerStack(layers={"M1": layer})
    fused_island = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    monkeypatch.setattr(
        xao_adapter,
        "fuse_polygons",
        lambda _component, source: (
            fused_island if source is expression else pytest.fail("unexpected source")
        ),
    )

    xao_adapter._write_sgb_input_gds(
        component=SimpleNamespace(name="TOP"), stack=stack, gds_path=gds_path
    )
    cell = next(
        cell for cell in gdstk.read_gds(gds_path).cells if isinstance(cell, gdstk.Cell)
    )
    polygons = cell.polygons

    assert polygons
    assert {(polygon.layer, polygon.datatype) for polygon in polygons} == {(40, 1)}
    assert sum(polygon.area() for polygon in polygons) == pytest.approx(12.0)
    assert [tuple(map(tuple, polygon.points)) for polygon in polygons] == [
        tuple(map(tuple, polygon.points))
        for polygon in xao_adapter._gdstk_polygons_from_island(
            island=fused_island, layer=40, datatype=1
        )
    ]


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
        route_context="C",
    )
    info = groups["boundary_surfaces"]["SA__D1__AIR_ABOVE__0000"]

    assert info["face_kind"] == "top"
    assert info["source_id"] == "D1__AIR_ABOVE"
    assert info["surface_id"] == "SURF__SA__D1__AIR_ABOVE__0000"
    assert info["owner_semantic_ids"] == ("D1", "AIR_ABOVE")
    assert {"layer", "metal_body_id", "metal_volume_id"}.isdisjoint(info)
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
            route_context="C",
        )
