"""Route A/B layout-sheet lumped-port integration fixture."""

from __future__ import annotations

import json
from math import cos, pi, sin, sqrt
from pathlib import Path

import gdsfactory as gf
import meshio
import pytest
from gdsfactory.technology import LogicalLayer

from gsim.common.stack import Layer, LayerStack
from gsim.palace import EigenmodeSim
from gsim.palace.resolve.derived.participation import load_port_epr_summary

pytest.importorskip("semantic_geometry_builder")


def _stack() -> LayerStack:
    stack = LayerStack(
        layers={
            "SUBSTRATE": Layer(
                name="SUBSTRATE",
                gds_layer=(10, 0),
                zmin=-2,
                zmax=0,
                thickness=2,
                material="Si",
                layer_type="substrate",
            ),
            "D1_BOTTOM_M1": Layer(
                name="D1_BOTTOM_M1",
                gds_layer=(1, 0),
                zmin=0,
                zmax=0.1,
                thickness=0.1,
                material="Al",
                layer_type="conductor",
                part_role="face_metal",
            ),
            "OUTER_VACUUM": Layer(
                name="OUTER_VACUUM",
                gds_layer=(20, 0),
                zmin=0,
                zmax=0,
                thickness=0,
                material="vacuum",
                layer_type="dielectric",
            ),
        },
        materials={
            "Si": {"permittivity": 11.45},
            "Al": {"conductivity": 3.8e7},
            "vacuum": {"permittivity": 1.0},
        },
    )
    stack.layers["D1_BOTTOM_M1"]._source_expression = LogicalLayer(layer=(1, 0))
    return stack


def _rounded_arm(*, left: bool) -> tuple[tuple[float, float], ...]:
    radius = 1.0
    if left:
        center = -1.5
        points = [(-5.0, -radius), (center, -radius)]
        points.extend(
            (center + radius * cos(angle), radius * sin(angle))
            for angle in (pi * value / 12 for value in range(-6, 7))
        )
        points.append((-5.0, radius))
    else:
        center = 1.5
        points = [(5.0, radius), (center, radius)]
        points.extend(
            (center + radius * cos(angle), radius * sin(angle))
            for angle in (pi * value / 12 for value in range(6, 19))
        )
        points.append((5.0, -radius))
    scale = sqrt(0.5)
    return tuple(((x - y) * scale, (x + y) * scale) for x, y in points)


def _rotate_sheet(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    scale = sqrt(0.5)
    return tuple(((x - y) * scale, (x + y) * scale) for x, y in points)


def _surface_nodes(mesh: meshio.Mesh, attributes: set[int]) -> set[int]:
    return {
        int(node)
        for cells, physical in zip(
            mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
        )
        if cells.type.startswith("triangle")
        for element, attribute in zip(cells.data, physical, strict=True)
        if int(attribute) in attributes
        for node in element
    }


def _structured_owner_pec_attributes(
    manifest: dict[str, object],
    port_metadata: dict[str, object],
) -> dict[str, set[int]]:
    """Map each SGB-declared port owner to its Route B PEC attributes."""
    entries = manifest["entries"]
    assert isinstance(entries, list)
    attribute = port_metadata["physical_attribute"]
    assert isinstance(attribute, dict)
    owner_provenance = attribute["owner_provenance"]
    assert isinstance(owner_provenance, list)
    owner_components = {
        owner["semantic_id"]: owner["conductor_component_id"]
        for owner in owner_provenance
    }
    assert len(owner_components) == 2
    pec_attributes = {
        int(value)
        for entry in entries
        if entry["role"] == "pec_surface"
        for value in entry["attributes"]
    }
    attrs_by_owner = {owner_id: set() for owner_id in owner_components}
    for entry in entries:
        if entry["role"] != "boundary_surface":
            continue
        metadata = entry["metadata"]
        if (
            metadata["sgb_record"] != "final_physical_group"
            or metadata["representation"] != "B"
        ):
            continue
        component_id = metadata["conductor_component_id"]
        for owner_id, expected_component_id in owner_components.items():
            if component_id != expected_component_id:
                continue
            assert owner_id in metadata["owner_semantic_ids"]
            attrs_by_owner[owner_id].update(
                int(value) for value in entry["attributes"] if value in pec_attributes
            )
    assert all(attrs_by_owner.values())
    return attrs_by_owner


def _component() -> gf.Component:
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(_rounded_arm(left=True), layer=(1, 0))
    component.add_polygon(_rounded_arm(left=False), layer=(1, 0))
    component.add_polygon(((20, -1), (22, -1), (22, 1), (20, 1)), layer=(202, 1))
    component.add_polygon(
        _rotate_sheet(((-2.25, -0.5), (2.25, -0.5), (2.25, 0.5), (-2.25, 0.5))),
        layer=(202, 2),
    )
    component.add_port(
        "o_junction_lumped",
        center=(0, 0),
        width=2,
        orientation=225,
        layer=(202, 2),
    )
    return component


@pytest.mark.parametrize("route", ["A", "B"])
def test_route_ab_lumped_port_sheet_uses_gf_port_layer_only(
    route: str, tmp_path: Path
) -> None:
    output_dir = tmp_path / route
    sim = EigenmodeSim()
    sim.set_geometry(_component())
    sim.set_stack(_stack())
    sim.activate_substrate("SUBSTRATE", margin_x=3, margin_y=3)
    sim.activate_outer_vacuum(
        "OUTER_VACUUM", margin_x=3, margin_y=3, z_above=3, z_below=3
    )
    sim.add_port(
        "o_junction_lumped",
        layer="D1_BOTTOM_M1",
        layout_sheet=True,
        inductance=2.15e-8,
    )
    sim.set_eigenmode(target=5e9)
    sim.set_surface_epr(representation=route, interfaces={})
    sim.set_output_dir(output_dir)

    result = sim.mesh(
        model_name=f"lumped_{route.lower()}",
        refined_mesh_size=1,
        max_mesh_size=2,
        verbose=False,
    )
    config_path = sim.write_config(
        validate_mesh=True,
        palace_version="0.16.1",
        validate_schema=True,
    )

    stack_input = json.loads(
        (output_dir / "geometry" / "semantic_geometry_input.stack.json").read_text()
    )
    source_layers = stack_input["metadata"]["port_sheet_source_layers"]
    assert len(source_layers) == 1
    assert source_layers[0]["layer"] == 202
    assert source_layers[0]["datatype"] == 2
    assert source_layers[0]["direction"] == pytest.approx([-sqrt(0.5), -sqrt(0.5), 0.0])

    manifest = json.loads((output_dir / "metadata" / "mesh_manifest.json").read_text())
    port_entries = [
        entry for entry in manifest["entries"] if entry["role"] == "port_surface"
    ]
    assert len(port_entries) == 1
    if route == "A":
        assert port_entries[0]["interface_of"] == ["SUBSTRATE", "AIR_ABOVE"]
    else:
        assert "interface_of" not in port_entries[0]
    metadata = port_entries[0]["metadata"]
    assert metadata["sgb_record"] == "final_physical_group"
    assert metadata["physical_attribute"]["port_index"] == 1
    assert metadata["physical_attribute"]["source_layer"] == "202/2"
    assert metadata["physical_attribute"]["target_layer"] == "D1_BOTTOM_M1"
    assert metadata["physical_attribute"]["direction"] == pytest.approx(
        [-sqrt(0.5), -sqrt(0.5), 0.0]
    )
    assert all(
        owner["conductor_component_id"]
        for owner in metadata["physical_attribute"]["owner_provenance"]
    )
    assert metadata["source_provenance"]["direction_sign_convention"] == (
        "gdsfactory_port_orientation_outward"
    )

    config = json.loads(config_path.read_text())
    [port] = config["Boundaries"]["LumpedPort"]
    assert port["Direction"] == pytest.approx([-sqrt(0.5), -sqrt(0.5), 0.0])
    assert port["L"] == 2.15e-8
    assert port["Excitation"] is False
    assert set(port) == {"Index", "Direction", "Excitation", "Attributes", "L"}
    port_information = json.loads(
        (output_dir / "metadata" / "port_information.json").read_text()
    )
    assert port_information["ports"] == [
        {
            "portnumber": 1,
            "name": "o_junction_lumped",
            "type": "lumped_sheet",
            "direction": pytest.approx([-sqrt(0.5), -sqrt(0.5), 0.0]),
            "physical_name": "LUMPED_PORT__o_junction_lumped",
            "attributes": port["Attributes"],
            "source_layer": "202/2",
            "target_layer": "D1_BOTTOM_M1",
        }
    ]
    index_map = json.loads(
        (output_dir / "metadata" / "palace_index_map.json").read_text()
    )
    [port_index] = [
        entry
        for entry in index_map["entries"]
        if entry["section"] == "Boundaries.LumpedPort"
    ]
    assert port_index["index"] == 1
    assert port_index["entry_name"] == "P1"
    assert port_index["port_name"] == "o_junction_lumped"
    assert port_index["attributes"] == port["Attributes"]
    (output_dir / "results" / "palace" / "port-EPR.csv").write_text("m,p[1]\n1,0.25\n")
    [port_epr] = load_port_epr_summary(output_dir).to_dict(orient="records")
    assert port_epr["port_index"] == 1
    assert port_epr["port_name"] == "o_junction_lumped"
    assert port_epr["source_name"] == "o_junction_lumped"
    assert port_epr["role"] == "port_surface"
    assert port_epr["attributes"] == tuple(port["Attributes"])
    assert port_epr["metadata"]["owner_semantic_ids"] == metadata["owner_semantic_ids"]

    mesh = meshio.read(result.mesh_path)
    port_nodes = _surface_nodes(mesh, {int(port["Attributes"][0])})
    if route == "B":
        for owner_id, owner_attributes in _structured_owner_pec_attributes(
            manifest, metadata
        ).items():
            assert port_nodes & _surface_nodes(mesh, owner_attributes), owner_id
    else:
        pec_nodes = _surface_nodes(
            mesh, {int(value) for value in config["Boundaries"]["PEC"]["Attributes"]}
        )
        assert len(port_nodes & pec_nodes) >= 2
    tetra_attributes = {
        int(attribute)
        for cells, physical in zip(
            mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
        )
        if cells.type.startswith("tetra")
        for attribute in physical
    }
    volume_attributes = {
        attribute
        for entry in manifest["entries"]
        if entry["role"] == "dielectric_volume"
        for attribute in entry["attributes"]
    }
    assert tetra_attributes == volume_attributes


def test_layout_sheet_port_has_no_native_geometry_fallback(tmp_path: Path) -> None:
    sim = EigenmodeSim()
    sim.set_geometry(_component())
    sim.set_stack(_stack())
    sim.activate_substrate("SUBSTRATE", margin_x=3, margin_y=3)
    sim.activate_outer_vacuum(
        "OUTER_VACUUM", margin_x=3, margin_y=3, z_above=3, z_below=3
    )
    sim.add_port(
        "o_junction_lumped",
        layer="D1_BOTTOM_M1",
        layout_sheet=True,
        inductance=2.15e-8,
    )
    sim.set_eigenmode(target=5e9)
    sim.set_output_dir(tmp_path)

    with pytest.raises(ValueError, match="layout_sheet ports require set_surface_epr"):
        sim.mesh(verbose=False)
