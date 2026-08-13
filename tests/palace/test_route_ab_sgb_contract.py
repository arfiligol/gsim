"""End-to-end stabilization for the optional SGB Route A/B Palace contract."""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import meshio
import pytest
from gdsfactory.technology import LogicalLayer

from gsim.common.stack import Layer, LayerStack
from gsim.palace import ElectrostaticSim, resolve_palace_result

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
            "M1": Layer(
                name="M1",
                gds_layer=(1, 0),
                zmin=0,
                zmax=0.1,
                thickness=0.1,
                material="Al",
                layer_type="conductor",
                part_role="face_metal",
            ),
            "GROUND": Layer(
                name="GROUND",
                gds_layer=(2, 0),
                zmin=0,
                zmax=0.1,
                thickness=0.1,
                material="Al",
                layer_type="conductor",
                part_role="face_metal",
                net_id="Ground",
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
            "Si": {"permittivity": 11.45, "loss_tangent": 2.6e-7},
            "Al": {"conductivity": 3.8e7},
            "vacuum": {"permittivity": 1.0},
        },
    )
    stack.layers["M1"]._source_expression = LogicalLayer(layer=(1, 0))
    stack.layers["GROUND"]._source_expression = LogicalLayer(layer=(2, 0))
    return stack


def _component(route: str) -> gf.Component:
    gf.gpdk.PDK.activate()
    component = gf.Component(f"route_ab_sgb_contract_{route.lower()}")
    for xmin, xmax in ((0, 2), (4, 6)):
        component.add_polygon(
            ((xmin, 0), (xmax, 0), (xmax, 2), (xmin, 2)), layer=(1, 0)
        )
    for xmin, xmax in ((8, 10), (12, 14)):
        component.add_polygon(
            ((xmin, 0), (xmax, 0), (xmax, 2), (xmin, 2)), layer=(2, 0)
        )
    return component


def _surface_epr_interfaces() -> dict[str, dict[str, object]]:
    return {
        interface: {
            "preset_name": interface,
            "preset": {
                "interface_type": interface,
                "thickness": 0.003,
                "permittivity": 10.0,
                "loss_tangent": 1e-3,
            },
            "role": "boundary_surface",
        }
        for interface in ("MA", "MS", "SA")
    }


def _attributes(entries: list[dict[str, object]]) -> set[int]:
    return {
        int(attribute)
        for entry in entries
        for attribute in entry.get("attributes", [])
    }


@pytest.mark.parametrize("route", ("A", "B"))
def test_route_ab_sgb_mesh_manifest_config_and_handoff(
    route: str, tmp_path: Path
) -> None:
    """Freeze the accepted structured geometry-to-handoff contract."""
    output_dir = tmp_path / f"route_{route.lower()}"
    sim = ElectrostaticSim()
    sim.set_geometry(_component(route))
    sim.set_stack(_stack())
    sim.activate_substrate("SUBSTRATE", margin_x=2, margin_y=2)
    sim.activate_outer_vacuum(
        "OUTER_VACUUM", margin_x=2, margin_y=2, z_above=2, z_below=2
    )
    sim.add_terminal("Qubit_Pad1", layer="M1", center=(1, 1))
    sim.add_terminal("Qubit_Pad2", layer="M1", center=(5, 1))
    sim.add_terminal("GND", net_id="Ground")
    sim.set_electrostatic(
        unassigned_conductor_policy="error", exterior_boundary_policy="ground"
    )
    sim.set_surface_epr(
        representation=route, interfaces=_surface_epr_interfaces()
    )
    sim.set_output_dir(output_dir)

    result = sim.mesh(
        model_name=f"route_{route.lower()}",
        refined_mesh_size=1,
        max_mesh_size=2,
        verbose=False,
    )
    for palace_version in ("0.16.0", "0.16.1"):
        config_path = sim.write_config(
            validate_mesh=True,
            palace_version=palace_version,
            validate_schema=True,
        )

    manifest_path = output_dir / "metadata" / "mesh_manifest.json"
    index_path = output_dir / "metadata" / "palace_index_map.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]
    config = json.loads(config_path.read_text())
    index_map = json.loads(index_path.read_text())

    resolved = resolve_palace_result(output_dir, problem_type="Electrostatic")
    assert resolved.run_summary.mesh_manifest["present"] is True
    assert resolved.run_summary.mesh_manifest["entry_count"] == len(entries)

    volume_entries = {
        entry["name"]: entry for entry in entries if entry["role"] == "dielectric_volume"
    }
    assert set(volume_entries) == {"SUBSTRATE", "AIR_ABOVE", "AIR_BELOW"}
    assert all(entry["dimension"] == 3 for entry in volume_entries.values())
    assert all(
        entry["source"] == "semantic_geometry_builder"
        and entry["metadata"]["representation"] == "solution_volume"
        for entry in volume_entries.values()
    )

    mesh = meshio.read(result.mesh_path)
    tetra_attributes = {
        int(attribute)
        for cells, physical in zip(
            mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
        )
        if cells.type.startswith("tetra")
        for attribute in physical
    }
    volume_attributes = _attributes(list(volume_entries.values()))
    assert tetra_attributes == volume_attributes
    assert {
        name for name, field in mesh.field_data.items() if int(field[1]) == 3
    } == set(volume_entries)

    conductor_entries = [
        entry
        for entry in entries
        if entry["role"] == "boundary_surface"
        and entry["metadata"].get("conductor_component_id")
    ]
    components = {
        entry["metadata"]["conductor_component_id"] for entry in conductor_entries
    }
    assert len(components) == 4
    assert sum(
        entry["metadata"]["net_id"] == "Ground" for entry in conductor_entries
    ) == (4 if route == "A" else 6)
    expected_faces = (
        {("MA", "top"), ("MS", "bottom")}
        if route == "A"
        else {("MA", "top"), ("MS", "bottom"), ("MA", "sidewall")}
    )
    expected_representation = "surface_sheet" if route == "A" else "cutout_boundary_shell"
    for component_id in components:
        component_entries = [
            entry
            for entry in conductor_entries
            if entry["metadata"]["conductor_component_id"] == component_id
        ]
        assert {
            (entry["metadata"]["interface_type"], entry["metadata"]["face_kind"])
            for entry in component_entries
        } == expected_faces
        for entry in component_entries:
            metadata = entry["metadata"]
            assert metadata["sgb_record"] == "final_physical_group"
            assert metadata["representation"] == route
            assert metadata["sgb_representation"] == expected_representation
            assert metadata["owner_semantic_ids"][0] == component_id.removeprefix(
                "COMP__"
            )
            assert metadata["adjacent_solution_volume_ids"]
            assert {
                source["route"]
                for source in metadata["source_provenance"]["sources"]
            } == {route}

    pec_entries = [entry for entry in entries if entry["role"] == "pec_surface"]
    assert len(pec_entries) == len(components) * (1 if route == "A" else 3)
    assert _attributes(pec_entries) == _attributes(conductor_entries)
    assert all(entry["dimension"] == 2 for entry in pec_entries)
    if route == "A":
        assert all(
            entry["metadata"]["bbox"][5] - entry["metadata"]["bbox"][2]
            <= 2.1e-7
            for entry in pec_entries
        )

    sa_entries = [
        entry
        for entry in entries
        if entry["role"] == "boundary_surface"
        and entry["metadata"].get("interface_type") == "SA"
    ]
    assert len(sa_entries) == 2
    sa_by_air = {
        next(
            owner
            for owner in entry["metadata"]["owner_semantic_ids"]
            if owner.startswith("AIR_")
        ): entry
        for entry in sa_entries
    }
    assert sa_by_air["AIR_ABOVE"]["metadata"]["face_kind"] == "top"
    assert sa_by_air["AIR_BELOW"]["metadata"]["face_kind"] == "bottom"
    assert all(
        tuple(entry["interface_of"])
        == tuple(entry["metadata"]["adjacent_solution_volume_ids"])
        and {
            source["route"]
            for source in entry["metadata"]["source_provenance"]["sources"]
        }
        == {route}
        for entry in sa_entries
    )

    terminal_rows = config["Boundaries"]["Terminal"]
    assert len(terminal_rows) == 3
    terminal_attributes = [
        int(attribute)
        for row in terminal_rows
        for attribute in row["Attributes"]
    ]
    assert len(terminal_attributes) == len(set(terminal_attributes))
    assert set(terminal_attributes) == _attributes(conductor_entries)

    terminal_index = [
        entry
        for entry in index_map["entries"]
        if entry["section"] == "Boundaries.Terminal"
    ]
    assert {entry["terminal_name"] for entry in terminal_index} == {
        "Qubit_Pad1",
        "Qubit_Pad2",
        "GND",
    }
    ground_index = [
        entry for entry in terminal_index if entry["terminal_name"] == "GND"
    ]
    assert {
        entry["metadata"]["conductor_component_id"] for entry in ground_index
    } == {"COMP__GROUND_0000", "COMP__GROUND_0001"}
    assert {entry["metadata"]["net_id"] for entry in ground_index} == {"Ground"}
    gnd_terminal = next(row for row in terminal_rows if row["Index"] == 3)
    assert set(gnd_terminal["Attributes"]) == _attributes(ground_index)

    exterior_entries = [
        entry
        for entry in entries
        if entry["role"] == "boundary_surface"
        and entry["source"] == "domain_boundary"
    ]
    exterior_attributes = _attributes(exterior_entries)
    assert set(config["Boundaries"]["Ground"]["Attributes"]) == exterior_attributes
    assert exterior_attributes.isdisjoint(terminal_attributes)
    assert "LumpedPort" not in config["Boundaries"]
    assert "WavePort" not in config["Boundaries"]

    dielectric = config["Boundaries"]["Postprocessing"]["Dielectric"]
    assert {entry["Type"] for entry in dielectric} == {"MA", "MS", "SA"}
    assert all(entry["Attributes"] for entry in dielectric)
    assert {
        entry["metadata"]["representation"]
        for entry in index_map["entries"]
        if entry["section"] == "Boundaries.Postprocessing.Dielectric"
    } == {route}

    handle = sim.generate_handoff_package(
        write_config=False,
        include_hashes=True,
        metadata={"route": route},
        archive_path=tmp_path / f"route_{route.lower()}.tar.gz",
    )
    archive_manifest = json.loads(handle.archive_manifest_path.read_text())
    archived_paths = {entry["path"] for entry in archive_manifest["files"]}
    assert handle.archive_path.is_file()
    assert handle.metadata["palace_config_version"] == "0.16.1"
    assert archive_manifest["metadata"]["route"] == route
    assert {
        "config.json",
        result.mesh_path.name,
        "metadata/mesh_manifest.json",
        "metadata/palace_index_map.json",
        f"geometry/semantic_geometry_route_{route.lower()}.xao",
    }.issubset(archived_paths)

    sim.terminals = [
        terminal for terminal in sim.terminals if terminal.name != "Qubit_Pad2"
    ]
    with pytest.raises(ValueError, match="Unassigned SGB conductor components"):
        sim.write_config(validate_mesh=False, palace_version="0.16.1")
