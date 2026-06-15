"""Lightweight tests for Palace mesh manifests and postprocessing config."""

from __future__ import annotations

import json
from pathlib import Path

from gsim.common.stack import LayerStack
from gsim.palace.mesh.config_generator import generate_palace_config, write_config
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.manifest import build_mesh_manifest
from gsim.palace.mesh.postprocessing import (
    DielectricInterfaceSpec,
    SurfaceFluxSpec,
    build_postprocessing_config_from_manifest,
    build_terminal_index_map_from_manifest,
)


def _minimal_groups() -> dict:
    return {
        "volumes": {
            "substrate": {"phys_group": 11, "tags": [101], "dim": 3},
            "via": {"phys_group": 12, "tags": [102], "is_via": True, "dim": 3},
            "core": {
                "phys_group": 13,
                "tags": [103],
                "is_shaped_dielectric": True,
                "dim": 3,
            },
        },
        "conductor_surfaces": {
            "metal_xy": {"phys_group": 21, "tags": [201], "dim": 2},
        },
        "pec_surfaces": {
            "ground": {"phys_group": 22, "tags": [202], "dim": 2},
        },
        "port_surfaces": {
            "P1": {"phys_group": 31, "tags": [301], "type": "lumped", "dim": 2},
            "P2": {
                "type": "cpw",
                "elements": [
                    {"phys_group": 32, "tags": [302], "direction": "X", "dim": 2},
                    {"phys_group": 33, "tags": [303], "direction": "Y", "dim": 2},
                ],
            },
        },
        "boundary_surfaces": {
            "absorbing": {"phys_group": [41, 42], "tags": [401, 402], "dim": 2},
            "metal___substrate": {"phys_group": 43, "tags": [403], "dim": 2},
            "metal___None": {"phys_group": 44, "tags": [404], "dim": 2},
            "legacy__substrate": {"phys_group": 45, "tags": [405], "dim": 2},
        },
        "refinement_lines": {
            "ground_edge": {"phys_group": 51, "tags": [501], "dim": 1},
        },
        "via_boundary_surfaces": {
            "via_boundary": [61, 62],
        },
    }


def test_build_mesh_manifest_classifies_roles_and_preserves_ids() -> None:
    manifest = build_mesh_manifest(_minimal_groups())
    entries = {entry.name: entry for entry in manifest.entries}

    assert entries["substrate"].role == "dielectric_volume"
    assert entries["substrate"].attributes == (11,)
    assert entries["substrate"].entity_tags == (101,)
    assert entries["substrate"].metadata["dim"] == 3

    assert entries["via"].role == "via_volume"
    assert entries["core"].role == "shaped_dielectric_volume"

    assert entries["metal_xy"].role == "conductor_surface"
    assert entries["metal_xy"].attributes == (21,)
    assert entries["metal_xy"].metadata["dim"] == 2

    assert entries["ground"].role == "pec_surface"
    assert entries["P1"].role == "port_surface"
    assert entries["P1"].metadata == {
        "type": "lumped",
        "dim": 2,
        "port": "P1",
        "port_type": "lumped",
    }

    assert entries["P2_E0"].role == "port_surface"
    assert entries["P2_E0"].attributes == (32,)
    assert entries["P2_E0"].metadata["port"] == "P2"
    assert entries["P2_E0"].metadata["element_index"] == 0
    assert entries["P2_E1"].attributes == (33,)

    assert entries["absorbing"].role == "boundary_surface"
    assert entries["absorbing"].attributes == (41, 42)
    assert entries["metal___substrate"].physical_names == ("metal___substrate",)
    assert entries["metal___substrate"].interface_of == ("metal", "substrate")
    assert entries["metal___substrate"].dimension == 2
    assert entries["metal___substrate"].source == "gsim_gmsh"
    assert entries["metal___None"].exterior_of == "metal"
    assert entries["legacy__substrate"].interface_of == ("legacy", "substrate")
    assert entries["ground_edge"].role == "refinement_line"
    assert entries["ground_edge"].metadata["dim"] == 1
    assert entries["via_boundary"].role == "via_boundary_surface"
    assert entries["via_boundary"].attributes == (61, 62)

    assert manifest.attributes_for_role("port_surface") == (31, 32, 33)


def test_mesh_result_populates_manifest_without_mutating_groups(tmp_path: Path) -> None:
    groups = _minimal_groups()
    original_port_surfaces = groups["port_surfaces"]

    result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        groups=groups,
        output_dir=tmp_path,
    )

    assert result.groups is groups
    assert result.groups["port_surfaces"] is original_port_surfaces
    assert result.manifest.attributes_for_role("port_surface") == (31, 32, 33)

    manifest_path = result.manifest.write_json(tmp_path / "mesh_manifest.json")
    manifest_json = json.loads(manifest_path.read_text())

    assert manifest_json["schema_version"] == 1
    assert manifest_json["entries"][0]["physical_names"] == ["substrate"]


def test_generate_palace_config_merges_postprocessing_config(tmp_path: Path) -> None:
    groups = {
        "volumes": {"air": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        absorbing_boundary=False,
        postprocessing_config={"SurfaceFlux": [{"Attributes": [22]}]},
    )

    postprocessing = json.loads(config_path.read_text())["Domains"]["Postprocessing"]
    assert postprocessing["Energy"] == []
    assert postprocessing["Probe"] == []
    assert postprocessing["SurfaceFlux"] == [{"Attributes": [22]}]


def test_write_config_merges_postprocessing_config(tmp_path: Path) -> None:
    groups = {
        "volumes": {"air": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        groups=groups,
        output_dir=tmp_path,
        model_name="palace",
        fmax=10e9,
    )

    config_path = write_config(
        mesh_result=mesh_result,
        stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
        ports=[],
        absorbing_boundary=False,
        postprocessing_config={"SurfaceFlux": [{"Attributes": [22]}]},
    )

    postprocessing = json.loads(config_path.read_text())["Domains"]["Postprocessing"]
    assert mesh_result.config_path == config_path
    assert postprocessing["Energy"] == []
    assert postprocessing["Probe"] == []
    assert postprocessing["SurfaceFlux"] == [{"Attributes": [22]}]


def test_write_config_applies_material_overlay_without_mutating_stack(
    tmp_path: Path,
) -> None:
    groups = {
        "volumes": {"Si": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    stack = LayerStack(materials={"Si": {"permittivity": 11.9, "conductivity": 2.0}})
    mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        groups=groups,
        output_dir=tmp_path,
        model_name="palace",
        fmax=10e9,
    )

    config_path = write_config(
        mesh_result=mesh_result,
        stack=stack,
        ports=[],
        absorbing_boundary=False,
        material_overlay={
            "materials": {
                "Si": {
                    "relative_permittivity": 11.45,
                    "dispersion_models": [
                        {
                            "type": "constant",
                            "permittivity": 11.45,
                            "validity_frequency": [0, 20e9],
                            "source": "test PDK",
                        }
                    ],
                }
            }
        },
    )

    material = json.loads(config_path.read_text())["Domains"]["Materials"][0]
    assert material["Permittivity"] == 11.45
    assert material["Conductivity"] == 2.0
    assert stack.materials["Si"]["permittivity"] == 11.9
    assert stack.materials["Si"]["conductivity"] == 2.0

    material_resolution = json.loads(
        (tmp_path / "palace_material_resolution.json").read_text()
    )
    row = material_resolution["materials"][0]
    assert row["material_attribute"] == 1
    assert row["volume_name"] == "Si"
    assert row["stack_material_name"] == "Si"
    assert row["matched_material_name"] == "Si"
    assert row["palace_material"]["Permittivity"] == 11.45
    assert row["model_type"] == "constant"
    assert row["model_source"] == "test PDK"


def test_write_config_resolves_dielectric_interface_material_overlay(
    tmp_path: Path,
) -> None:
    groups = {
        "volumes": {"Si": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {"sa_boundary": {"phys_group": 70}},
    }
    stack = LayerStack(materials={"Si": {"permittivity": 11.45}})
    postprocessing = {
        "Dielectric": [
            {
                "Index": 7,
                "Attributes": [70],
                "Type": "SA",
                "Thickness": 0.003,
                "_MaterialName": "AlOx_native_generic",
            }
        ]
    }

    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="palace",
        fmax=5e9,
        absorbing_boundary=False,
        boundary_postprocessing_config=postprocessing,
        material_overlay={
            "materials": {
                "AlOx_native_generic": {
                    "relative_permittivity": 10.0,
                    "loss_tangent": 0.0017,
                    "dispersion_models": [
                        {
                            "type": "constant",
                            "permittivity": 10.0,
                            "validity_frequency": [0, 10e9],
                            "source": "test PDK interface material",
                        }
                    ],
                }
            }
        },
    )

    interface = json.loads(config_path.read_text())["Boundaries"]["Postprocessing"][
        "Dielectric"
    ][0]
    assert "_MaterialName" not in interface
    assert interface["Permittivity"] == 10.0
    assert interface["LossTan"] == 0.0017

    material_resolution = json.loads(
        (tmp_path / "palace_material_resolution.json").read_text()
    )
    row = material_resolution["interfaces"][0]
    assert row["surface_index"] == 7
    assert row["surface_attributes"] == [70]
    assert row["interface_type"] == "SA"
    assert row["interface_material_name"] == "AlOx_native_generic"
    assert row["matched_material_name"] == "AlOx_native_generic"
    assert row["palace_interface"]["Permittivity"] == 10.0
    assert row["palace_interface"]["LossTan"] == 0.0017
    assert row["model_type"] == "constant"
    assert row["model_source"] == "test PDK interface material"


def test_build_postprocessing_config_from_manifest_has_stable_indices(
    tmp_path: Path,
) -> None:
    manifest = build_mesh_manifest(_minimal_groups())

    config = build_postprocessing_config_from_manifest(
        manifest,
        surface_flux=(
            SurfaceFluxSpec(
                role="conductor_surface",
                flux_type="Electric",
                two_sided=True,
            ),
            SurfaceFluxSpec(
                role="boundary_surface",
                entry_names=("absorbing",),
                flux_type="Power",
                two_sided=None,
            ),
        ),
        dielectric_interfaces=(
            DielectricInterfaceSpec(
                role="boundary_surface",
                entry_names=("absorbing",),
                interface_type="SA",
                thickness=0.002,
                permittivity=10.0,
                loss_tangent=1.0,
            ),
        ),
    )

    assert config.domains["Energy"] == [
        {"Index": 1, "Attributes": [11]},
        {"Index": 2, "Attributes": [13]},
    ]
    assert config.domains["Probe"] == []
    assert config.boundaries["SurfaceFlux"] == [
        {
            "Index": 1,
            "Attributes": [21],
            "Type": "Electric",
            "TwoSided": True,
        },
        {"Index": 2, "Attributes": [41, 42], "Type": "Power"},
    ]
    assert config.boundaries["Dielectric"] == [
        {
            "Index": 1,
            "Attributes": [41, 42],
            "Type": "SA",
            "Thickness": 0.002,
            "Permittivity": 10.0,
            "LossTan": 1.0,
        }
    ]
    assert config.index_map.to_rows() == (
        {
            "section": "Domains.Postprocessing.Energy",
            "index": 1,
            "entry_name": "substrate",
            "role": "dielectric_volume",
            "attributes": [11],
            "physical_names": ["substrate"],
            "entity_tags": [101],
            "dimension": 3,
            "source": "gsim_gmsh",
            "metadata": {"dim": 3},
        },
        {
            "section": "Domains.Postprocessing.Energy",
            "index": 2,
            "entry_name": "core",
            "role": "shaped_dielectric_volume",
            "attributes": [13],
            "physical_names": ["core"],
            "entity_tags": [103],
            "dimension": 3,
            "source": "gsim_gmsh",
            "metadata": {"is_shaped_dielectric": True, "dim": 3},
        },
        {
            "section": "Boundaries.Postprocessing.SurfaceFlux",
            "index": 1,
            "entry_name": "metal_xy",
            "role": "conductor_surface",
            "attributes": [21],
            "physical_names": ["metal_xy"],
            "entity_tags": [201],
            "dimension": 2,
            "source": "gsim_gmsh",
            "metadata": {"dim": 2},
            "Type": "Electric",
        },
        {
            "section": "Boundaries.Postprocessing.SurfaceFlux",
            "index": 2,
            "entry_name": "absorbing",
            "role": "boundary_surface",
            "attributes": [41, 42],
            "physical_names": ["absorbing"],
            "entity_tags": [401, 402],
            "dimension": 2,
            "source": "gsim_gmsh",
            "metadata": {"dim": 2},
            "Type": "Power",
        },
        {
            "section": "Boundaries.Postprocessing.Dielectric",
            "index": 1,
            "entry_name": "absorbing",
            "role": "boundary_surface",
            "attributes": [41, 42],
            "physical_names": ["absorbing"],
            "entity_tags": [401, 402],
            "dimension": 2,
            "source": "gsim_gmsh",
            "metadata": {"dim": 2},
            "Type": "SA",
        },
    )
    assert config.index_map.to_dict()["schema_version"] == 1

    index_map_path = config.index_map.write_json(tmp_path / "palace_index_map.json")
    index_map_json = json.loads(index_map_path.read_text())

    assert index_map_json["schema_version"] == 1
    assert index_map_json["entries"][0]["entry_name"] == "substrate"


def test_build_postprocessing_config_supports_interface_material_reference() -> None:
    manifest = build_mesh_manifest(_minimal_groups())

    config = build_postprocessing_config_from_manifest(
        manifest,
        dielectric_interfaces=(
            DielectricInterfaceSpec(
                role="boundary_surface",
                entry_names=("absorbing",),
                interface_type="SA",
                thickness=0.003,
                material_name="AlOx_native_generic",
            ),
        ),
    )

    assert config.boundaries["Dielectric"] == [
        {
            "Index": 1,
            "Attributes": [41, 42],
            "Type": "SA",
            "Thickness": 0.003,
            "LossTan": 0.0,
            "_MaterialName": "AlOx_native_generic",
        }
    ]


def test_postprocessing_index_map_supports_bidirectional_lookup() -> None:
    manifest = build_mesh_manifest(_minimal_groups())

    config = build_postprocessing_config_from_manifest(
        manifest,
        surface_flux=(
            SurfaceFluxSpec(
                role="conductor_surface",
                flux_type="Electric",
            ),
            SurfaceFluxSpec(
                role="boundary_surface",
                entry_names=("absorbing",),
                flux_type="Power",
            ),
        ),
    )

    index_map = config.index_map

    assert (
        index_map.physical_name_for_index("Boundaries.Postprocessing.SurfaceFlux", 1)
        == "metal_xy"
    )
    assert index_map.physical_name_for_index("missing", 1) is None
    assert index_map.indices_for_name("absorbing") == (2,)
    assert index_map.indices_for_physical_name("absorbing") == (2,)
    assert index_map.indices_for_name(
        "absorbing", section="Boundaries.Postprocessing.SurfaceFlux"
    ) == (2,)
    assert [
        entry.entry_name
        for entry in index_map.entries_for_attribute(
            41, section="Boundaries.Postprocessing.SurfaceFlux"
        )
    ] == ["absorbing"]
    assert index_map.entry_for_index(
        "Boundaries.Postprocessing.SurfaceFlux", 2
    ).attributes == (41, 42)


def test_build_terminal_index_map_from_manifest_links_boundary_indices() -> None:
    manifest = build_mesh_manifest(_minimal_groups())

    index_map = build_terminal_index_map_from_manifest(
        manifest,
        (
            {"Index": 1, "Attributes": [21]},
            {"Index": 2, "Attributes": [22, 61]},
        ),
        terminal_names=("signal", "ground"),
    )

    assert index_map.to_rows() == (
        {
            "section": "Boundaries.Terminal",
            "index": 1,
            "entry_name": "metal_xy",
            "role": "conductor_surface",
            "attributes": [21],
            "physical_names": ["metal_xy"],
            "entity_tags": [201],
            "dimension": 2,
            "source": "gsim_gmsh",
            "metadata": {"dim": 2},
            "terminal_name": "signal",
            "terminal_attributes": [21],
        },
        {
            "section": "Boundaries.Terminal",
            "index": 2,
            "entry_name": "ground",
            "role": "pec_surface",
            "attributes": [22],
            "physical_names": ["ground"],
            "entity_tags": [202],
            "dimension": 2,
            "source": "gsim_gmsh",
            "metadata": {"dim": 2},
            "terminal_name": "ground",
            "terminal_attributes": [22, 61],
        },
        {
            "section": "Boundaries.Terminal",
            "index": 2,
            "entry_name": "via_boundary",
            "role": "via_boundary_surface",
            "attributes": [61, 62],
            "physical_names": ["via_boundary"],
            "dimension": 2,
            "source": "gsim_gmsh",
            "terminal_name": "ground",
            "terminal_attributes": [22, 61],
        },
    )
    assert index_map.physical_name_for_index("Boundaries.Terminal", 1) == "metal_xy"
    assert index_map.indices_for_physical_name("via_boundary") == (2,)


def test_generate_palace_config_merges_boundary_postprocessing(tmp_path: Path) -> None:
    groups = {
        "volumes": {"air": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        absorbing_boundary=False,
        boundary_postprocessing_config={
            "SurfaceFlux": [{"Index": 1, "Attributes": [21], "Type": "Electric"}],
            "Dielectric": [
                {
                    "Index": 1,
                    "Attributes": [41],
                    "Type": "SA",
                    "Thickness": 0.002,
                    "Permittivity": 10.0,
                    "LossTan": 1.0,
                }
            ],
        },
    )

    postprocessing = json.loads(config_path.read_text())["Boundaries"]["Postprocessing"]
    assert postprocessing["SurfaceFlux"] == [
        {"Index": 1, "Attributes": [21], "Type": "Electric"}
    ]
    assert postprocessing["Dielectric"][0]["Type"] == "SA"


def test_write_config_merges_boundary_postprocessing(tmp_path: Path) -> None:
    groups = {
        "volumes": {"air": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        groups=groups,
        output_dir=tmp_path,
        model_name="palace",
        fmax=10e9,
    )

    config_path = write_config(
        mesh_result=mesh_result,
        stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
        ports=[],
        absorbing_boundary=False,
        boundary_postprocessing_config={
            "SurfaceFlux": [{"Index": 1, "Attributes": [21], "Type": "Electric"}],
        },
    )

    postprocessing = json.loads(config_path.read_text())["Boundaries"]["Postprocessing"]
    assert mesh_result.config_path == config_path
    assert postprocessing["SurfaceFlux"] == [
        {"Index": 1, "Attributes": [21], "Type": "Electric"}
    ]
