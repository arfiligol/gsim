"""Consumer tests for meshwell physical-name handoff artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import gmsh

from gsim.palace.mesh.manifest import build_mesh_manifest
from gsim.palace.mesh.postprocessing import (
    SurfaceFluxSpec,
    build_dielectric_interface_specs_from_material_kinds,
    build_postprocessing_config_from_manifest,
)
from gsim.palace.results import load_postprocessing_index_map

FIXTURE_DIR = Path(__file__).with_name("test_meshwell_handoff_contract")
MSH_FIXTURE = FIXTURE_DIR / "physical_name_contract.msh"
EXPECTED_NAMES = FIXTURE_DIR / "expected_physical_names.json"


def _load_expected_fixture_contract() -> dict:
    """Load the expected meshwell physical-name contract for the fixture."""
    payload = json.loads(EXPECTED_NAMES.read_text())
    assert payload["contract"] == {
        "interface_delimiter": "___",
        "boundary_delimiter": "None",
        "owner": "meshwell",
        "consumer": "gsim",
    }
    return payload


def _meshwell_fixture_physical_groups() -> dict[int, dict[str, dict]]:
    """Read physical groups from the committed meshwell-generated MSH fixture."""
    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()

    try:
        gmsh.open(str(MSH_FIXTURE))
        groups_by_dim: dict[int, dict[str, dict]] = {}
        for dim, physical_group in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, physical_group)
            if not name:
                continue
            entity_tags = [
                int(tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, physical_group)
            ]
            groups_by_dim.setdefault(int(dim), {})[name] = {
                "phys_group": int(physical_group),
                "tags": entity_tags,
                "dim": int(dim),
                "source": "meshwell_msh_fixture",
            }
        gmsh.clear()
        return groups_by_dim
    finally:
        if not was_initialized:
            gmsh.finalize()


def _load_meshwell_handoff_groups() -> dict:
    """Convert a meshwell-generated MSH fixture into gsim group categories."""
    expected = _load_expected_fixture_contract()
    groups_by_dim = _meshwell_fixture_physical_groups()
    expected_by_dim = {
        int(dim): tuple(names)
        for dim, names in expected["physical_groups_by_dimension"].items()
    }

    for dim, expected_names in expected_by_dim.items():
        assert tuple(groups_by_dim.get(dim, {})) == expected_names

    present_names = {
        name for groups_for_dim in groups_by_dim.values() for name in groups_for_dim
    }
    for name in expected["absent_physical_names"]:
        assert name not in present_names

    return {
        "volumes": groups_by_dim[3],
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": groups_by_dim[2],
        "refinement_lines": {},
        "via_boundary_surfaces": {},
    }


def test_meshwell_handoff_groups_build_manifest_relations() -> None:
    """Meshwell-style interface and exterior names become manifest relations."""
    manifest = build_mesh_manifest(_load_meshwell_handoff_groups())
    entries = {entry.name: entry for entry in manifest.entries}

    assert entries["A"].role == "dielectric_volume"
    assert entries["B"].role == "dielectric_volume"
    assert entries["A___B"].interface_of == ("A", "B")
    assert entries["B___helper"].interface_of == ("B", "helper")
    assert entries["A___None"].interface_of is None
    assert entries["A___None"].exterior_of == "A"
    assert entries["B___None"].exterior_of == "B"

    for name in ("A___B", "B___helper", "A___None", "B___None"):
        entry = entries[name]
        assert entry.role == "boundary_surface"
        assert entry.physical_names == (name,)
        assert entry.source == "meshwell_msh_fixture"
        assert entry.dimension == 2


def test_meshwell_handoff_material_kind_specs_build_index_map(tmp_path: Path) -> None:
    """Meshwell-style interfaces remain reportable through gsim index maps."""
    manifest = build_mesh_manifest(_load_meshwell_handoff_groups())
    manifest_entries = {entry.name: entry for entry in manifest.entries}
    specs = build_dielectric_interface_specs_from_material_kinds(
        manifest,
        material_kind_by_name={
            "A": "superconductor",
            "B": "dielectric",
            "helper": "vacuum",
        },
        presets={
            "public_ma": {
                "interface_type": "MA",
                "thickness": 0.001,
                "permittivity": 10.0,
                "source": "meshwell handoff MA fixture",
            },
            "public_ms": {
                "interface_type": "MS",
                "thickness": 0.002,
                "permittivity": 11.45,
                "source": "meshwell handoff MS fixture",
            },
            "public_sa": {
                "interface_type": "SA",
                "thickness": 0.003,
                "permittivity": 3.9,
                "source": "meshwell handoff SA fixture",
            },
        },
        preset_by_interface_type={
            "MA": "public_ma",
            "MS": "public_ms",
            "SA": "public_sa",
        },
    )

    assert tuple(spec.entry_names for spec in specs) == (
        ("A___B",),
        ("B___helper",),
    )

    config = build_postprocessing_config_from_manifest(
        manifest,
        surface_flux=(
            SurfaceFluxSpec(
                role="boundary_surface",
                entry_names=("A___None",),
                flux_type="Power",
                two_sided=None,
            ),
        ),
        dielectric_interfaces=specs,
    )

    assert config.boundaries["Dielectric"] == [
        {
            "Index": 1,
            "Attributes": list(manifest_entries["A___B"].attributes),
            "Type": "MS",
            "Thickness": 0.002,
            "LossTan": 0.0,
            "Permittivity": 11.45,
        },
        {
            "Index": 2,
            "Attributes": list(manifest_entries["B___helper"].attributes),
            "Type": "SA",
            "Thickness": 0.003,
            "LossTan": 0.0,
            "Permittivity": 3.9,
        },
    ]
    assert config.boundaries["SurfaceFlux"] == [
        {
            "Index": 1,
            "Attributes": list(manifest_entries["A___None"].attributes),
            "Type": "Power",
        }
    ]

    index_map_path = config.index_map.write_json(tmp_path / "palace_index_map.json")
    loaded = load_postprocessing_index_map(tmp_path, index_map_path=index_map_path)

    assert (
        loaded.physical_name_for_index(
            "Boundaries.Postprocessing.Dielectric",
            1,
        )
        == "A___B"
    )
    assert (
        loaded.physical_name_for_index(
            "Boundaries.Postprocessing.Dielectric",
            2,
        )
        == "B___helper"
    )
    assert (
        loaded.physical_name_for_index(
            "Boundaries.Postprocessing.SurfaceFlux",
            1,
        )
        == "A___None"
    )
    assert loaded.indices_for_physical_name("A___B") == (1,)
    assert (
        loaded.entries_for_attribute(manifest_entries["A___None"].attributes[0])[
            0
        ].exterior_of
        == "A"
    )
