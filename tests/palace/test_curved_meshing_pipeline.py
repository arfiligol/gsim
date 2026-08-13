"""Unit tests for curved meshing pipeline behavior."""

from __future__ import annotations

import builtins
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gsim.common import Layer, LayerStack
from gsim.palace.mesh import generator as mesh_generator
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.mesh.manifest import MeshManifest, MeshPhysicalGroup
from gsim.palace.mesh.postprocessing import (
    DielectricInterfaceSpec,
    build_postprocessing_config_from_manifest,
)
from gsim.palace.models.problems import DrivenConfig


class _FakeOption:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def setNumber(self, name: str, value: float) -> None:  # noqa: N802 (gmsh API)
        self.calls.append((name, value))


class _FakeMeshOps:
    def __init__(self) -> None:
        self.generated_dim: int | None = None

    def generate(self, dim: int) -> None:
        self.generated_dim = dim

    def setOrder(self, _order: int) -> None:  # noqa: N802 (gmsh API)
        return

    def optimize(self, _method: str) -> None:
        return


class _FakeModel:
    def __init__(self) -> None:
        self.occ = object()
        self.mesh = _FakeMeshOps()
        self._models: list[str] = []

    def list(self) -> builtins.list[str]:
        return list(self._models)

    def setCurrent(self, _name: str) -> None:  # noqa: N802 (gmsh API)
        return

    def remove(self) -> None:
        self._models.clear()

    def add(self, name: str) -> None:
        self._models.append(name)


class _FakeFltk:
    def run(self) -> None:
        return


class _FakeGmsh:
    def __init__(self) -> None:
        self.option = _FakeOption()
        self.model = _FakeModel()
        self.fltk = _FakeFltk()
        self.cleared = False
        self.finalized = False
        self.writes: list[str] = []

    def initialize(self) -> None:
        return

    def clear(self) -> None:
        self.cleared = True

    def finalize(self) -> None:
        self.finalized = True

    def write(self, path: str) -> None:
        self.writes.append(path)


def test_generate_mesh_forwards_curve_fit_and_decimation(monkeypatch, tmp_path) -> None:
    """Curved meshing settings propagate through generate_mesh internals."""
    captured: dict[str, object] = {}
    fake_gmsh = _FakeGmsh()

    monkeypatch.setattr(mesh_generator, "gmsh", fake_gmsh)

    def _fake_extract_geometry(_component, _stack, decimate_tolerance=None):
        captured["decimate_tolerance"] = decimate_tolerance
        return SimpleNamespace(polygons=[object()], bbox=(0.0, 0.0, 10.0, 10.0))

    monkeypatch.setattr(mesh_generator, "extract_geometry", _fake_extract_geometry)
    monkeypatch.setattr(mesh_generator, "add_metals", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        mesh_generator,
        "add_ports",
        lambda *_args, **_kwargs: ({"P1": [11]}, []),
    )
    monkeypatch.setattr(
        mesh_generator,
        "add_dielectrics",
        lambda *_args, **_kwargs: {"air": [31]},
    )

    def _fake_add_patterned_dielectrics(
        _kernel,
        _geometry,
        _stack,
        *,
        curve_fit_mode,
        curve_fit_layers,
        curve_fit_tolerance_um,
        curve_fit_min_points,
        curve_fit_corner_angle_deg,
    ):
        captured["curve_fit"] = {
            "curve_fit_mode": curve_fit_mode,
            "curve_fit_layers": curve_fit_layers,
            "curve_fit_tolerance_um": curve_fit_tolerance_um,
            "curve_fit_min_points": curve_fit_min_points,
            "curve_fit_corner_angle_deg": curve_fit_corner_angle_deg,
        }
        return {"core": [21]}

    monkeypatch.setattr(
        mesh_generator,
        "add_patterned_dielectrics",
        _fake_add_patterned_dielectrics,
    )

    def _fake_build_entities(
        metal_tags,
        dielectric_tags,
        patterned_dielectric_tags,
        port_tags,
        port_info,
        pec_block_tags,
        stack,
    ):
        captured["build_entities_args"] = {
            "dielectric_tags": dielectric_tags,
            "patterned_dielectric_tags": patterned_dielectric_tags,
            "port_tags": port_tags,
            "port_info": port_info,
            "pec_block_tags": pec_block_tags,
            "stack": stack,
            "metal_tags": metal_tags,
        }
        return []

    monkeypatch.setattr(mesh_generator, "build_entities", _fake_build_entities)
    monkeypatch.setattr(
        mesh_generator.gmsh_utils,
        "run_boolean_pipeline",
        lambda _entities: {},
    )

    def _fake_assign_physical_groups(
        _kernel,
        _metal_tags,
        all_dielectric_tags,
        _port_tags,
        _port_info,
        _entities,
        _pg_map,
        _stack,
        pec_block_tags=None,
    ):
        captured["all_dielectric_tags"] = all_dielectric_tags
        captured["assign_pec_block_tags"] = pec_block_tags
        return {
            "volumes": {},
            "conductor_surfaces": {},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        }

    monkeypatch.setattr(
        mesh_generator,
        "assign_physical_groups",
        _fake_assign_physical_groups,
    )
    monkeypatch.setattr(
        mesh_generator, "_setup_mesh_fields", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(mesh_generator, "collect_mesh_stats", lambda: {"nodes": 1})

    stack = LayerStack()
    result = mesh_generator.generate_mesh(
        component=object(),
        stack=stack,
        ports=[],
        output_dir=tmp_path,
        curve_fit_mode="bspline",
        curve_fit_layers=["core", "core2"],
        curve_fit_tolerance_um=0.02,
        curve_fit_min_points=12,
        curve_fit_corner_angle_deg=30.0,
        decimate_tolerance=0.005,
        verbosity=7,
        write_config=False,
    )

    assert captured["decimate_tolerance"] == 0.005
    assert captured["curve_fit"] == {
        "curve_fit_mode": "bspline",
        "curve_fit_layers": ["core", "core2"],
        "curve_fit_tolerance_um": 0.02,
        "curve_fit_min_points": 12,
        "curve_fit_corner_angle_deg": 30.0,
    }
    assert captured["all_dielectric_tags"] == {"air": [31], "core": [21]}
    assert captured["assign_pec_block_tags"] is None
    assert ("General.Verbosity", 7) in fake_gmsh.option.calls
    assert result.mesh_path == tmp_path / "palace.msh"
    assert fake_gmsh.cleared is True
    assert fake_gmsh.finalized is True


def test_generate_palace_config_shaped_dielectric_layer_material(tmp_path) -> None:
    """Shaped dielectrics resolve material properties via the stack layer map."""
    stack = LayerStack()
    stack.layers["CORE"] = Layer(
        name="CORE",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.22,
        thickness=0.22,
        material="silicon",
        layer_type="dielectric",
    )
    stack.materials = {
        "silicon": {
            "permittivity": 12.1,
            "loss_tangent": 0.002,
        }
    }

    groups = {
        "volumes": {
            "CORE": {
                "phys_group": 101,
                "is_shaped_dielectric": True,
            }
        },
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }

    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="driven",
        absorbing_boundary=False,
    )

    config = json.loads(config_path.read_text())
    materials = config["Domains"]["Materials"]

    core_mat = next(
        (entry for entry in materials if 101 in entry.get("Attributes", [])),
        None,
    )
    assert core_mat is not None
    assert core_mat["Permittivity"] == 12.1
    assert core_mat["LossTan"] == 0.002


@pytest.mark.parametrize("palace_version", ["0.16.0", "0.16.1"])
def test_surface_epr_material_name_resolves_before_config_serialization(
    tmp_path, palace_version
) -> None:
    """Surface-EPR material tokens resolve into schema-valid Palace rows."""
    stack = LayerStack()
    stack.layers["SUB"] = Layer(
        name="SUB",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=1.0,
        thickness=1.0,
        material="silicon",
        layer_type="dielectric",
    )
    stack.materials = {
        "silicon": {"permittivity": 11.45, "loss_tangent": 2.6e-7},
        "interface_oxide": {"permittivity": 3.9, "loss_tangent": 0.001},
    }
    groups = {
        "volumes": {"SUB": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    manifest = MeshManifest(
        entries=(
            MeshPhysicalGroup(
                name="SA",
                role="boundary_surface",
                attributes=(2,),
                interface_of=("SUB", "AIR"),
            ),
        )
    )
    postprocessing = build_postprocessing_config_from_manifest(
        manifest,
        energy_roles=(),
        dielectric_interfaces=(
            DielectricInterfaceSpec(
                interface_type="SA",
                thickness=1e-3,
                material_name="interface_oxide",
                role="boundary_surface",
                entry_names=("SA",),
                preset_name="public-oxide",
                preset_source="public-fixture",
            ),
        ),
    )
    output_path = tmp_path / palace_version
    output_path.mkdir()

    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=output_path,
        model_name="palace",
        fmax=10e9,
        simulation_type="driven",
        driven_config=DrivenConfig(fmin=1e9, fmax=10e9, num_points=2),
        absorbing_boundary=False,
        boundary_postprocessing_config=postprocessing.boundaries,
        palace_version=palace_version,
        validate_schema=True,
    )

    config = json.loads(config_path.read_text())
    interface = config["Boundaries"]["Postprocessing"]["Dielectric"][0]
    assert interface == {
        "Index": 1,
        "Attributes": [2],
        "Type": "SA",
        "Thickness": 1e-3,
        "LossTan": 0.001,
        "Permittivity": 3.9,
    }
    assert postprocessing.index_map.entries[0].extra == {
        "Type": "SA",
        "preset_name": "public-oxide",
        "preset_source": "public-fixture",
        "material_name": "interface_oxide",
    }
    resolution = json.loads(
        (output_path / "metadata" / "palace_material_resolution.json").read_text()
    )
    assert resolution["interfaces"][0]["interface_material_name"] == "interface_oxide"
    assert resolution["interfaces"][0]["palace_interface"] == interface

    def _keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from _keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from _keys(child)

    assert not any(key.startswith("_") for key in _keys(config))


def test_surface_epr_material_name_and_private_config_keys_fail_closed(
    tmp_path,
) -> None:
    """Unknown material tokens and unconsumed private keys cannot serialize."""
    stack = LayerStack()
    stack.materials = {}
    groups = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    dielectric = {
        "Dielectric": [
            {
                "Index": 1,
                "Attributes": [2],
                "Type": "SA",
                "Thickness": 1e-3,
                "LossTan": 0.0,
                "_MaterialName": "unknown-material",
            }
        ]
    }
    with pytest.raises(ValueError, match="Permittivity"):
        generate_palace_config(
            groups=groups,
            ports=[],
            port_info=[],
            stack=stack,
            output_path=tmp_path,
            model_name="palace",
            fmax=10e9,
            simulation_type="driven",
            absorbing_boundary=False,
            boundary_postprocessing_config=dielectric,
        )
    with pytest.raises(ValueError, match="Internal Palace configuration key"):
        generate_palace_config(
            groups=groups,
            ports=[],
            port_info=[],
            stack=stack,
            output_path=tmp_path,
            model_name="palace",
            fmax=10e9,
            simulation_type="driven",
            absorbing_boundary=False,
            hints={"Solver": {"_injected": True}},
        )
    with pytest.raises(ValueError, match="Internal Palace configuration key"):
        generate_palace_config(
            groups=groups,
            ports=[],
            port_info=[],
            stack=stack,
            output_path=tmp_path,
            model_name="palace",
            fmax=10e9,
            simulation_type="driven",
            absorbing_boundary=False,
            hints={"Solver": {"nested": ({"_injected": True},)}},
        )


def test_surface_epr_material_fragment_is_reusable_without_mutation(tmp_path) -> None:
    """Material-backed postprocessing fragments resolve afresh on each write."""
    stack = LayerStack()
    stack.materials = {"interface_oxide": {"permittivity": 3.0}}
    groups = {
        "volumes": {},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    fragment = {
        "Dielectric": [
            {
                "Index": 1,
                "Attributes": [2],
                "Type": "SA",
                "Thickness": 1e-3,
                "LossTan": 0.0,
                "_MaterialName": "interface_oxide",
            }
        ]
    }
    original_fragment = deepcopy(fragment)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_output.mkdir()
    second_output.mkdir()

    first = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=first_output,
        model_name="palace",
        fmax=10e9,
        simulation_type="driven",
        absorbing_boundary=False,
        boundary_postprocessing_config=fragment,
        material_overlay={
            "materials": {
                "interface_oxide": {"permittivity": 3.1, "loss_tangent": 0.01}
            }
        },
    )
    second = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=second_output,
        model_name="palace",
        fmax=20e9,
        simulation_type="driven",
        absorbing_boundary=False,
        boundary_postprocessing_config=fragment,
        material_overlay={
            "materials": {
                "interface_oxide": {"permittivity": 4.2, "loss_tangent": 0.02}
            }
        },
    )

    assert fragment == original_fragment
    first_interface = json.loads(first.read_text())["Boundaries"]["Postprocessing"][
        "Dielectric"
    ][0]
    second_interface = json.loads(second.read_text())["Boundaries"]["Postprocessing"][
        "Dielectric"
    ][0]
    assert first_interface["Permittivity"] == 3.1
    assert second_interface["Permittivity"] == 4.2
    for output_path in (first_output, second_output):
        resolution = json.loads(
            (output_path / "metadata" / "palace_material_resolution.json").read_text()
        )
        assert resolution["interfaces"][0]["interface_material_name"] == (
            "interface_oxide"
        )


@pytest.mark.parametrize("palace_version", ["0.16.0", "0.16.1"])
def test_schema_valid_native_hints_keep_solver_and_output_owners(
    tmp_path, palace_version
) -> None:
    """Native hints tune permitted solver/output leaves for both schemas."""
    stack = LayerStack()
    stack.layers["SUB"] = Layer(
        name="SUB",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=1.0,
        thickness=1.0,
        material="silicon",
        layer_type="dielectric",
    )
    stack.materials = {"silicon": {"permittivity": 11.45, "loss_tangent": 0.0}}
    config_path = generate_palace_config(
        groups={
            "volumes": {"SUB": {"phys_group": 1}},
            "conductor_surfaces": {},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        },
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        simulation_type="driven",
        driven_config=DrivenConfig(fmin=1e9, fmax=10e9, num_points=2),
        absorbing_boundary=False,
        hints={
            "Model": {"Refinement": {"MaxIts": 3}},
            "Solver": {"Linear": {"EstimatorMG": True}},
            "Problem": {"OutputFormats": {"Paraview": True}},
        },
        palace_version=palace_version,
        validate_schema=True,
    )
    config = json.loads(config_path.read_text())
    assert config["Model"]["Refinement"]["MaxIts"] == 3
    assert config["Solver"]["Linear"]["EstimatorMG"] is True
    assert config["Problem"]["OutputFormats"] == {"Paraview": True}
