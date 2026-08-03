"""Tests for mesh configuration."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.palace.mesh import MeshConfig, gmsh_utils
from gsim.palace.mesh import validation as mesh_validation
from gsim.palace.mesh.generator import generate_mesh
from gsim.palace.mesh.geometry import (
    GeometryData,
    _snap_via_z_range,
    add_dielectrics,
    add_metals,
    add_ports,
    build_entities,
    extract_geometry,
)
from gsim.palace.mesh.gmsh_utils import Entity
from gsim.palace.mesh.groups import assign_physical_groups
from gsim.palace.mesh.manifest import build_mesh_manifest
from gsim.palace.mesh.sheets import AuthoredSheetPolygon
from gsim.palace.mesh.surface_epr import build_interface_surface_catalog
from gsim.palace.models import (
    ActivatedRegion,
    PalacePort,
    PortGeometry,
    PortType,
    SimulationLayerCatalog,
)
from gsim.palace.ports import configure_cpw_port, extract_ports


def test_finite_conductor_shell_surfaces_do_not_claim_sgb_interface_catalog() -> None:
    import gmsh

    stack = LayerStack(
        layers={
            "D0_SUBSTRATE": Layer(
                name="D0_SUBSTRATE",
                gds_layer=(200, 0),
                zmin=-10.0,
                zmax=0.0,
                thickness=10.0,
                material="silicon",
                layer_type="substrate",
            ),
            "D0_TOP_M1": Layer(
                name="D0_TOP_M1",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.2,
                thickness=0.2,
                material="aluminum",
                layer_type="conductor",
            ),
        },
        materials={
            "silicon": {"permittivity": 11.45},
            "air": {"permittivity": 1.0},
            "aluminum": {"conductivity": 3.7e7},
        },
        dielectrics=[
            {
                "name": "substrate",
                "material": "silicon",
                "zmin": -10.0,
                "zmax": 0.0,
            },
            {
                "name": "air",
                "material": "air",
                "zmin": 0.0,
                "zmax": 5.0,
            },
        ],
    )
    geometry = GeometryData(
        polygons=[
            (1, [0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 4.0, 4.0], []),
            (1, [6.0, 10.0, 10.0, 6.0], [0.0, 0.0, 4.0, 4.0], []),
        ],
        bbox=(0.0, 0.0, 10.0, 4.0),
        layer_bboxes={1: (0.0, 0.0, 10.0, 4.0)},
    )

    already_initialized = gmsh.isInitialized()
    if not already_initialized:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        gmsh.model.add("finite_conductor_b_interfaces")
        kernel = gmsh.model.occ
        metal_result = add_metals(
            kernel,
            geometry,
            stack,
            planar_conductors=False,
        )
        dielectric_tags = add_dielectrics(
            kernel,
            geometry,
            stack,
            margin_x=1.0,
            margin_y=1.0,
            air_margin=0.0,
        )
        entities = build_entities(
            metal_tags=metal_result.metal_tags,
            dielectric_tags=dielectric_tags,
            patterned_dielectric_tags={},
            port_tags={},
            port_info=[],
            stack=stack,
        )
        pg_map = gmsh_utils.run_boolean_pipeline(entities)
        groups = assign_physical_groups(
            kernel,
            metal_tags=metal_result.metal_tags,
            dielectric_tags=dielectric_tags,
            port_tags={},
            port_info=[],
            entities=entities,
            pg_map=pg_map,
            _stack=stack,
        )
    finally:
        gmsh.clear()
        if not already_initialized:
            gmsh.finalize()

    catalog = build_interface_surface_catalog(groups)
    catalog_names = {surface.physical_group_name for surface in catalog.surfaces}

    assert catalog_names == set()
    assert set(groups["conductor_surfaces"]) == {"D0_TOP_M1_xy", "D0_TOP_M1_z"}


class TestMeshConfig:
    """Test MeshConfig class."""

    def test_default_config(self):
        """Test default MeshConfig values."""
        config = MeshConfig()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.margin == 0.0
        assert config.fmax == 100e9
        assert config.show_gui is False
        assert config.boundary_conditions is not None
        assert len(config.boundary_conditions) == 6
        assert config.curve_fit_mode == "line"
        assert config.curve_fit_layers == ["core", "core2"]
        assert config.curve_fit_tolerance_um == 0.0
        assert config.curve_fit_min_points == 8
        assert config.curve_fit_corner_angle_deg == 45.0
        assert config.high_order_elements is False
        assert config.high_order_order == 2
        assert config.high_order_optimize is True

    def test_coarse_preset(self):
        """Test coarse mesh preset."""
        config = MeshConfig.coarse()
        assert config.refined_mesh_size == 10.0
        assert config.max_mesh_size == 600.0
        assert config.cells_per_wavelength == 5

    def test_default_preset(self):
        """Test default mesh preset."""
        config = MeshConfig.default()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.cells_per_wavelength == 10

    def test_fine_preset(self):
        """Test fine mesh preset."""
        config = MeshConfig.fine()
        assert config.refined_mesh_size == 2.0
        assert config.max_mesh_size == 70.0
        assert config.cells_per_wavelength == 20

    def test_preset_with_overrides(self):
        """Test preset with custom overrides."""
        config = MeshConfig.coarse(margin=100.0, fmax=50e9)
        assert config.refined_mesh_size == 10.0  # From preset
        assert config.margin == 100.0  # Override
        assert config.fmax == 50e9  # Override

    def test_custom_config(self):
        """Test fully custom config."""
        config = MeshConfig(
            refined_mesh_size=3.0,
            max_mesh_size=200.0,
            margin=75.0,
        )
        assert config.refined_mesh_size == 3.0
        assert config.max_mesh_size == 200.0
        assert config.margin == 75.0

    def test_curve_fit_overrides(self):
        """Test custom curve-fit settings."""
        config = MeshConfig(
            curve_fit_mode="bspline",
            curve_fit_layers=["core"],
            curve_fit_tolerance_um=0.02,
            curve_fit_min_points=12,
            curve_fit_corner_angle_deg=30.0,
        )
        assert config.curve_fit_mode == "bspline"
        assert config.curve_fit_layers == ["core"]
        assert config.curve_fit_tolerance_um == 0.02
        assert config.curve_fit_min_points == 12
        assert config.curve_fit_corner_angle_deg == 30.0

    def test_high_order_overrides(self):
        """Test custom high-order mesh settings."""
        config = MeshConfig(
            high_order_elements=True,
            high_order_order=3,
            high_order_optimize=False,
        )
        assert config.high_order_elements is True
        assert config.high_order_order == 3
        assert config.high_order_optimize is False


def test_add_dielectrics_margin_applies_only_to_airlike(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air", "zmin": 0.5, "zmax": 8.0, "material": "air"},
            {"name": "passivation", "zmin": 8.0, "zmax": 9.0, "material": "passive"},
        ],
        materials={
            "SiO2": {"permittivity": 4.1},
            "air": {"permittivity": 1.0},
            "passive": {"permittivity": 6.6},
        },
    )

    add_dielectrics(
        _Kernel(), geometry, stack, margin_x=5.0, margin_y=7.0, air_margin=0.0
    )

    # SiO2 (substrate starting at z=-2 < 0) and air expand by margins.
    # Passivation (thin film on top) keeps original bbox.
    assert calls[0] == (-5.0, -7.0, -2.0, 15.0, 27.0, 0.5)
    assert calls[1] == (-5.0, -7.0, 0.5, 15.0, 27.0, 8.0)
    assert calls[2] == (0.0, 0.0, 8.0, 10.0, 20.0, 9.0)


def test_add_dielectrics_explicit_airbox_z_extents(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air", "zmin": 0.5, "zmax": 8.0, "material": "air"},
        ],
        materials={
            "SiO2": {"permittivity": 4.1},
            "air": {"permittivity": 1.0},
        },
    )

    tags = add_dielectrics(
        _Kernel(),
        geometry,
        stack,
        margin_x=5.0,
        margin_y=7.0,
        air_margin=0.0,
        airbox_z_above=100.0,
        airbox_z_below=100.0,
    )

    assert "airbox" in tags
    # Air layer is skipped when explicit airbox is built.
    assert list(tags.keys()) == ["SiO2", "airbox"]


def test_add_dielectrics_explicit_airbox_skips_named_air_regions(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {
                "name": "air_box_top",
                "zmin": 0.5,
                "zmax": 8.0,
                "material": "ambient",
            },
        ],
        materials={
            "SiO2": {"type": "dielectric", "permittivity": 4.1},
            # Intentionally non-air metadata to force name-based detection.
            "ambient": {"type": "unknown"},
        },
    )

    tags = add_dielectrics(
        _Kernel(),
        geometry,
        stack,
        margin_x=5.0,
        margin_y=7.0,
        air_margin=0.0,
        airbox_z_above=100.0,
        airbox_z_below=100.0,
    )

    assert list(tags.keys()) == ["SiO2", "airbox"]
    # Oxide + explicit airbox only.
    assert len(calls) == 2


def test_add_dielectrics_activated_regions_preserve_stack_layer_names(
    monkeypatch,
) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        layers={
            "D0_SUBSTRATE": Layer(
                name="D0_SUBSTRATE",
                gds_layer=(201, 0),
                zmin=-500.0,
                zmax=0.0,
                thickness=500.0,
                material="Si",
                layer_type="substrate",
            ),
            "D1_SUBSTRATE": Layer(
                name="D1_SUBSTRATE",
                gds_layer=(201, 2),
                zmin=10.2,
                zmax=510.2,
                thickness=500.0,
                material="Si",
                layer_type="substrate",
            ),
            "D0_TO_D1_GAP": Layer(
                name="D0_TO_D1_GAP",
                gds_layer=(201, 1),
                zmin=0.0,
                zmax=10.2,
                thickness=10.2,
                material="vacuum",
                layer_type="dielectric",
            ),
            "OUTER_VACUUM": Layer(
                name="OUTER_VACUUM",
                gds_layer=(201, 3),
                zmin=510.2,
                zmax=1510.2,
                thickness=1000.0,
                material="vacuum",
                layer_type="dielectric",
            ),
        },
        materials={
            "Si": {"permittivity": 11.9},
            "vacuum": {"permittivity": 1.0},
        },
    )

    tags = add_dielectrics(
        _Kernel(),
        geometry,
        stack,
        margin_x=5.0,
        margin_y=7.0,
        activated_regions=(
            ActivatedRegion(
                layer="D0_SUBSTRATE",
                role="substrate",
                margin_x=5.0,
                margin_y=7.0,
            ),
            ActivatedRegion(
                layer="D1_SUBSTRATE",
                role="substrate",
                margin_x=6.0,
                margin_y=8.0,
            ),
            ActivatedRegion(
                layer="D0_TO_D1_GAP",
                role="inter_die_vacuum",
                lower_die="D0",
                upper_die="D1",
                margin_x=3.0,
                margin_y=4.0,
            ),
            ActivatedRegion(
                layer="OUTER_VACUUM",
                role="outer_vacuum",
                margin_x=9.0,
                margin_y=11.0,
                z_above=100.0,
                z_below=20.0,
            ),
        ),
    )

    assert list(tags) == [
        "D0_SUBSTRATE",
        "D1_SUBSTRATE",
        "D0_TO_D1_GAP",
        "OUTER_VACUUM",
    ]
    assert tags == {
        "D0_SUBSTRATE": [1],
        "D1_SUBSTRATE": [2],
        "D0_TO_D1_GAP": [3],
        "OUTER_VACUUM": [4],
    }
    assert calls[0] == (-5.0, -7.0, -500.0, 15.0, 27.0, 0.0)
    assert calls[1] == (-6.0, -8.0, 10.2, 16.0, 28.0, 510.2)
    assert calls[2] == (-3.0, -4.0, 0.0, 13.0, 24.0, 10.2)
    assert calls[3] == (-15.0, -19.0, -520.0, 25.0, 39.0, 610.2)


def test_add_dielectrics_activated_outer_vacuum_requires_substrate() -> None:
    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        layers={
            "OUTER_VACUUM": Layer(
                name="OUTER_VACUUM",
                gds_layer=(201, 3),
                zmin=0.0,
                zmax=100.0,
                thickness=100.0,
                material="vacuum",
                layer_type="dielectric",
            )
        },
        materials={"vacuum": {"permittivity": 1.0}},
    )

    with pytest.raises(ValueError, match="requires at least one active substrate"):
        add_dielectrics(
            _Kernel(),
            geometry,
            stack,
            margin_x=5.0,
            activated_regions=(
                ActivatedRegion(layer="OUTER_VACUUM", role="outer_vacuum"),
            ),
        )


def test_add_dielectrics_activated_regions_reject_airbox_controls() -> None:
    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        layers={
            "D0_SUBSTRATE": Layer(
                name="D0_SUBSTRATE",
                gds_layer=(201, 0),
                zmin=-500.0,
                zmax=0.0,
                thickness=500.0,
                material="Si",
                layer_type="substrate",
            )
        },
        materials={"Si": {"permittivity": 11.9}},
    )

    with pytest.raises(ValueError, match="air_margin"):
        add_dielectrics(
            _Kernel(),
            geometry,
            stack,
            margin_x=5.0,
            air_margin=1.0,
            activated_regions=(
                ActivatedRegion(layer="D0_SUBSTRATE", role="substrate"),
            ),
        )


def test_generate_mesh_activated_regions_reject_airbox_controls(tmp_path) -> None:
    with pytest.raises(ValueError, match="air_margin"):
        generate_mesh(
            component=object(),
            stack=LayerStack(),
            ports=[],
            output_dir=tmp_path,
            activated_regions=(
                ActivatedRegion(layer="D0_SUBSTRATE", role="substrate"),
            ),
        )

    with pytest.raises(ValueError, match="airbox_z_above"):
        generate_mesh(
            component=object(),
            stack=LayerStack(),
            ports=[],
            output_dir=tmp_path,
            air_margin=0.0,
            airbox_z_above=0.0,
            activated_regions=(
                ActivatedRegion(layer="D0_SUBSTRATE", role="substrate"),
            ),
        )


def test_assign_physical_groups_preserves_activated_region_metadata() -> None:
    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(
        layers={
            "OUTER_VACUUM": Layer(
                name="OUTER_VACUUM",
                gds_layer=(201, 3),
                zmin=0.0,
                zmax=100.0,
                thickness=100.0,
                material="vacuum",
                layer_type="dielectric",
            )
        },
        materials={
            "vacuum": {"permittivity": 1.0},
            "clean_vacuum": {"permittivity": 1.02},
        },
    )

    groups = assign_physical_groups(
        _Kernel(),
        metal_tags={},
        dielectric_tags={"OUTER_VACUUM": [44]},
        port_tags={},
        port_info=[],
        entities=[Entity(name="OUTER_VACUUM", dim=3, mesh_order=4, tags=[44])],
        pg_map={"OUTER_VACUUM": 77},
        _stack=stack,
        activated_regions=(
            ActivatedRegion(
                layer="OUTER_VACUUM",
                role="outer_vacuum",
                margin_x=9.0,
                margin_y=11.0,
                z_above=100.0,
                material="clean_vacuum",
            ),
        ),
    )

    info = groups["volumes"]["OUTER_VACUUM"]
    manifest_entry = {
        entry.name: entry for entry in build_mesh_manifest(groups).entries
    }["OUTER_VACUUM"]

    assert info["material"] == "clean_vacuum"
    assert info["material_override"] == "clean_vacuum"
    assert info["activated_region_role"] == "outer_vacuum"
    assert info["margin_x"] == 9.0
    assert manifest_entry.physical_names == ("OUTER_VACUUM",)
    assert manifest_entry.metadata["material_override"] == "clean_vacuum"


def test_add_dielectrics_activated_regions_reject_conductors() -> None:
    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        layers={
            "D0_TOP_M1": Layer(
                name="D0_TOP_M1",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.2,
                thickness=0.2,
                material="Al",
                layer_type="conductor",
            )
        }
    )

    with pytest.raises(ValueError, match="must be a dielectric or substrate"):
        add_dielectrics(
            _Kernel(),
            geometry,
            stack,
            margin_x=5.0,
            activated_regions=(ActivatedRegion(layer="D0_TOP_M1", role="substrate"),),
        )


def test_add_ports_waveport_max_size_uses_3d_domain_bounds(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_port_rectangle(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return 77

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_port_rectangle",
        _fake_create_port_rectangle,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(
        layers={
            "metal1": _mk_layer("metal1", 1.0, 2.0, "conductor"),
        }
    )
    port = PalacePort(
        name="o1",
        port_type=PortType.WAVEPORT,
        geometry=PortGeometry.INPLANE,
        center=(5.0, 0.0),
        width=4.0,
        orientation=0.0,
        layer="metal1",
        max_size=True,
    )

    port_tags, port_info = add_ports(
        _Kernel(),
        [port],
        stack,
        domain_bounds=(-100.0, -60.0, -20.0, 120.0, 80.0, 40.0),
    )

    assert port_tags == {"P1": [77]}
    assert calls == [(5.0, -60.0, -20.0, 5.0, 80.0, 40.0)]
    assert port_info[0]["zmin"] == -20.0
    assert port_info[0]["zmax"] == 40.0
    assert port_info[0]["ymin"] == -60.0
    assert port_info[0]["ymax"] == 80.0


def _mk_layer(
    name: str,
    zmin: float,
    zmax: float,
    ltype: Literal["conductor", "via", "dielectric", "substrate"] = "conductor",
    gds_layer: tuple[int, int] = (0, 0),
) -> Layer:
    return Layer(
        name=name,
        gds_layer=gds_layer,
        zmin=zmin,
        zmax=zmax,
        thickness=zmax - zmin,
        material="aluminum",
        layer_type=ltype,
    )


def test_add_ports_lumped_inplane_uses_orientation_for_generated_sheet(
    monkeypatch,
) -> None:
    calls: list[tuple[list[float], list[float], float]] = []

    def _fake_create_polygon_surface(_kernel, pts_x, pts_y, z, *_args, **_kwargs):
        calls.append((pts_x, pts_y, z))
        return 91

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_polygon_surface",
        _fake_create_polygon_surface,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})
    port = PalacePort(
        name="o1",
        center=(10.0, 20.0),
        width=2.0,
        length=4.0,
        orientation=45.0,
        layer="metal1",
        direction="+X",
    )

    port_tags, port_info = add_ports(_Kernel(), [port], stack)

    root2 = np.sqrt(2.0)
    longitudinal = np.array([1.0 / root2, 1.0 / root2])
    transverse = np.array([-1.0 / root2, 1.0 / root2])
    center = np.array([10.0, 20.0])
    expected = np.array(
        [
            center - 2.0 * longitudinal - transverse,
            center + 2.0 * longitudinal - transverse,
            center + 2.0 * longitudinal + transverse,
            center - 2.0 * longitudinal + transverse,
        ]
    )

    assert port_tags == {"P1": [91]}
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(expected[:, 0])
    assert calls[0][1] == pytest.approx(expected[:, 1])
    assert calls[0][2] == pytest.approx(1.0)
    assert port_info[0]["direction"] == [1.0, 0.0, 0.0]
    assert np.array(port_info[0]["corners"]) == pytest.approx(expected)


def _sheet_polygon(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    gds_layer: tuple[int, int] = (202, 1),
) -> AuthoredSheetPolygon:
    return AuthoredSheetPolygon(
        gds_layer=gds_layer,
        pts_x=[xmin, xmax, xmax, xmin],
        pts_y=[ymin, ymin, ymax, ymax],
        holes=[],
    )


def test_add_ports_lumped_inplane_uses_authored_sheet_polygon(monkeypatch) -> None:
    calls: list[tuple[list[float], list[float], float]] = []

    def _fake_create_polygon_surface(_kernel, pts_x, pts_y, z, *_args, **_kwargs):
        calls.append((pts_x, pts_y, z))
        return 101

    monkeypatch.setattr(
        "gsim.palace.mesh.sheets.gmsh_utils.create_polygon_surface",
        _fake_create_polygon_surface,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})
    catalog = SimulationLayerCatalog({"sim_sheet": {"gds_layer": (202, 1), "z": 3.0}})
    port = PalacePort(
        name="o1",
        center=(0.0, 0.0),
        width=2.0,
        length=4.0,
        orientation=45.0,
        layer="metal1",
        direction="+X",
        generate_sheet=False,
        sheet_gds_layer=(202, 1),
    )

    port_tags, port_info = add_ports(
        _Kernel(),
        [port],
        stack,
        simulation_layers=catalog,
        authored_sheet_polygons={(202, 1): [_sheet_polygon(-2.0, -1.0, 2.0, 1.0)]},
    )

    assert port_tags == {"P1": [101]}
    assert calls == [([-2.0, 2.0, 2.0, -2.0], [-1.0, -1.0, 1.0, 1.0], 3.0)]
    assert port_info[0]["sheet_source"] == "layout-authored"
    assert port_info[0]["sheet_layer"] == "sim_sheet"
    assert port_info[0]["sheet_gds_layer"] == [202, 1]
    assert port_info[0]["zmin"] == 3.0
    assert "corners" not in port_info[0]


def test_add_ports_lumped_authored_sheet_requires_catalog() -> None:
    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})
    port = PalacePort(
        name="o1",
        center=(0.0, 0.0),
        width=2.0,
        layer="metal1",
        generate_sheet=False,
        sheet_gds_layer=(202, 1),
    )

    with pytest.raises(ValueError, match="set_simulation_layers"):
        add_ports(_Kernel(), [port], stack)


def test_add_ports_lumped_authored_sheet_rejects_multiple_matches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gsim.palace.mesh.sheets.gmsh_utils.create_polygon_surface",
        lambda *_args, **_kwargs: 102,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})
    catalog = SimulationLayerCatalog({"sim_sheet": {"gds_layer": (202, 1), "z": 3.0}})
    port = PalacePort(
        name="o1",
        center=(0.0, 0.0),
        width=2.0,
        layer="metal1",
        generate_sheet=False,
        sheet_gds_layer=(202, 1),
    )

    with pytest.raises(ValueError, match="2 polygons cover center"):
        add_ports(
            _Kernel(),
            [port],
            stack,
            simulation_layers=catalog,
            authored_sheet_polygons={
                (202, 1): [
                    _sheet_polygon(-2.0, -2.0, 2.0, 2.0),
                    _sheet_polygon(-1.0, -1.0, 1.0, 1.0),
                ]
            },
        )


def test_add_ports_lumped_authored_sheet_rejects_missing_match(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gsim.palace.mesh.sheets.gmsh_utils.create_polygon_surface",
        lambda *_args, **_kwargs: 103,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})
    catalog = SimulationLayerCatalog({"sim_sheet": {"gds_layer": (202, 1), "z": 3.0}})
    port = PalacePort(
        name="o1",
        center=(0.0, 0.0),
        width=2.0,
        layer="metal1",
        generate_sheet=False,
        sheet_gds_layer=(202, 1),
    )

    with pytest.raises(ValueError, match="no polygon covers center"):
        add_ports(
            _Kernel(),
            [port],
            stack,
            simulation_layers=catalog,
            authored_sheet_polygons={
                (202, 1): [_sheet_polygon(10.0, 10.0, 12.0, 12.0)]
            },
        )


def test_add_ports_cpw_uses_authored_sheet_polygons(monkeypatch) -> None:
    calls: list[tuple[list[float], list[float], float]] = []

    def _fake_create_polygon_surface(_kernel, pts_x, pts_y, z, *_args, **_kwargs):
        calls.append((pts_x, pts_y, z))
        return 200 + len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.sheets.gmsh_utils.create_polygon_surface",
        _fake_create_polygon_surface,
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})
    catalog = SimulationLayerCatalog({"sim_sheet": {"gds_layer": (202, 1), "z": 3.0}})
    port = PalacePort(
        name="cpw",
        layer="metal1",
        width=0.5,
        length=2.0,
        orientation=0.0,
        multi_element=True,
        centers=[(0.0, 2.0), (0.0, -2.0)],
        directions=[(0.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        generate_sheet=False,
        sheet_gds_layer=(202, 1),
    )

    port_tags, port_info = add_ports(
        _Kernel(),
        [port],
        stack,
        simulation_layers=catalog,
        authored_sheet_polygons={
            (202, 1): [
                _sheet_polygon(-1.0, 1.5, 1.0, 2.5),
                _sheet_polygon(-1.0, -2.5, 1.0, -1.5),
            ]
        },
    )

    assert port_tags == {"P1": [201, 202]}
    assert [call[2] for call in calls] == [3.0, 3.0]
    assert port_info[0]["type"] == "cpw"
    assert port_info[0]["sheet_source"] == "layout-authored"
    assert [element["sheet_source"] for element in port_info[0]["elements"]] == [
        "layout-authored",
        "layout-authored",
    ]


def test_sim_add_port_authored_sheet_records_gds_layer_from_gf_port() -> None:
    import gdsfactory as gf

    from gsim.palace import DrivenSim

    gf.gpdk.PDK.activate()

    component = gf.Component("authored_single_public_path")
    component.add_port(
        name="o1",
        center=(0.0, 0.0),
        width=2.0,
        orientation=0.0,
        layer=(202, 1),
        port_type="electrical",
    )
    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})

    sim = DrivenSim()
    sim.set_geometry(component)
    sim.stack = stack
    sim.set_simulation_layers({"sim_sheet": {"gds_layer": (202, 1), "z": 1.0}})
    sim.add_port("o1", layer="metal1", length=5.0, generate_sheet=False)
    sim._configure_ports_on_component(stack)

    palace_ports = extract_ports(component, stack)

    assert len(palace_ports) == 1
    assert palace_ports[0].generate_sheet is False
    assert palace_ports[0].sheet_gds_layer == (202, 1)


def test_sim_add_cpw_port_authored_sheet_records_gds_layer_from_gf_port() -> None:
    import gdsfactory as gf

    from gsim.palace import DrivenSim

    gf.gpdk.PDK.activate()

    component = gf.Component("authored_cpw_public_path")
    component.add_port(
        name="o1",
        center=(0.0, 0.0),
        width=10.0,
        orientation=0.0,
        layer=(202, 1),
        port_type="electrical",
    )
    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})

    sim = DrivenSim()
    sim.set_geometry(component)
    sim.stack = stack
    sim.set_simulation_layers({"sim_sheet": {"gds_layer": (202, 1), "z": 1.0}})
    sim.add_cpw_port(
        "o1",
        layer="metal1",
        s_width=10.0,
        gap_width=6.0,
        length=5.0,
        generate_sheet=False,
    )
    sim._configure_ports_on_component(stack)

    palace_ports = extract_ports(component, stack)

    assert len(palace_ports) == 1
    assert palace_ports[0].multi_element is True
    assert palace_ports[0].generate_sheet is False
    assert palace_ports[0].sheet_gds_layer == (202, 1)


def test_extract_geometry_excludes_simulation_sheet_layers() -> None:
    import gdsfactory as gf

    gf.gpdk.PDK.activate()

    component = gf.Component("authored_sheet_exclusion")
    component.add_polygon([(0, 0), (10, 0), (10, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(2, -1), (4, -1), (4, 1), (2, 1)], layer=(202, 1))

    stack = LayerStack(
        layers={
            "metal1": _mk_layer("metal1", 0.0, 0.1, gds_layer=(1, 0)),
            "sim_sheet": _mk_layer(
                "sim_sheet",
                0.0,
                0.0,
                ltype="dielectric",
                gds_layer=(202, 1),
            ),
        }
    )

    geometry = extract_geometry(
        component,
        stack,
        exclude_gds_layers={(202, 1)},
    )

    assert [layernum for layernum, *_ in geometry.polygons] == [1]


def test_extract_cpw_port_uses_orientation_transverse_direction_vectors() -> None:
    import gdsfactory as gf

    gf.gpdk.PDK.activate()

    component = gf.Component("cpw_direction_vector")
    component.add_port(
        name="o1",
        center=(0.0, 0.0),
        width=10.0,
        orientation=45.0,
        layer=(1, 0),
        port_type="electrical",
    )
    configure_cpw_port(
        component.ports["o1"],
        layer="metal1",
        s_width=10.0,
        gap_width=6.0,
        length=5.0,
    )
    stack = LayerStack(layers={"metal1": _mk_layer("metal1", 1.0, 2.0)})

    palace_ports = extract_ports(component, stack)

    assert len(palace_ports) == 1
    root2 = np.sqrt(2.0)
    assert np.array(palace_ports[0].directions) == pytest.approx(
        np.array(
            [
                (1.0 / root2, -1.0 / root2, 0.0),
                (-1.0 / root2, 1.0 / root2, 0.0),
            ]
        )
    )


class TestSnapViaZRange:
    """Snap via z-range to avoid sliver volumes against adjacent conductors."""

    def test_snaps_top_overlap_with_conductor(self):
        # Mimics IHP vmim z=[5.58, 6.24] vs topmetal1 z=[6.23, 8.23]
        stack = LayerStack(
            layers={
                "vmim": _mk_layer("vmim", 5.58, 6.24, "via"),
                "topmetal1": _mk_layer("topmetal1", 6.23, 8.23, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "vmim", 5.58, 6.24)
        assert new_zmin == pytest.approx(5.58)
        assert new_zmax == pytest.approx(6.23)

    def test_does_not_snap_large_overlap(self):
        # Overlap > tol must be left alone (likely a real geometric intersection).
        stack = LayerStack(
            layers={
                "via": _mk_layer("via", 0.0, 1.0, "via"),
                "metal": _mk_layer("metal", 0.5, 2.0, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "via", 0.0, 1.0, tol=0.05)
        assert (new_zmin, new_zmax) == (0.0, 1.0)

    def test_no_overlap_unchanged(self):
        stack = LayerStack(
            layers={
                "via": _mk_layer("via", 1.0, 2.0, "via"),
                # touches, no overlap
                "metal": _mk_layer("metal", 2.0, 3.0, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "via", 1.0, 2.0)
        assert (new_zmin, new_zmax) == (1.0, 2.0)

    def test_snaps_bottom_overlap_with_conductor(self):
        stack = LayerStack(
            layers={
                "via": _mk_layer("via", 1.99, 3.0, "via"),
                "metal": _mk_layer("metal", 1.0, 2.0, "conductor"),
            }
        )
        new_zmin, new_zmax = _snap_via_z_range(stack, "via", 1.99, 3.0)
        assert new_zmin == pytest.approx(2.0)
        assert new_zmax == pytest.approx(3.0)


class TestValidationHelpers:
    """Test helper routines used by mesh validation."""

    def test_parse_direction(self):
        """Direction parser returns normalized vectors and rejects unknown keys."""
        vec = mesh_validation._parse_direction("+X")
        assert np.allclose(vec, np.array([1.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="Unknown port direction"):
            mesh_validation._parse_direction("north")

    def test_perp_dist(self):
        """Perpendicular distance to an axis-aligned line is computed correctly."""
        v = np.array([0.0, 3.0, 4.0])
        origin = np.zeros(3)
        normals = [np.array([1.0, 0.0, 0.0])]
        dist = mesh_validation._perp_dist(v, normals, origin)
        assert dist == pytest.approx(5.0)

    def test_palace_obb_planar_rectangle(self):
        """OBB helper handles a simple planar rectangle point cloud."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        _center, axes, planar = mesh_validation._palace_obb(pts)
        lengths = [2.0 * float(np.linalg.norm(ax)) for ax in axes]
        assert planar is True
        assert max(lengths) == pytest.approx(2.0)
        assert min(lengths) == pytest.approx(0.0)
