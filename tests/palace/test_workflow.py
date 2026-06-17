"""Integration tests for Palace simulation workflows.

These tests verify the full local workflow for each sim type:
configure -> validate -> mesh -> write_config, stopping before cloud submission.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.palace import DrivenSim, EigenmodeSim, ElectrostaticSim, MagnetostaticSim
from gsim.palace.mesh import (
    SurfaceFluxSpec,
    build_postprocessing_config_from_manifest,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_cpw_component():
    """Create a simple GSG electrode component."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def gsg_electrode(
        length: float = 300,
        s_width: float = 10,
        g_width: float = 40,
        gap_width: float = 6,
        layer=gf.gpdk.LAYER.M1,
    ) -> gf.Component:
        c = gf.Component()
        r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r1.move((0, (g_width + s_width) / 2 + gap_width))
        c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
        r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r3.move((0, -(g_width + s_width) / 2 - gap_width))
        c.add_port(
            name="o1",
            center=(-length / 2, 0),
            width=s_width,
            orientation=0,
            port_type="electrical",
            layer=layer,
        )
        c.add_port(
            name="o2",
            center=(length / 2, 0),
            width=s_width,
            orientation=180,
            port_type="electrical",
            layer=layer,
        )
        return c

    return gsg_electrode()


@pytest.fixture(scope="module")
def cpw_component():
    """Create a shared CPW component for all tests in the module."""
    return _make_cpw_component()


# ---------------------------------------------------------------------------
# DrivenSim workflow
# ---------------------------------------------------------------------------


class TestDrivenSimWorkflow:
    """End-to-end DrivenSim: configure -> validate -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def driven_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh a DrivenSim with CPW ports."""
        tmp_path = tmp_path_factory.mktemp("driven")
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        sim.mesh(preset="coarse")
        return sim

    def test_validate_before_mesh(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "val"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        result = sim.validate_config()
        assert result.valid, f"Validation failed: {result}"

    def test_validate_mesh_raises_before_mesh(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "val-no-mesh"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)

        with pytest.raises(RuntimeError, match="No mesh generated"):
            sim.validate_mesh()

    def test_mesh_creates_file(self, driven_sim):
        mesh_path = Path(driven_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()
        assert mesh_path.stat().st_size > 0

    def test_mesh_has_physical_groups(self, driven_sim):
        groups = driven_sim._last_mesh_result.groups
        assert len(groups["volumes"]) > 0
        assert "P1" in groups["port_surfaces"]
        assert "P2" in groups["port_surfaces"]

    def test_mesh_has_absorbing_boundary(self, driven_sim):
        groups = driven_sim._last_mesh_result.groups
        assert "absorbing" in groups["boundary_surfaces"]

    def test_write_config_creates_json(self, driven_sim):
        driven_sim.write_config()
        config_path = Path(driven_sim._output_dir) / "config.json"
        assert config_path.exists()

    def test_config_json_structure(self, driven_sim):
        """Config JSON must have required Palace sections."""
        driven_sim.write_config()
        config_path = Path(driven_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())

        assert config["Problem"]["Type"] == "Driven"
        assert "Domains" in config
        assert "Boundaries" in config
        assert "Solver" in config

        boundaries = config["Boundaries"]
        assert "LumpedPort" in boundaries
        assert len(boundaries["LumpedPort"]) == 2
        assert "Absorbing" in boundaries

    def test_validate_mesh_passes(self, driven_sim):
        result = driven_sim.validate_mesh()
        assert result.valid, f"Mesh validation failed: {result}"

    def test_write_config_photonic_bypasses_validation(self, driven_sim, monkeypatch):
        def _validate_mesh_should_not_run(_self):
            raise AssertionError("validate_mesh() should be bypassed for photonic")

        monkeypatch.setattr(
            type(driven_sim),
            "validate_mesh",
            _validate_mesh_should_not_run,
        )
        config_path = driven_sim.write_config(photonic=True)
        assert Path(config_path).exists()

    def test_write_config_accepts_manifest_postprocessing(self, driven_sim):
        """High-level write_config wires postprocessing config and artifacts."""
        postprocessing = build_postprocessing_config_from_manifest(
            driven_sim._last_mesh_result.manifest,
            surface_flux=(
                SurfaceFluxSpec(
                    role="boundary_surface",
                    entry_names=("absorbing",),
                    flux_type="Power",
                    two_sided=None,
                ),
            ),
        )

        config_path = driven_sim.write_config(postprocessing=postprocessing)
        output_dir = Path(config_path).parent
        config = json.loads(config_path.read_text())

        assert config["Domains"]["Postprocessing"]["Energy"]
        assert config["Boundaries"]["Postprocessing"]["SurfaceFlux"][0]["Type"] == (
            "Power"
        )

        manifest_path = output_dir / "metadata" / "mesh_manifest.json"
        index_map_path = output_dir / "metadata" / "palace_index_map.json"
        assert manifest_path.exists()
        assert index_map_path.exists()

        manifest_json = json.loads(manifest_path.read_text())
        index_map_json = json.loads(index_map_path.read_text())
        assert manifest_json["entries"]
        assert index_map_json["entries"][0]["section"].startswith(
            "Domains.Postprocessing"
        )

        # Upload/run helpers regenerate config.json without passing arguments.
        # The last explicit postprocessing config must therefore survive a
        # later write_config() call.
        driven_sim.write_config()
        regenerated_config = json.loads(config_path.read_text())
        assert (
            regenerated_config["Boundaries"]["Postprocessing"]["SurfaceFlux"][0]["Type"]
            == "Power"
        )


# ---------------------------------------------------------------------------
# DrivenSim with inplane lumped ports
# ---------------------------------------------------------------------------


class TestDrivenSimInplanePorts:
    """DrivenSim with single-element inplane lumped ports."""

    @pytest.fixture(scope="class")
    def inplane_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh a DrivenSim with inplane lumped ports."""
        tmp_path = tmp_path_factory.mktemp("inplane")
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_port("o1", layer="metal1", length=5.0, impedance=50.0)
        sim.add_port("o2", layer="metal1", length=5.0, impedance=50.0)
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=20)
        sim.mesh(preset="coarse")
        return sim

    def test_mesh_creates_file(self, inplane_sim):
        mesh_path = Path(inplane_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()

    def test_config_has_lumped_ports(self, inplane_sim):
        inplane_sim.write_config()
        config_path = Path(inplane_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        assert "LumpedPort" in config["Boundaries"]
        assert len(config["Boundaries"]["LumpedPort"]) == 2

    def test_validate_mesh_passes(self, inplane_sim):
        result = inplane_sim.validate_mesh()
        assert result.valid, f"Mesh validation failed: {result}"


def test_validate_mesh_autogenerates_config(tmp_path, cpw_component):
    """validate_mesh should write config.json when it does not exist yet."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim-autoconfig"))
    sim.set_geometry(cpw_component)
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.add_port("o1", layer="metal1", length=5.0, impedance=50.0)
    sim.add_port("o2", layer="metal1", length=5.0, impedance=50.0)
    sim.set_driven(fmin=1e9, fmax=50e9, num_points=20)
    sim.mesh(preset="coarse")

    assert sim._output_dir is not None
    config_path = Path(sim._output_dir) / "config.json"
    if config_path.exists():
        config_path.unlink()

    result = sim.validate_mesh()
    assert result.valid, f"Mesh validation failed: {result}"
    assert config_path.exists(), "validate_mesh should auto-generate config.json"


def test_reactive_port_parameters_go_to_lumped_element(tmp_path, cpw_component):
    """Reactive parameters must go into a passive LumpedPort (Active: false).

    Driven simulations forbid L/C on excited ports in Palace.  The fix is to
    emit an additional passive port entry (Active: false) on the same boundary
    surface carrying the reactive R/L/C, while the excited port gets only R.
    """
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim-reactive"))
    sim.set_geometry(cpw_component)
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.add_port(
        "o1",
        layer="metal1",
        length=5.0,
        impedance=50.0,
        resistance=2.5,
        inductance=10e-9,
        capacitance=1e-15,
    )
    sim.add_port("o2", layer="metal1", length=5.0, impedance=50.0)
    sim.set_driven(fmin=1e9, fmax=50e9, num_points=20)
    sim.mesh(preset="coarse")
    sim.write_config()

    assert sim._output_dir is not None
    config_path = Path(sim._output_dir) / "config.json"
    config = json.loads(config_path.read_text())
    boundaries = config["Boundaries"]

    assert "LumpedPort" in boundaries
    # 2 primary ports + 1 passive reactive port
    assert len(boundaries["LumpedPort"]) == 3
    assert "LumpedElement" not in boundaries

    # Excited port 1 must have only R (no reactive terms)
    p1 = next(port for port in boundaries["LumpedPort"] if port["Index"] == 1)
    assert p1["R"] == 50.0
    assert "L" not in p1
    assert "C" not in p1

    # Passive reactive port must be Active: false and carry R/L/C
    passive = [p for p in boundaries["LumpedPort"] if not p.get("Active", True)]
    assert len(passive) == 1
    rp = passive[0]
    assert rp.get("Active") is False
    assert rp["R"] == 2.5
    assert rp["L"] == 10e-9
    assert rp["C"] == 1e-15
    assert "Attributes" in rp
    assert "Direction" in rp
    # Passive port must share the same boundary attributes as the excited port
    assert rp["Attributes"] == p1["Attributes"]


def test_run_local_direct_palace_supports_serial_wrapper_flag(tmp_path, monkeypatch):
    """Direct local Palace runs can request the wrapper's serial path."""

    sim = ElectrostaticSim()
    sim.set_output_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")
    palace_executable = tmp_path / "palace"
    palace_executable.write_text("#!/bin/sh\n", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        assert env
        commands.append(cmd)
        postpro_dir = Path(cwd) / "results" / "palace"
        postpro_dir.mkdir(parents=True, exist_ok=True)
        (postpro_dir / "terminal-C.csv").write_text("i\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = sim.run_local(
        use_apptainer=False,
        palace_executable=palace_executable,
        num_processes=1,
        num_threads=1,
        serial=True,
        verbose=False,
    )

    assert commands == [
        [
            str(palace_executable),
            "-serial",
            "-np",
            "1",
            "-nt",
            "1",
            "config.json",
        ]
    ]
    assert (
        results["terminal-C.csv"] == tmp_path / "results" / "palace" / "terminal-C.csv"
    )
    metadata = json.loads(
        (tmp_path / "metadata" / "palace_run_metadata.json").read_text()
    )
    assert metadata["schema_version"] == 1
    assert metadata["status"] == "completed"
    assert metadata["return_code"] == 0
    assert metadata["elapsed_seconds"] >= 0
    assert metadata["launcher"] == {
        "kind": "executable",
        "executable_mode": "wrapper",
        "serial": True,
        "palace_executable_configured": True,
        "palace_executable_name": "palace",
    }
    assert metadata["resources"] == {
        "num_processes": 1,
        "num_threads": 1,
        "omp_num_threads": None,
    }
    assert metadata["command"]["argv"] == [
        "palace",
        "-serial",
        "-np",
        "1",
        "-nt",
        "1",
        "config.json",
    ]
    assert metadata["outputs"]["terminal-C.csv"]["bytes"] > 0


def test_run_local_direct_palace_binary_mode_omits_wrapper_flags(tmp_path, monkeypatch):
    """Direct solver binaries receive only config.json and OMP threads via env."""

    sim = ElectrostaticSim()
    sim.set_output_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")
    palace_binary = tmp_path / "palace-arm64.bin"
    palace_binary.write_text("#!/bin/sh\n", encoding="utf-8")

    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        commands.append(cmd)
        envs.append(env)
        postpro_dir = Path(cwd) / "results" / "palace"
        postpro_dir.mkdir(parents=True, exist_ok=True)
        (postpro_dir / "terminal-C.csv").write_text("i\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = sim.run_local(
        use_apptainer=False,
        executable_mode="binary",
        palace_executable=palace_binary,
        num_processes=1,
        num_threads=2,
        verbose=False,
    )

    assert commands == [[str(palace_binary), "config.json"]]
    assert envs[0]["OMP_NUM_THREADS"] == "2"
    assert (
        results["terminal-C.csv"] == tmp_path / "results" / "palace" / "terminal-C.csv"
    )
    metadata = json.loads(
        (tmp_path / "metadata" / "palace_run_metadata.json").read_text()
    )
    assert metadata["launcher"] == {
        "kind": "executable",
        "executable_mode": "binary",
        "serial": False,
        "palace_executable_configured": True,
        "palace_executable_name": "palace-arm64.bin",
    }
    assert metadata["resources"] == {
        "num_processes": 1,
        "num_threads": 2,
        "omp_num_threads": "2",
    }
    assert metadata["command"]["argv"] == ["palace-arm64.bin", "config.json"]


def test_run_local_direct_palace_supports_setup_commands(tmp_path, monkeypatch):
    """Caller-provided setup commands can activate Palace for notebook local runs."""

    sim = ElectrostaticSim()
    sim.set_output_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        assert env["OMP_NUM_THREADS"] == "4"
        commands.append(cmd)
        postpro_dir = Path(cwd) / "results" / "palace"
        postpro_dir.mkdir(parents=True, exist_ok=True)
        (postpro_dir / "terminal-C.csv").write_text("i\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = sim.run_local(
        use_apptainer=False,
        executable_mode="binary",
        palace_executable="palace-x86_64.bin",
        setup_commands=(
            "source /opt/spack/share/spack/setup-env.sh",
            "spack load palace@0.16.0",
        ),
        num_threads=4,
        verbose=False,
    )

    assert commands[0][:2] == ["/bin/bash", "-lc"]
    script = commands[0][2]
    assert "source /opt/spack/share/spack/setup-env.sh" in script
    assert "spack load palace@0.16.0" in script
    assert "command -v palace-x86_64.bin >/dev/null" in script
    assert "exec palace-x86_64.bin config.json" in script
    assert (
        results["terminal-C.csv"] == tmp_path / "results" / "palace" / "terminal-C.csv"
    )
    metadata = json.loads(
        (tmp_path / "metadata" / "palace_run_metadata.json").read_text()
    )
    assert metadata["launcher"] == {
        "kind": "executable",
        "executable_mode": "binary",
        "serial": False,
        "palace_executable_configured": True,
        "palace_executable_name": "palace-x86_64.bin",
    }
    assert metadata["command"]["argv"] == ["palace-x86_64.bin", "config.json"]


def test_run_local_direct_palace_binary_mode_rejects_multi_process(tmp_path):
    """Direct solver binaries do not accept the wrapper's -np launcher flag."""

    sim = ElectrostaticSim()
    sim.set_output_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")
    palace_binary = tmp_path / "palace-arm64.bin"
    palace_binary.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(ValueError, match="single-process"):
        sim.run_local(
            use_apptainer=False,
            executable_mode="binary",
            palace_executable=palace_binary,
            num_processes=2,
            verbose=False,
        )


def test_run_local_direct_palace_rejects_unknown_executable_mode(tmp_path):
    sim = ElectrostaticSim()
    sim.set_output_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "palace.msh").write_text("$MeshFormat\n", encoding="utf-8")

    with pytest.raises(ValueError, match="executable_mode"):
        sim.run_local(
            use_apptainer=False,
            executable_mode="unknown",  # type: ignore[arg-type]
            palace_executable="/usr/bin/true",
            verbose=False,
        )


# ---------------------------------------------------------------------------
# EigenmodeSim workflow
# ---------------------------------------------------------------------------


class TestEigenmodeSimWorkflow:
    """End-to-end EigenmodeSim: configure -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def eigenmode_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh an EigenmodeSim."""
        tmp_path = tmp_path_factory.mktemp("eigenmode")
        sim = EigenmodeSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.set_eigenmode(num_modes=5, target=50e9)
        sim.mesh(preset="coarse")
        return sim

    def test_mesh_creates_file(self, eigenmode_sim):
        mesh_path = Path(eigenmode_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()

    def test_config_json_type(self, eigenmode_sim):
        eigenmode_sim.write_config()
        config_path = Path(eigenmode_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        assert config["Problem"]["Type"] == "Eigenmode"

    def test_config_has_eigenmode_solver(self, eigenmode_sim):
        """Config must contain Eigenmode solver section."""
        eigenmode_sim.write_config()
        config_path = Path(eigenmode_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        solver = config["Solver"]
        assert "Eigenmode" in solver

    def test_validate_mesh_passes(self, eigenmode_sim):
        result = eigenmode_sim.validate_mesh()
        assert result.valid, f"Mesh validation failed: {result}"

    def test_config_has_floquet_periodic_boundary(self, tmp_path, cpw_component):
        """Floquet in eigenmode emits Palace Periodic boundary section."""
        sim = EigenmodeSim()
        sim.set_output_dir(str(tmp_path / "palace-sim-floquet"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.set_eigenmode(
            num_modes=5,
            target=50e9,
            floquet=True,
            phi_target=1.57,
            n_eff_guess=2.2,
        )
        sim.mesh(preset="coarse", periodic_axis="x")
        sim.write_config()

        assert sim._output_dir is not None
        config_path = Path(sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        periodic = config["Boundaries"]["Periodic"]

        assert len(periodic["FloquetWaveVector"]) == 3
        assert periodic["FloquetWaveVector"][0] > 0
        assert periodic["FloquetWaveVector"][1] == pytest.approx(0.0)
        assert periodic["FloquetWaveVector"][2] == pytest.approx(0.0)
        assert len(periodic["BoundaryPairs"]) == 1
        pair = periodic["BoundaryPairs"][0]
        assert len(pair["DonorAttributes"]) > 0
        assert len(pair["ReceiverAttributes"]) > 0


# ---------------------------------------------------------------------------
# ElectrostaticSim workflow
# ---------------------------------------------------------------------------


class TestElectrostaticSimWorkflow:
    """End-to-end ElectrostaticSim: configure -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def electrostatic_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh an ElectrostaticSim with terminals."""
        tmp_path = tmp_path_factory.mktemp("electrostatic")
        sim = ElectrostaticSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_terminal("T1", layer="metal1")
        sim.add_terminal("T2", layer="metal1")
        sim.set_electrostatic()
        sim.mesh(preset="coarse")
        return sim

    def test_mesh_creates_file(self, electrostatic_sim):
        mesh_path = Path(electrostatic_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()

    def test_write_config_generates_valid_json(self, electrostatic_sim):
        """Electrostatic config generation produces valid Palace JSON."""
        electrostatic_sim.write_config()
        config_path = Path(electrostatic_sim._output_dir) / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["Problem"]["Type"] == "Electrostatic"
        assert "Electrostatic" in config["Solver"]
        boundaries = config["Boundaries"]
        assert "Terminal" in boundaries
        assert len(boundaries["Terminal"]) == 2
        # Each terminal must have at least one attribute (regression: planar
        # conductor terminals were silently grounded when the loop only
        # scanned conductor_surfaces).
        for term in boundaries["Terminal"]:
            assert term["Attributes"], (
                f"Terminal {term['Index']} has no attributes — "
                "likely a layer/surface lookup miss"
            )
        assert "LumpedPort" not in boundaries
        assert "WavePort" not in boundaries

        index_map_path = (
            Path(electrostatic_sim._output_dir) / "metadata" / "palace_index_map.json"
        )
        assert index_map_path.exists()
        index_map = json.loads(index_map_path.read_text())
        terminal_rows = [
            row
            for row in index_map["entries"]
            if row["section"] == "Boundaries.Terminal"
        ]
        assert {row["index"] for row in terminal_rows} == {1, 2}
        assert {row["terminal_name"] for row in terminal_rows} == {"T1", "T2"}

    def test_planar_conductor_terminal_not_grounded(
        self, tmp_path_factory, cpw_component
    ):
        """Terminals on planar (thin) conductor layers must be picked up.

        Regression: the terminal loop previously only scanned
        ``conductor_surfaces`` (thick metals with shell surfaces) and
        seeded ``ground_attrs`` with ``pec_attrs``. A terminal defined on
        a planar layer therefore ended up with empty attributes and the
        layer's PG was sent to Ground instead.
        """
        tmp_path = tmp_path_factory.mktemp("electrostatic_planar")
        sim = ElectrostaticSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_terminal("T1", layer="metal1")
        sim.add_terminal("T2", layer="metal1")
        sim.set_electrostatic()
        sim.mesh(preset="coarse", planar_conductors=True)
        sim.write_config()

        output_dir = sim._output_dir
        assert output_dir is not None
        config = json.loads((Path(output_dir) / "config.json").read_text())
        boundaries = config["Boundaries"]
        terminal_attrs: set[int] = set()
        for term in boundaries["Terminal"]:
            assert term["Attributes"], "Planar-conductor terminal has no attributes"
            terminal_attrs.update(term["Attributes"])
        # And the terminal's PG must not also appear in Ground.
        ground_attrs = set(boundaries.get("Ground", {}).get("Attributes", []))
        assert terminal_attrs.isdisjoint(ground_attrs), (
            f"Terminal attrs {terminal_attrs} overlap Ground {ground_attrs}"
        )

    def test_same_layer_planar_terminals_can_select_islands_by_center(
        self,
        tmp_path_factory,
    ):
        """Center selectors split same-layer PEC islands into distinct terminals."""
        gf.gpdk.PDK.activate()
        component = gf.Component()
        left = component << gf.c.rectangle(
            (30, 20),
            centered=True,
            layer=gf.gpdk.LAYER.M1,
        )
        left.movex(-35)
        right = component << gf.c.rectangle(
            (30, 20),
            centered=True,
            layer=gf.gpdk.LAYER.M1,
        )
        right.movex(35)

        tmp_path = tmp_path_factory.mktemp("electrostatic_same_layer")
        sim = ElectrostaticSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_terminal("left", layer="metal1", center=(-35, 0))
        sim.add_terminal("right", layer="metal1", center=(35, 0))
        sim.set_electrostatic()
        sim.mesh(preset="coarse", planar_conductors=True)
        sim.write_config()

        output_dir = sim._output_dir
        assert output_dir is not None
        config = json.loads((Path(output_dir) / "config.json").read_text())
        terminals = config["Boundaries"]["Terminal"]
        assert len(terminals) == 2
        assert terminals[0]["Attributes"]
        assert terminals[1]["Attributes"]
        assert set(terminals[0]["Attributes"]).isdisjoint(terminals[1]["Attributes"])

        index_map = json.loads(
            (Path(output_dir) / "metadata" / "palace_index_map.json").read_text()
        )
        terminal_rows = [
            row
            for row in index_map["entries"]
            if row["section"] == "Boundaries.Terminal"
        ]
        assert {row["terminal_name"] for row in terminal_rows} == {"left", "right"}
        assert {row["role"] for row in terminal_rows} == {"pec_surface"}


# ---------------------------------------------------------------------------
# MagnetostaticSim workflow
# ---------------------------------------------------------------------------


class TestMagnetostaticSimWorkflow:
    """End-to-end MagnetostaticSim: configure -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def magnetostatic_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh a MagnetostaticSim with current sources."""
        tmp_path = tmp_path_factory.mktemp("magnetostatic")
        sim = MagnetostaticSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0)
        sim.set_airbox(margin_x=50.0, margin_y=50.0, z_above=100.0, z_below=20.0)
        sim.add_current_source(
            "signal",
            layer="metal1",
            center=(0, 0),
            direction=[1.0, 0.0, 0.0],
            coordinate_system="Cartesian",
        )
        sim.add_current_source(
            "return",
            layer="metal1",
            center=(0, 31),
            direction="-X",
        )
        sim.set_magnetostatic(save_fields=1)
        sim.mesh(preset="coarse", planar_conductors=True)
        return sim

    def test_write_config_generates_magnetostatic_boundaries(self, magnetostatic_sim):
        """Magnetostatic config emits Palace sources and magnetic flux rows."""
        magnetostatic_sim.write_config()
        config_path = Path(magnetostatic_sim._output_dir) / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["Problem"]["Type"] == "Magnetostatic"
        assert config["Solver"]["Magnetostatic"]["Save"] == 1

        boundaries = config["Boundaries"]
        assert "SurfaceCurrent" in boundaries
        assert len(boundaries["SurfaceCurrent"]) == 2
        directions = [entry["Direction"] for entry in boundaries["SurfaceCurrent"]]
        assert [1.0, 0.0, 0.0] in directions
        assert "-X" in directions
        assert boundaries["SurfaceCurrent"][0]["CoordinateSystem"] == "Cartesian"
        for entry in boundaries["SurfaceCurrent"]:
            assert entry["Attributes"], "SurfaceCurrent source has no attributes"
        assert set(boundaries["SurfaceCurrent"][0]["Attributes"]).isdisjoint(
            boundaries["SurfaceCurrent"][1]["Attributes"]
        )
        assert "PMC" in boundaries
        assert boundaries["PMC"]["Attributes"]

        flux_entries = boundaries["Postprocessing"]["SurfaceFlux"]
        assert len(flux_entries) == 2
        assert {entry["Type"] for entry in flux_entries} == {"Magnetic"}
        assert {entry["TwoSided"] for entry in flux_entries} == {False}

        assert "LumpedPort" not in boundaries
        assert "WavePort" not in boundaries
        assert "Terminal" not in boundaries

    def test_write_config_generates_source_index_map(self, magnetostatic_sim):
        """Magnetostatic source indices map back to manifest physical names."""
        magnetostatic_sim.write_config()
        index_map_path = (
            Path(magnetostatic_sim._output_dir) / "metadata" / "palace_index_map.json"
        )
        assert index_map_path.exists()
        index_map = json.loads(index_map_path.read_text())
        source_rows = [
            row
            for row in index_map["entries"]
            if row["section"] == "Boundaries.SurfaceCurrent"
        ]
        flux_rows = [
            row
            for row in index_map["entries"]
            if row["section"] == "Boundaries.Postprocessing.SurfaceFlux"
        ]
        assert {row["index"] for row in source_rows} == {1, 2}
        assert {row["current_source_name"] for row in source_rows} == {
            "signal",
            "return",
        }
        assert {row["role"] for row in source_rows} == {"pec_surface"}
        assert {row["index"] for row in flux_rows} == {1, 2}
        assert {row["Type"] for row in flux_rows} == {"Magnetic"}

    def test_manifest_postprocessing_preserves_magnetic_flux_rows(
        self, magnetostatic_sim
    ):
        """Manifest postprocessing must not erase solver-owned flux output."""
        postprocessing = build_postprocessing_config_from_manifest(
            magnetostatic_sim._last_mesh_result.manifest,
            include_empty_sections=False,
        )

        config_path = magnetostatic_sim.write_config(postprocessing=postprocessing)
        config = json.loads(config_path.read_text())

        energy_rows = config["Domains"]["Postprocessing"]["Energy"]
        assert energy_rows

        flux_rows = config["Boundaries"]["Postprocessing"]["SurfaceFlux"]
        assert len(flux_rows) == 2
        assert {row["Type"] for row in flux_rows} == {"Magnetic"}

        index_map = json.loads(
            (
                Path(magnetostatic_sim._output_dir)
                / "metadata"
                / "palace_index_map.json"
            ).read_text()
        )
        sections = {row["section"] for row in index_map["entries"]}
        assert "Domains.Postprocessing.Energy" in sections
        assert "Boundaries.Postprocessing.SurfaceFlux" in sections

    def test_write_config_supports_multielement_current_source(
        self, tmp_path, cpw_component
    ):
        """Multielement current sources emit Palace Elements and flux rows."""
        sim = MagnetostaticSim()
        sim.set_output_dir(str(tmp_path / "multielement"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0)
        sim.set_airbox(margin_x=50.0, margin_y=50.0, z_above=100.0, z_below=20.0)
        sim.add_current_source(
            "loop",
            elements=(
                {
                    "layer": "metal1",
                    "center": (0, 0),
                    "direction": "+X",
                },
                {
                    "layer": "metal1",
                    "center": (0, 31),
                    "direction": [0.0, -1.0, 0.0],
                    "coordinate_system": "Cartesian",
                },
            ),
        )
        sim.mesh(preset="coarse", planar_conductors=True)
        config_path = sim.write_config()
        config = json.loads(config_path.read_text())

        surface_current = config["Boundaries"]["SurfaceCurrent"][0]
        assert surface_current["Index"] == 1
        assert "Attributes" not in surface_current
        assert len(surface_current["Elements"]) == 2
        assert surface_current["Elements"][0]["Direction"] == "+X"
        assert surface_current["Elements"][1]["Direction"] == [0.0, -1.0, 0.0]
        assert surface_current["Elements"][1]["CoordinateSystem"] == "Cartesian"
        assert set(surface_current["Elements"][0]["Attributes"]).isdisjoint(
            surface_current["Elements"][1]["Attributes"]
        )

        flux = config["Boundaries"]["Postprocessing"]["SurfaceFlux"][0]
        assert flux["Index"] == 1
        assert flux["Type"] == "Magnetic"
        assert set(flux["Attributes"]) == set(
            surface_current["Elements"][0]["Attributes"]
            + surface_current["Elements"][1]["Attributes"]
        )

        index_map = json.loads(
            (Path(sim._output_dir) / "metadata" / "palace_index_map.json").read_text()
        )
        source_rows = [
            row
            for row in index_map["entries"]
            if row["section"] == "Boundaries.SurfaceCurrent"
        ]
        assert {row["current_source_name"] for row in source_rows} == {"loop"}
        assert {row["current_source_element_count"] for row in source_rows} == {2}
        assert {row["current_source_element_index"] for row in source_rows} == {1, 2}
        assert any(row["Direction"] == "+X" for row in source_rows)
        assert any(row["Direction"] == [0.0, -1.0, 0.0] for row in source_rows)
        assert {
            row["CoordinateSystem"]
            for row in source_rows
            if row.get("CoordinateSystem") is not None
        } == {"Cartesian"}

    def test_write_config_rejects_overlapping_multielement_current_source(
        self, tmp_path, cpw_component
    ):
        """Multielement source selectors must not claim the same attribute."""
        sim = MagnetostaticSim()
        sim.set_output_dir(str(tmp_path / "overlap"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0)
        sim.set_airbox(margin_x=50.0, margin_y=50.0, z_above=100.0, z_below=20.0)
        sim.add_current_source(
            "loop",
            elements=(
                {
                    "layer": "metal1",
                    "center": (0, 0),
                    "direction": "+X",
                },
                {
                    "layer": "metal1",
                    "center": (0, 0),
                    "direction": "-X",
                },
            ),
        )
        sim.mesh(preset="coarse", planar_conductors=True)

        with pytest.raises(ValueError, match="already selected"):
            sim.write_config()


# ---------------------------------------------------------------------------
# Validation error tests (no gmsh needed)
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Test validation catches configuration errors (no gmsh needed)."""

    def test_missing_geometry(self):
        sim = DrivenSim()
        result = sim.validate_config()
        assert not result.valid

    def test_missing_output_dir(self):
        """Missing output_dir produces a warning (not an error)."""
        sim = DrivenSim()
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(air_above=300.0)
        result = sim.validate_config()
        # validate_config warns about missing ports, but output_dir is
        # checked at mesh() time, not validation time
        assert result.valid

    def test_driven_valid_config(self, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "test"))
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        result = sim.validate_config()
        assert result.valid, f"Validation failed: {result}"

    def test_eigenmode_no_ports_ok(self, tmp_path):
        """Eigenmode sims don't require ports."""
        sim = EigenmodeSim()
        sim.set_output_dir(str(tmp_path / "test"))
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(air_above=300.0)
        sim.set_eigenmode(num_modes=5)
        result = sim.validate_config()
        assert result.valid, f"Validation failed: {result}"

    def test_electrostatic_needs_terminals(self, tmp_path):
        """Electrostatic sims need at least one terminal."""
        sim = ElectrostaticSim()
        sim.set_output_dir(str(tmp_path / "test"))
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(air_above=300.0)
        result = sim.validate_config()
        assert not result.valid


# ---------------------------------------------------------------------------
# Material overrides
# ---------------------------------------------------------------------------


class TestMaterialOverrides:
    """Test material override configuration flows through to config."""

    @pytest.fixture(scope="class")
    def material_sim(self, tmp_path_factory, cpw_component):
        """Create a DrivenSim with material overrides."""
        tmp_path = tmp_path_factory.mktemp("materials")
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.set_material("aluminum", conductivity=3.77e7)
        sim.mesh(preset="coarse")
        return sim

    def test_material_override_in_config(self, material_sim):
        material_sim.write_config()
        config_path = Path(material_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        # The material should appear in Domains.Materials
        assert "Domains" in config


# ---------------------------------------------------------------------------
# Numerical solver overrides
# ---------------------------------------------------------------------------


class TestNumericalConfig:
    """Test numerical solver configuration."""

    def test_set_numerical(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "numerical"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.set_numerical(order=2, tolerance=1e-8, max_iterations=600)

        assert sim.numerical.order == 2
        assert sim.numerical.tolerance == 1e-8
        assert sim.numerical.max_iterations == 600

    def test_numerical_settings_flow_to_config(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "numerical-config"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.set_numerical(
            order=3,
            tolerance=2e-7,
            max_iterations=777,
            solver_type="Default",
            preconditioner="AMS",
            device="CPU",
        )

        sim.mesh(preset="coarse")
        sim.write_config()
        assert sim._output_dir is not None
        config_path = sim._output_dir / "config.json"
        config = json.loads(config_path.read_text())

        linear = config["Solver"]["Linear"]
        assert linear["Type"] == "Default"
        assert linear["KSPType"] == "GMRES"
        assert linear["Tol"] == 2e-7
        assert linear["MaxIts"] == 777
        assert linear["Preconditioner"] == "AMS"
        assert config["Solver"]["Order"] == 3
        assert config["Solver"]["Device"] == "CPU"

    def test_write_config_merges_solver_hints(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "solver-hints"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.mesh(preset="coarse")
        sim.write_config(hints={"Solver": {"Device": "GPU", "Backend": "/gpu/cuda"}})
        assert sim._output_dir is not None
        config = json.loads((sim._output_dir / "config.json").read_text())

        assert config["Solver"]["Device"] == "GPU"
        assert config["Solver"]["Backend"] == "/gpu/cuda"
        assert config["Solver"]["Linear"]["Type"] == "Default"
        assert config["Solver"]["Driven"]["Samples"]

    def test_mumps_solver_config_defaults(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "numerical-mumps"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.set_numerical(solver_type="MUMPS", tolerance=1e-8)

        sim.mesh(preset="coarse")
        sim.write_config()
        assert sim._output_dir is not None
        config_path = sim._output_dir / "config.json"
        config = json.loads(config_path.read_text())

        linear = config["Solver"]["Linear"]
        assert linear["Type"] == "MUMPS"
        assert linear["KSPType"] == "GMRES"
        assert linear["Tol"] == 1e-8
        assert linear["MaxIts"] == 1
        assert linear["MGMaxLevels"] == 1
        assert linear["EstimatorMaxIts"] == 0
        assert linear["EstimatorTol"] == 1e-6
        assert linear["DivFreeTol"] == 1e-6
        assert linear["DivFreeMaxIts"] == 0
        assert linear["PCMatReal"] is False
        assert linear["ComplexCoarseSolve"] is True
