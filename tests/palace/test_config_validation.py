"""Tests for versioned Palace config schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gsim.common.stack import LayerStack
from gsim.palace import DrivenSim
from gsim.palace.config_validation import (
    PalaceConfigValidationError,
    palace_config_schema,
    validate_palace_config,
)
from gsim.palace.mesh import config_generator
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import TerminalConfig
from gsim.palace.models.versions import (
    SUPPORTED_PALACE_CONFIG_VERSIONS,
    normalize_palace_config_version,
)


def _minimal_groups() -> dict:
    return {
        "volumes": {"air": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }


def _structured_conductor_groups() -> dict:
    """Return current SGB MA/MS attributes split across whole components."""

    def surface(component_id: str | None, net_id: str | None, attribute: int) -> dict:
        return {
            "phys_group": attribute,
            "source": "volume_interface",
            "solver_use": "solver_active",
            "interface_type": "MA",
            "sgb_record": "final_physical_group",
            "conductor_component_id": component_id,
            "net_id": net_id,
        }

    return {
        "volumes": {"AIR": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {
            "signal_top": surface("COMP__SIGNAL", "M1@signal", 11),
            "signal_side": surface("COMP__SIGNAL", "M1@signal", 12),
            "floating": surface("COMP__FLOATING", None, 13),
        },
    }


def test_supported_palace_schema_snapshots_load() -> None:
    for version in SUPPORTED_PALACE_CONFIG_VERSIONS:
        schema = palace_config_schema(version)
        assert sorted(schema["properties"]) == [
            "Boundaries",
            "Domains",
            "Model",
            "Problem",
            "Solver",
        ]


def test_unsupported_palace_config_version_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unsupported Palace config version"):
        normalize_palace_config_version("0.14.0")


def _write_minimal_config(
    tmp_path: Path,
    sim: DrivenSim,
) -> dict:
    config_path = generate_palace_config(
        groups=_minimal_groups(),
        ports=[],
        port_info=[],
        stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        absorbing_boundary=False,
        numerical_config=sim.numerical,
        refinement_config=sim.refinement,
        palace_version=sim.palace_version,
        problem_output_formats=sim.output_formats or None,
    )
    return json.loads(config_path.read_text())


def test_set_refinement_writes_schema_valid_fragment(tmp_path: Path) -> None:
    sim = DrivenSim()
    sim.set_refinement(max_its=4, tol=1.0e-3, uniform_levels=1)

    config = _write_minimal_config(tmp_path, sim)
    refinement = config["Model"]["Refinement"]
    assert refinement["MaxIts"] == 4
    assert refinement["Tol"] == 1.0e-3
    assert refinement["UniformLevels"] == 1


def test_set_linear_solver_extra_uses_schema_gate(tmp_path: Path) -> None:
    sim = DrivenSim()
    sim.set_linear_solver(
        type="AMS",
        tol=1.0e-8,
        mg_max_levels=3,
        mg_cycle_its=1,
        mg_smooth_its=1,
        div_free_tol=1.0e-7,
        div_free_max_its=4,
        estimator_tol=1.0e-6,
        estimator_max_its=2,
        estimator_mg=True,
        ams_max_its=2,
    )

    config = _write_minimal_config(tmp_path, sim)
    linear = config["Solver"]["Linear"]
    assert linear["Type"] == "AMS"
    assert linear["Tol"] == 1.0e-8
    assert linear["MGMaxLevels"] == 3
    assert linear["MGCycleIts"] == 1
    assert linear["MGSmoothIts"] == 1
    assert linear["DivFreeTol"] == 1.0e-7
    assert linear["DivFreeMaxIts"] == 4
    assert linear["EstimatorTol"] == 1.0e-6
    assert linear["EstimatorMaxIts"] == 2
    assert linear["EstimatorMG"] is True
    assert linear["AMSMaxIts"] == 2

    sim.set_palace_version("0.15.0")
    with pytest.raises(PalaceConfigValidationError, match=r"\$\.Solver\.Linear"):
        _write_minimal_config(tmp_path / "v015", sim)


def test_set_output_formats_writes_schema_valid_fragment(tmp_path: Path) -> None:
    sim = DrivenSim()
    sim.set_output_formats(paraview=True, grid_function=True)

    config = _write_minimal_config(tmp_path, sim)
    assert config["Problem"]["OutputFormats"] == {
        "Paraview": True,
        "GridFunction": True,
    }


def test_generated_config_validates_against_supported_versions(tmp_path: Path) -> None:
    for version in SUPPORTED_PALACE_CONFIG_VERSIONS:
        output_dir = tmp_path / version
        config_path = generate_palace_config(
            groups=_minimal_groups(),
            ports=[],
            port_info=[],
            stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
            output_path=output_dir,
            model_name="palace",
            fmax=10e9,
            absorbing_boundary=False,
            palace_version=version,
        )
        validate_palace_config(
            json.loads(config_path.read_text()),
            palace_version=version,
        )


def test_schema_validation_reports_json_path(tmp_path: Path) -> None:
    config_path = generate_palace_config(
        groups=_minimal_groups(),
        ports=[],
        port_info=[],
        stack=LayerStack(materials={"air": {"permittivity": 1.0}}),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        absorbing_boundary=False,
    )
    config = json.loads(config_path.read_text())
    config["Solver"]["Linear"]["AMSMaxIts"] = 2

    with pytest.raises(PalaceConfigValidationError, match=r"\$\.Solver\.Linear"):
        validate_palace_config(config, palace_version="0.15.0")


def test_structured_electrostatic_boundaries_assign_whole_components():
    """Exact terminal nets select whole components; unmatched ones become ground."""
    boundaries = config_generator._sgb_electrostatic_boundaries(
        _structured_conductor_groups(),
        [
            TerminalConfig(
                name="signal",
                layer="M1",
                physical_label="signal",
                center=(0.0, 0.0),
            )
        ],
    )

    assert boundaries == {
        "Terminal": [{"Index": 1, "Attributes": [11, 12]}],
        "Ground": {"Attributes": [13]},
    }


def test_structured_electrostatic_rejects_duplicate_or_conflicting_authority():
    """Components cannot mix nets or share solver-live attributes."""
    groups = _structured_conductor_groups()
    groups["boundary_surfaces"]["signal_side"]["net_id"] = "M1@other"
    with pytest.raises(ValueError, match="conflicting terminal nets"):
        config_generator._sgb_electrostatic_boundaries(groups, [])

    groups = _structured_conductor_groups()
    groups["boundary_surfaces"]["floating"]["phys_group"] = 11
    with pytest.raises(ValueError, match="multiple conductor components"):
        config_generator._sgb_electrostatic_boundaries(groups, [])


def test_structured_electrostatic_missing_component_fails_before_legacy_assignment():
    """A broken MA/MS SGB record cannot fall through to native selector logic."""
    groups = _structured_conductor_groups()
    groups["boundary_surfaces"]["signal_top"]["conductor_component_id"] = None

    assert config_generator._has_sgb_structured_conductor_surfaces(groups)
    with pytest.raises(ValueError, match="lacks conductor_component_id"):
        config_generator._sgb_electrostatic_boundaries(groups, [])


def test_structured_electrostatic_config_has_only_solution_materials_and_no_ports(
    tmp_path: Path,
) -> None:
    """Structured conductor shells become boundaries, never PEC material volumes."""
    groups = _structured_conductor_groups()
    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=LayerStack(materials={"AIR": {"permittivity": 1.0}}),
        output_path=tmp_path,
        model_name="structured",
        fmax=1e9,
        simulation_type="electrostatic",
        terminals=[
            TerminalConfig(
                name="signal",
                layer="M1",
                physical_label="signal",
                center=(0.0, 0.0),
            )
        ],
        validate_schema=False,
    )
    config = json.loads(config_path.read_text())

    assert config["Domains"]["Materials"] == [
        {"Attributes": [1], "LossTan": 0.0, "Permittivity": 1.0}
    ]
    assert config["Boundaries"]["Terminal"] == [{"Attributes": [11, 12], "Index": 1}]
    assert config["Boundaries"]["Ground"] == {"Attributes": [13]}
    assert config["Boundaries"].get("LumpedPort", []) == []
    assert config["Boundaries"].get("WavePort", []) == []
