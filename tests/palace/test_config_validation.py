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
from gsim.palace.mesh.config_generator import generate_palace_config
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
