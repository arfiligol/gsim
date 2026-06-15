"""Tests for Palace material resolution with frequency-dependent dispersion."""

from __future__ import annotations

import pytest
from scipy.constants import c as C0  # noqa: N812

from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace.materials import (
    resolve_palace_materials_at_frequency,
    resolve_palace_materials_with_report,
)
from gsim.palace.models import DrivenConfig


class TestDrivenConfigCenterFrequency:
    def test_center_frequency_band(self):
        d = DrivenConfig(fmin=1e9, fmax=100e9)
        assert d.center_frequency == pytest.approx(50.5e9)

    def test_center_frequency_single_point(self):
        d = DrivenConfig(fmin=10e9, fmax=10e9)
        assert d.center_frequency == pytest.approx(10e9)

    def test_single_freq_mode(self):
        d = DrivenConfig(fmin=50e9, fmax=50e9, num_points=1)
        assert d.fmin == d.fmax
        assert d.center_frequency == pytest.approx(50e9)


class TestSetDrivenSingleFreq:
    def test_f_sets_fmin_fmax(self):
        from gsim.palace import DrivenSim

        sim = DrivenSim()
        sim.set_driven(f=50e9)
        assert sim.driven.fmin == 50e9
        assert sim.driven.fmax == 50e9
        assert sim.driven.num_points == 1
        assert sim.driven.center_frequency == pytest.approx(50e9)

    def test_fmin_fmax_override_f(self):
        from gsim.palace import DrivenSim

        sim = DrivenSim()
        sim.set_driven(f=50e9, fmin=1e9, fmax=100e9, num_points=40)
        assert sim.driven.fmin == 1e9
        assert sim.driven.fmax == 100e9
        assert sim.driven.num_points == 40


class TestResolvePalaceMaterialsAtFrequency:
    def test_sio2_at_optical_frequency(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "SiO2" in resolved
        assert resolved["SiO2"]["permittivity"] == pytest.approx(2.085, abs=0.01)

    def test_sio2_at_rf_frequency(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = 5e9
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "SiO2" in resolved
        assert resolved["SiO2"]["permittivity"] == pytest.approx(4.1)

    def test_silicon_at_optical_frequency(self):
        materials = {"silicon": MATERIALS_DB["silicon"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "silicon" in resolved
        n_sq = resolved["silicon"]["permittivity"]
        n = n_sq**0.5
        assert 3.4 < n < 3.6

    def test_conductor_unchanged(self):
        materials = {"aluminum": MATERIALS_DB["aluminum"].to_dict()}
        resolved = resolve_palace_materials_at_frequency(materials, 5e9)
        assert "aluminum" in resolved
        assert resolved["aluminum"]["conductivity"] == 3.77e7

    def test_unknown_material_preserved(self):
        materials = {"custom_mat": {"permittivity": 5.0}}
        resolved = resolve_palace_materials_at_frequency(materials, 5e9)
        assert "custom_mat" in resolved
        assert resolved["custom_mat"]["permittivity"] == 5.0

    def test_material_overlay_overrides_stack_material_values(self):
        materials = {"Si": {"permittivity": 11.9, "conductivity": 2.0}}
        overlay = {
            "materials": {
                "Si": {
                    "relative_permittivity": 11.45,
                    "loss_tangent": 1.0e-6,
                    "dispersion_models": [
                        {
                            "type": "constant",
                            "permittivity": 11.45,
                            "validity_frequency": [0, 10e9],
                            "source": "test PDK",
                        }
                    ],
                }
            }
        }

        resolved = resolve_palace_materials_at_frequency(
            materials,
            5e9,
            material_overlay=overlay,
        )

        assert resolved["Si"]["permittivity"] == pytest.approx(11.45)
        assert resolved["Si"]["loss_tangent"] == pytest.approx(1.0e-6)
        assert resolved["Si"]["conductivity"] == pytest.approx(2.0)
        assert materials["Si"]["permittivity"] == pytest.approx(11.9)
        assert materials["Si"]["conductivity"] == pytest.approx(2.0)

    def test_material_overlay_report_records_model_source_and_validity(self):
        materials = {"Si": {"permittivity": 11.9, "conductivity": 2.0}}
        overlay = {
            "materials": {
                "Si": {
                    "relative_permittivity": 11.45,
                    "loss_tangent": 1.0e-6,
                    "dispersion_models": [
                        {
                            "type": "constant",
                            "permittivity": 11.45,
                            "validity_frequency": [0, 10e9],
                            "source": "test PDK",
                        }
                    ],
                }
            }
        }

        resolved, report = resolve_palace_materials_with_report(
            materials,
            5e9,
            material_overlay=overlay,
        )

        row = report["materials"][0]
        assert resolved["Si"]["permittivity"] == pytest.approx(11.45)
        assert row["stack_material_name"] == "Si"
        assert row["matched_material_name"] == "Si"
        assert row["evaluation_frequency_hz"] == pytest.approx(5e9)
        assert row["model_type"] == "constant"
        assert row["model_source"] == "test PDK"
        assert row["within_validity"] is True
        assert row["effective_material"]["loss_tangent"] == pytest.approx(1.0e-6)

    def test_material_overlay_report_expands_material_aliases(self):
        materials = {
            "air": {"permittivity": 1.0, "loss_tangent": 0.0},
            "silicon": {"permittivity": 11.9, "conductivity": 2.0},
        }
        overlay = {
            "materials": {
                "vacuum": {
                    "relative_permittivity": 1.0,
                    "permeability": 1.0,
                    "dispersion_models": [
                        {
                            "type": "constant",
                            "permittivity": 1.0,
                            "source": "test PDK vacuum",
                            "validity_frequency": [0, 20e9],
                        }
                    ],
                },
                "Si": {
                    "relative_permittivity": 11.45,
                    "permeability": 1.0,
                    "dispersion_models": [
                        {
                            "type": "constant",
                            "permittivity": 11.45,
                            "source": "test PDK silicon",
                            "validity_frequency": [0, 20e9],
                        }
                    ],
                },
            },
            "material_aliases": {"air": "vacuum", "silicon": "Si"},
        }

        resolved, report = resolve_palace_materials_with_report(
            materials,
            5e9,
            material_overlay=overlay,
        )

        rows = {row["stack_material_name"]: row for row in report["materials"]}
        assert resolved["air"]["permeability"] == pytest.approx(1.0)
        assert rows["air"]["matched_material_name"] == "air"
        assert rows["air"]["model_source"] == "test PDK vacuum"
        assert rows["silicon"]["matched_material_name"] == "silicon"
        assert rows["silicon"]["model_source"] == "test PDK silicon"
        assert resolved["silicon"]["permittivity"] == pytest.approx(11.45)

    def test_material_overlay_preserves_unknown_materials(self):
        materials = {"custom_mat": {"permittivity": 5.0}}
        overlay = {"materials": {"Si": {"relative_permittivity": 11.45}}}

        resolved = resolve_palace_materials_at_frequency(
            materials,
            5e9,
            material_overlay=overlay,
        )

        assert resolved["custom_mat"]["permittivity"] == pytest.approx(5.0)

    def test_empty_materials(self):
        resolved = resolve_palace_materials_at_frequency({}, 5e9)
        assert resolved == {}

    def test_preserves_nonoptical_fields(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert resolved["SiO2"]["permittivity"] is not None

    def test_does_not_mutate_input(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        original_permittivity = materials["SiO2"]["permittivity"]
        resolve_palace_materials_at_frequency(materials, 5e9)
        assert materials["SiO2"]["permittivity"] == original_permittivity

    def test_sapphire_anisotropic_resolved(self):
        materials = {"sapphire": MATERIALS_DB["sapphire"].to_dict()}
        freq_hz = C0 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "sapphire" in resolved
        assert isinstance(resolved["sapphire"]["permittivity"], list)
