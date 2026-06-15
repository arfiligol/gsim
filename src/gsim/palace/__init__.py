"""Palace API module for EM simulation with gdsfactory.

This module provides a comprehensive API for setting up and running
electromagnetic simulations using the Palace solver with gdsfactory components.

Features:
    - Problem-specific simulation classes
      (DrivenSim, EigenmodeSim, ElectrostaticSim, MagnetostaticSim)
    - Layer stack extraction from PDK
    - Port configuration (inplane, via, CPW)
    - Mesh generation with quality presets
    - Palace config file generation

Usage:
    from gsim.palace import DrivenSim

    # Create and configure simulation
    sim = DrivenSim()
    sim.set_geometry(component)
    sim.set_stack()
    sim.set_airbox(margin_x=120.0, margin_above=120.0, margin_below=20.0)
    sim.add_cpw_port("o1", layer="topmetal2", s_width=10, gap_width=6)
    sim.set_driven(fmin=1e9, fmax=100e9)

    # Generate mesh and run
    sim.set_output_dir("./sim")
    sim.mesh(preset="fine")
    results = sim.run()
"""

from __future__ import annotations

from gsim.gcloud import RunResult, register_result_parser

# New simulation classes (composition, no inheritance)
from gsim.palace.driven import DrivenSim
from gsim.palace.eigenmode import EigenmodeSim
from gsim.palace.electrostatic import ElectrostaticSim
from gsim.palace.magnetostatic import MagnetostaticSim

# Mesh utilities
from gsim.palace.mesh import (
    MeshConfig,
    generate_mesh,
)

# Results utilities
from gsim.palace.results import (
    DrivenReport,
    EigenmodeReport,
    ElectrostaticReport,
    SParams,
    TerminalMatrix,
    load_dielectric_interface_summary,
    load_domain_energy_summary,
    load_domain_material_summary,
    load_driven_report,
    load_eigenmode_report,
    load_electrostatic_report,
    load_fields,
    load_postprocessing_index_map,
    load_sparams,
    load_surface_q_summary,
    load_terminal_matrix,
    summarize_domain_loss,
    summarize_loss_budget,
    summarize_surface_loss,
)

__all__ = [
    "DrivenReport",
    "DrivenSim",
    "EigenmodeReport",
    "EigenmodeSim",
    "ElectrostaticReport",
    "ElectrostaticSim",
    "MagnetostaticSim",
    "MeshConfig",
    "SParams",
    "TerminalMatrix",
    "generate_mesh",
    "load_dielectric_interface_summary",
    "load_domain_energy_summary",
    "load_domain_material_summary",
    "load_driven_report",
    "load_eigenmode_report",
    "load_electrostatic_report",
    "load_fields",
    "load_postprocessing_index_map",
    "load_sparams",
    "load_surface_q_summary",
    "load_terminal_matrix",
    "summarize_domain_loss",
    "summarize_loss_budget",
    "summarize_surface_loss",
]


def _parse_palace_result(run_result: RunResult) -> SParams | dict:
    """Parse Palace cloud results into SParams."""
    from gsim.palace.results import load_sparams

    try:
        return load_sparams(run_result.files)
    except FileNotFoundError:
        return run_result.files


register_result_parser("palace", _parse_palace_result)
