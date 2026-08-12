"""Notebook-facing facade for Palace simulation, Resolve, and Results.

The package root exposes the small public surface users need for notebooks:
problem-specific simulation classes, mesh generation entrypoints, the Resolve
entrypoint, report bundle models, concrete report models, and selected Typed
Data classes.

Implementation responsibility is intentionally below the root:

* ``gsim.palace.run_folder`` owns the canonical Palace-only run folder:
  root execution inputs, ``metadata/`` sidecars, ``logs/``, optional
  ``geometry/design.gds``, and raw ``results/palace/`` solver outputs.
* ``gsim.palace.run_stage`` owns lightweight run-stage handles returned by
  package/execution APIs before Resolve starts.
* ``gsim.palace.run`` owns concrete execution implementations, such as local
  Palace process execution. ``PalaceSimBase`` keeps notebook-facing method
  entrypoints and delegates execution details to this package.
* ``gsim.palace.handoff`` owns Palace archive/script packaging around that
  run folder. AEDT/HFSS export and result packaging belong to public PDK code,
  not to this package.
* ``gsim.palace.resolve`` adapts completed Palace run folders/cloud mappings
  into source-audit objects. Its root entrypoint is
  ``resolve_palace_result(...)``; report loading remains an explicit second
  step through ``PalaceResolvedResult.load_report(...)``.
* ``gsim.palace.results`` owns semantic Typed Data and Problem Type Reports.
* ``gsim.palace.display`` owns generic table/plot primitives used by Typed
  Data visualizers.
* ``gsim.palace.mesh`` owns mesh and postprocessing config generation.

The root does not expose direct problem-specific report loaders. Review code
should use ``resolve_palace_result(...).load_report()`` so the source audit,
typed report construction, and concrete report type remain visible.
Missing artifacts are represented by the Resolve summary until callers choose
``require_report=True`` or call ``require_report()``, which raise instead of
returning placeholder reports.

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

    # Generate a reviewable handoff package
    handle = sim.generate_handoff_package()

    # Or resolve artifacts first, then compose a typed report explicitly
    resolved = resolve_palace_result("./sim", problem_type="Driven")
    report_bundle = resolved.load_report(require_report=True)
    report = report_bundle.require_report()
"""

from __future__ import annotations

from gsim.gcloud import RunResult, register_result_parser
from gsim.palace.boundarymode import BoundaryModeSim
from gsim.palace.driven import DrivenSim
from gsim.palace.eigenmode import EigenmodeSim
from gsim.palace.electrostatic import ElectrostaticSim
from gsim.palace.field_viz import extract_streamplot_inputs_2d, plot_fields_2d
from gsim.palace.magnetostatic import MagnetostaticSim
from gsim.palace.mesh import (
    MeshConfig,
    generate_mesh,
)
from gsim.palace.models import BoundaryModeConfig, CrossSectionPlaneConfig
from gsim.palace.resolve import (
    PalaceResolvedResult,
    PalaceResultBundle,
    PalaceRunArtifacts,
    PalaceRunSummary,
    resolve_palace_result,
)
from gsim.palace.results.driven import SParams
from gsim.palace.results.reports.driven import DrivenReport
from gsim.palace.results.reports.eigenmode import EigenmodeReport
from gsim.palace.results.reports.electrostatic import ElectrostaticReport
from gsim.palace.run_stage import PalaceRunHandle

__all__ = [
    "BoundaryModeConfig",
    "BoundaryModeSim",
    "CrossSectionPlaneConfig",
    "DrivenReport",
    "DrivenSim",
    "EigenmodeReport",
    "EigenmodeSim",
    "ElectrostaticReport",
    "ElectrostaticSim",
    "MagnetostaticSim",
    "MeshConfig",
    "PalaceResolvedResult",
    "PalaceResultBundle",
    "PalaceRunArtifacts",
    "PalaceRunHandle",
    "PalaceRunSummary",
    "SParams",
    "extract_streamplot_inputs_2d",
    "generate_mesh",
    "plot_fields_2d",
    "resolve_palace_result",
]


def _parse_palace_result(run_result: RunResult) -> SParams | dict:
    """Parse Palace cloud results into SParams."""
    from gsim.palace.results.driven import load_sparams

    try:
        return load_sparams(run_result.files)
    except FileNotFoundError:
        return run_result.files


register_result_parser("palace", _parse_palace_result)
