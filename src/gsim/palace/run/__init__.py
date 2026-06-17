"""Run-stage execution helpers for Palace workflows.

Responsibility:
Owns concrete execution implementations that operate on an already prepared
Palace run folder.

Does not own:
Simulation model construction, mesh/config generation, Resolve/report loading,
typed result models, display helpers, or PDK-specific run-profile catalogs.

Inputs:
Canonical run folders produced through ``PalaceSimBase`` methods and
caller-supplied execution options.

Outputs:
Local solver side effects under ``results/palace`` and runtime metadata under
``metadata``. Notebook-facing methods on ``PalaceSimBase`` remain the public
composition points.

Pipeline position:
``sim.write_config()`` -> ``sim.run_local()`` / handoff helpers ->
``resolve_palace_result(run_folder, ...)``.
"""

from __future__ import annotations

from gsim.palace.run.handoff import generate_palace_handoff_package
from gsim.palace.run.local import run_palace_local

__all__ = ["generate_palace_handoff_package", "run_palace_local"]
