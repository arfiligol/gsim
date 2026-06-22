"""Run-stage execution helpers for Palace workflows.

This package contains concrete execution helpers that operate on an already
prepared Palace run folder. Notebook-facing methods on ``PalaceSimBase`` remain
the public composition points for local execution and handoff packaging.

Simulation model construction, mesh/config generation, Resolve/report loading,
typed result models, display helpers, and PDK-specific run-profile catalogs live
outside this package. Run-stage helpers consume canonical run folders and write
solver side effects under ``results/palace`` plus runtime metadata under
``metadata`` before Resolve picks the folder back up.
"""

from __future__ import annotations

from gsim.palace.run.handoff import generate_palace_handoff_package
from gsim.palace.run.local import run_palace_local

__all__ = ["generate_palace_handoff_package", "run_palace_local"]
