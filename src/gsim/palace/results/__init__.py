"""Semantic Palace result objects and problem-type report models.

This package is the semantic result layer. It owns two reviewable levels:

* Typed Data objects, such as ``SParams``, ``Eigenmodes``,
  ``TerminalMatrix``, EPR participation tables, and loss tables. Each Typed
  Data object owns its own notebook-facing ``visualize()`` behavior.
* Problem Type Reports, such as ``DrivenReport``, ``EigenmodeReport``, and
  ``ElectrostaticReport``. Reports aggregate Typed Data and expose
  ``show_all_results()``; they do not create new visualization semantics.
  Simulation benchmark metadata is displayed explicitly through
  ``show_simulation_benchmark()`` because it answers run-cost questions rather
  than problem-physics questions.

This package does not discover folders, parse raw Palace artifact sets, or
decide source/missing-artifact audit policy. That adapter work lives in
``gsim.palace.resolve``. It also does not define plotting primitives; generic
table and plot helpers live in ``gsim.palace.display`` and are selected by the
Typed Data objects.

Typed Data public API requirement:

* ``visualize()`` is the common notebook/report entry point and must return a
  named mapping of tables, figures, or scalar display values.
* ``tables()`` is optional and should exist only when the object has a useful
  default table.
* ``figures()`` is optional and should exist only when the object has a useful
  default figure.
* Do not add empty ``tables()`` or ``figures()`` hooks for structural symmetry;
  absence is the signal that the view is not part of that Typed Data contract.

Review path:
``Run Stage handle / completed run folder -> resolve -> results typed data ->
results report -> display primitives``.
"""

from __future__ import annotations

from gsim.palace.results.benchmarks import SimulationBenchmark
from gsim.palace.results.driven import SParam, SParams
from gsim.palace.results.eigenmode import EigenmodeConvergence, Eigenmodes
from gsim.palace.results.electrostatic import TerminalMatrix, TerminalMatrixConvergence
from gsim.palace.results.indexed import IndexedCsv, IndexedCsvColumn
from gsim.palace.results.loss import (
    DOMAIN_LOSS_COLUMNS,
    LOSS_BUDGET_COLUMNS,
    SURFACE_LOSS_COLUMNS,
    DomainLoss,
    EprLossTable,
    LossBudget,
    ReportLoss,
    SurfaceLoss,
)
from gsim.palace.results.postprocessing import (
    DomainEnergy,
    PortEpr,
    PostprocessingTable,
    SurfaceQ,
)
from gsim.palace.results.reports import (
    DrivenReport,
    EigenmodeReport,
    ElectrostaticReport,
)

__all__ = [
    "DOMAIN_LOSS_COLUMNS",
    "LOSS_BUDGET_COLUMNS",
    "SURFACE_LOSS_COLUMNS",
    "DomainEnergy",
    "DomainLoss",
    "DrivenReport",
    "EigenmodeConvergence",
    "EigenmodeReport",
    "Eigenmodes",
    "ElectrostaticReport",
    "EprLossTable",
    "IndexedCsv",
    "IndexedCsvColumn",
    "LossBudget",
    "PortEpr",
    "PostprocessingTable",
    "ReportLoss",
    "SParam",
    "SParams",
    "SimulationBenchmark",
    "SurfaceLoss",
    "SurfaceQ",
    "TerminalMatrix",
    "TerminalMatrixConvergence",
]
