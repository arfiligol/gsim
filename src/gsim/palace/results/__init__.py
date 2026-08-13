"""Official Palace result utilities plus typed SCQ result/report objects.

The upstream 0.2 public result API remains in :mod:`legacy`; the typed
Resolve/report layer is additive and lives in explicit submodules.
"""

from __future__ import annotations

from gsim.palace.results.benchmarks import SimulationBenchmark
from gsim.palace.results.eigenmode import EigenmodeConvergence, Eigenmodes
from gsim.palace.results.electrostatic import TerminalMatrix, TerminalMatrixConvergence
from gsim.palace.results.indexed import IndexedCsv, IndexedCsvColumn
from gsim.palace.results.legacy import (
    ModeMetrics,
    PalaceTextResults,
    SParam,
    SParams,
    _resolve_source,
    get_port_map,
    load_fields,
    load_sparams,
    load_text_results,
)
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
    "ModeMetrics",
    "PalaceTextResults",
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
    "_resolve_source",
    "get_port_map",
    "load_fields",
    "load_sparams",
    "load_text_results",
]
