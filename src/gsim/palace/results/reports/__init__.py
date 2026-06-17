"""Problem Type Report models for Palace semantic results.

This package is the report half of ``gsim.palace.results``. This ``__init__``
is only a small concrete-report facade; report internals import sibling owner
modules directly so the package facade never becomes an implementation hub.

The structure is intentionally navigable:

* ``base`` defines the internal common report contract shared by every problem
  type.
* ``driven``, ``eigenmode``, and ``electrostatic`` define the concrete Problem
  Type Reports that notebooks and type checkers should inspect.

Report classes aggregate Typed Data objects and expose ``show_all_results()``.
They do not parse Palace artifacts, compute derived physics tables, or define
generic display primitives. Resolve assembly creates report instances;
Typed Data owns table/plot semantics; Display owns generic visualization
helpers. Shared table classes may be reused across reports, but report-level
accessors stay problem-specific because identical columns do not imply
identical loss semantics.
"""

from __future__ import annotations

from gsim.palace.results.reports.driven import DrivenReport
from gsim.palace.results.reports.eigenmode import EigenmodeReport
from gsim.palace.results.reports.electrostatic import ElectrostaticReport

__all__ = [
    "DrivenReport",
    "EigenmodeReport",
    "ElectrostaticReport",
]
