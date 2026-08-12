"""Primitive Palace artifact loaders.

This package owns direct parsing of already-located Palace artifacts: field
files, eigenmode CSVs, terminal matrices, indexed postprocessing CSVs, and
postprocessing index maps.

Loaders parse raw files into primitive frames, records, or narrowly scoped
typed artifacts only when that typed object is still loader-owned. Driven
``SParams`` loading lives with the ``SParams`` typed data owner in
``gsim.palace.results.driven`` so run-stage convenience paths do not import
Resolve. Loaders do not decide which optional reports are required, compute
derived physics quantities, assemble problem reports, or display results.
Source discovery lives in ``gsim.palace.resolve.sources``; derived semantic
transforms live in ``gsim.palace.resolve.derived``; problem report construction
lives in ``gsim.palace.resolve.assembly``.
"""

from __future__ import annotations
