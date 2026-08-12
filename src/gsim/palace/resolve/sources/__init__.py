"""Palace run-source discovery, audit, and sweep metadata.

This package owns the filesystem/source side of Resolve. It discovers Palace
output folders, handoff sidecars, runtime/resource records, Slurm metadata, and
sweep point manifests, then exposes structured source summaries for the next
pipeline stage.

It does not parse physics CSV semantics, derive EPR/loss quantities, assemble
problem reports, or render notebook views. Primitive CSV loaders live in
``gsim.palace.resolve.loaders``; derived semantic table builders live in
``gsim.palace.resolve.derived``; report construction lives in
``gsim.palace.resolve.assembly``.
"""

from __future__ import annotations
