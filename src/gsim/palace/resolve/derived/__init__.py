"""Derived semantic tables for Palace result review.

This package owns transformations from primitive loaded artifacts into
semantic tables such as material summaries, EPR participation summaries, and
loss budgets. It also owns derived typed-data construction when primitive
files need additional semantic packaging, such as Electrostatic terminal
matrices and their AMR convergence views.

Derived functions may combine loader output with run/source context, but they
do not discover folders, choose problem-type assembly policy, define Typed Data
classes, or render figures. Typed Data and report models live in
``gsim.palace.results``; source discovery lives in
``gsim.palace.resolve.sources``; problem report construction lives in
``gsim.palace.resolve.assembly``.
"""

from __future__ import annotations
