"""Shared EPR and loss-rate math for Palace artifact resolution.

This module owns small numeric conversions that are common to participation,
loss, and benchmark-style result loaders. It deliberately stays inside
``resolve`` because these helpers normalize solver outputs into report-ready
tables; typed result objects should receive already-normalized values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def float_or_nan(value: Any) -> float:
    """Return ``value`` as ``float`` or ``nan`` when conversion fails."""
    try:
        return float(cast("Any", value))
    except (TypeError, ValueError):
        return float("nan")


def numeric_or_default(value: Any, *, default: float) -> float:
    """Return a finite float, falling back to ``default`` otherwise."""
    numeric = float_or_nan(value)
    return numeric if np.isfinite(numeric) else default


def q_from_inverse_q(inverse_q: float) -> float:
    """Convert an inverse-Q contribution into an equivalent Q value."""
    if inverse_q == 0.0:
        return float("inf")
    if not np.isfinite(inverse_q):
        return 0.0
    return 1.0 / inverse_q


def inverse_q_from_q(q_value: float) -> float:
    """Convert a Q value into an inverse-Q contribution."""
    if not np.isfinite(q_value):
        return 0.0
    if q_value == 0.0:
        return float("inf")
    return 1.0 / q_value


def sum_numeric_column(frame: pd.DataFrame, column: str) -> float:
    """Return a numeric column sum, treating missing and nonnumeric values as zero."""
    import pandas as pd

    if column not in frame.columns:
        return 0.0
    values = cast("pd.Series", pd.to_numeric(frame[column], errors="coerce")).fillna(
        0.0
    )
    return float_or_nan(values.sum())


def fraction(value: float, total: float) -> float:
    """Return ``value / total`` with zero for invalid totals."""
    if total == 0.0 or not np.isfinite(total):
        return 0.0
    return value / total
