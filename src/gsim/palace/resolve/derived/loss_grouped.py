"""Grouped aggregate loss-budget helpers for non-mode-indexed reports.

This module is Resolve-side policy, not a Typed Data or Report model. It is
used when Palace postprocessing rows are organized by source/sample groups
instead of Eigenmode ``mode_index`` rows. Electrostatic and Driven reports can
both use this primitive while keeping their public loss accessors
problem-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gsim.palace._shared import optional_float
from gsim.palace.resolve.derived.epr_math import q_from_inverse_q, sum_numeric_column
from gsim.palace.resolve.derived.loss import (
    ordered_dataframe,
    rate_columns_for_frequency,
    summarize_loss_budget,
)
from gsim.palace.results.loss import LOSS_BUDGET_COLUMNS

if TYPE_CHECKING:
    import pandas as pd


def summarize_grouped_loss_budget(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    *,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Build an aggregate budget when Palace rows have no mode index.

    The preferred grouping key is source/sample provenance. If rows already
    have ``frequency_ghz`` from a Driven frequency sweep, that frequency is
    preserved and used for rate/T1 columns unless an explicit frequency is
    passed by the caller.
    """
    import pandas as pd

    group_columns = _loss_group_columns(domain_loss, surface_loss)
    if group_columns:
        group_values = _loss_group_values(
            domain_loss,
            surface_loss,
            group_columns,
        )
        rows = [
            _loss_budget_row(
                _rows_for_loss_group(domain_loss, values),
                _rows_for_loss_group(surface_loss, values),
                values,
                frequency_ghz=frequency_ghz,
            )
            for values in group_values
        ]
        return ordered_dataframe(
            pd.DataFrame.from_records(rows),
            LOSS_BUDGET_COLUMNS,
        )

    budget = summarize_loss_budget(
        domain_loss,
        surface_loss,
        frequency_ghz=frequency_ghz,
    )
    if not budget.empty or (domain_loss.empty and surface_loss.empty):
        return budget

    rows = [
        _loss_budget_row(
            domain_loss,
            surface_loss,
            {},
            frequency_ghz=frequency_ghz,
        )
    ]
    return ordered_dataframe(pd.DataFrame.from_records(rows), LOSS_BUDGET_COLUMNS)


def _loss_budget_row(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    group_values: dict[str, Any],
    *,
    frequency_ghz: float | None,
) -> dict[str, Any]:
    domain_inverse_q = sum_numeric_column(domain_loss, "inverse_q")
    surface_inverse_q = sum_numeric_column(surface_loss, "inverse_q")
    total_inverse_q = domain_inverse_q + surface_inverse_q
    resolved_frequency = frequency_ghz
    if resolved_frequency is None:
        resolved_frequency = optional_float(group_values.get("frequency_ghz"))
    return {
        **group_values,
        "mode_index": None,
        "frequency_ghz": resolved_frequency,
        "q_eig": float("nan"),
        "inverse_q_eig": float("nan"),
        "domain_inverse_q_sum": domain_inverse_q,
        "surface_inverse_q_sum": surface_inverse_q,
        "total_inverse_q_sum": total_inverse_q,
        "eig_with_surface_inverse_q_sum": float("nan"),
        "q_total": q_from_inverse_q(total_inverse_q),
        "q_eig_with_surface": float("nan"),
        "domain_vs_eig_relative_error": float("nan"),
        **rate_columns_for_frequency(
            frequency_ghz=resolved_frequency,
            inverse_q=total_inverse_q,
        ),
    }


def _loss_group_columns(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
) -> tuple[str, ...]:
    frames = [frame for frame in (domain_loss, surface_loss) if not frame.empty]
    if not frames:
        return ()
    common_columns = set(frames[0].columns)
    for frame in frames[1:]:
        common_columns &= set(frame.columns)

    if "source_index" in common_columns:
        columns = ["source_index"]
        columns.extend(
            column
            for column in ("sample_column", "sample_value", "frequency_ghz")
            if column in common_columns
        )
        return tuple(columns)

    if {"sample_column", "sample_value"} <= common_columns:
        columns = ["sample_column", "sample_value"]
        if "frequency_ghz" in common_columns:
            columns.append("frequency_ghz")
        return tuple(columns)
    if "frequency_ghz" in common_columns:
        return ("frequency_ghz",)
    return ()


def _loss_group_values(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not group_columns:
        return []

    import pandas as pd

    frames = [
        frame.loc[:, list(group_columns)]
        for frame in (domain_loss, surface_loss)
        if not frame.empty
    ]
    if not frames:
        return []
    groups = pd.concat(frames, ignore_index=True).drop_duplicates()
    groups = groups.sort_values(list(group_columns)).reset_index(drop=True)
    return [dict(row) for row in groups.to_dict(orient="records")]


def _rows_for_loss_group(
    frame: pd.DataFrame,
    group_values: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty or not group_values:
        return frame

    import pandas as pd

    mask = pd.Series(True, index=frame.index)
    for column, value in group_values.items():
        if column not in frame.columns:
            continue
        if pd.isna(value):
            mask &= frame[column].isna()
        else:
            mask &= frame[column] == value
    return cast("pd.DataFrame", frame.loc[mask])


__all__ = ["summarize_grouped_loss_budget"]
