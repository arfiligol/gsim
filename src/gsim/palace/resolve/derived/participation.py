"""Resolve Palace indexed EPR/energy CSV files into dataframes.

This module converts indexed Palace postprocessing CSV data into report-ready
domain energy, surface-Q, and Eigenmode port-EPR tables. It does not compute
loss budgets; loss-specific normalization lives in
``gsim.palace.resolve.derived.loss``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from gsim.palace.resolve.derived.epr_math import (
    float_or_nan,
    fraction,
    inverse_q_from_q,
    q_from_inverse_q,
    sum_numeric_column,
)
from gsim.palace.resolve.loaders.indexed_csv import (
    indexed_column_provenance,
    load_indexed_csv,
)


def load_domain_energy_summary(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load ``domain-E.csv`` as one row per sampled domain index.

    The returned frame preserves Palace quantities such as ``E_elec`` and
    ``p_elec`` while attaching physical-name provenance from
    ``palace_index_map.json``.
    """
    return _load_indexed_quantity_summary(
        source,
        "domain-E.csv",
        index_name="domain_index",
        quantity_columns={
            "E_elec": "E_elec_j",
            "E_mag": "E_mag_j",
            "p_elec": "p_elec",
            "p_mag": "p_mag",
        },
        index_map_path=index_map_path,
    )


def load_surface_q_summary(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load ``surface-Q.csv`` as one row per sampled surface index."""
    frame = _load_indexed_quantity_summary(
        source,
        "surface-Q.csv",
        index_name="surface_index",
        quantity_columns={
            "p_surf": "p_surf",
            "Q_surf": "q_surf",
        },
        index_map_path=index_map_path,
    )
    frame = _drop_native_mask_surface_rows(frame)
    if "q_surf" in frame.columns:
        frame["inverse_q"] = _inverse_q_values(frame["q_surf"])
    return frame


def summarize_surface_q_by_interface(
    surface_summary: pd.DataFrame,
    *,
    include_interface_types: tuple[str, ...] = ("MA", "MS", "SA"),
) -> pd.DataFrame:
    """Summarize a surface-Q frame by interface type."""
    import pandas as pd

    columns = [
        "interface_type",
        "surface_count",
        "p_surf_sum",
        "inverse_q_sum",
        "q_equivalent",
        "p_surf_fraction",
        "inverse_q_fraction",
    ]
    if surface_summary.empty:
        return pd.DataFrame(columns=columns)

    frame = surface_summary.copy()
    if "interface_type" not in frame.columns:
        frame["interface_type"] = ""
    if "p_surf" not in frame.columns:
        frame["p_surf"] = 0.0
    if "inverse_q" not in frame.columns:
        frame["inverse_q"] = 0.0

    interface_order = list(include_interface_types)
    for interface_type in frame["interface_type"].dropna().astype(str).unique():
        if interface_type and interface_type not in interface_order:
            interface_order.append(interface_type)

    rows: list[dict[str, Any]] = []
    for interface_type in interface_order:
        subset = frame.loc[frame["interface_type"] == interface_type]
        p_surf_sum = float(subset["p_surf"].sum()) if not subset.empty else 0.0
        inverse_q_sum = float(subset["inverse_q"].sum()) if not subset.empty else 0.0
        rows.append(
            {
                "interface_type": interface_type,
                "surface_count": len(subset),
                "p_surf_sum": p_surf_sum,
                "inverse_q_sum": inverse_q_sum,
                "q_equivalent": q_from_inverse_q(inverse_q_sum),
            }
        )

    summary = pd.DataFrame.from_records(rows, columns=columns[:5])
    p_total = sum_numeric_column(summary, "p_surf_sum")
    inverse_q_total = sum_numeric_column(summary, "inverse_q_sum")
    summary["p_surf_fraction"] = [
        fraction(float(value), p_total) for value in summary["p_surf_sum"]
    ]
    summary["inverse_q_fraction"] = [
        fraction(float(value), inverse_q_total) for value in summary["inverse_q_sum"]
    ]
    return cast("pd.DataFrame", summary.loc[:, columns])


def load_port_epr_summary(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Eigenmode ``port-EPR.csv`` as one row per sampled lumped port."""
    frame = _load_indexed_quantity_summary(
        source,
        "port-EPR.csv",
        index_name="port_index",
        quantity_columns={
            "p": "p_port",
            "p_port": "p_port",
        },
        index_map_path=index_map_path,
    )
    if "p_port" in frame.columns:
        frame["abs_p_port"] = frame["p_port"].abs()
        group_columns = _sample_group_columns(frame)
        if group_columns:
            totals = frame.groupby(group_columns, dropna=False)["abs_p_port"].transform(
                "sum"
            )
        else:
            totals = frame["abs_p_port"].sum()
        frame["abs_p_port_fraction"] = _fraction_values(frame["abs_p_port"], totals)
    return frame


def _load_indexed_quantity_summary(
    source: str | Path | dict,
    csv_name: str,
    *,
    index_name: str,
    quantity_columns: dict[str, str],
    index_map_path: str | Path | None,
) -> pd.DataFrame:
    import pandas as pd

    indexed = load_indexed_csv(
        source,
        csv_name,
        index_map_path=index_map_path,
        rename_columns=False,
    )
    frame = indexed.dataframe
    indexed_column_names = {column.original_name for column in indexed.columns}
    sample_columns = [
        column for column in frame.columns if column not in indexed_column_names
    ]

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row_offset, (_, source_row) in enumerate(frame.iterrows(), start=1):
        sample_metadata = _sample_metadata(source_row, sample_columns, row_offset)
        for column in indexed.columns:
            output_column = quantity_columns.get(column.quantity)
            if output_column is None:
                continue
            key = (row_offset, column.index)
            row = rows.setdefault(
                key,
                {
                    **sample_metadata,
                    **indexed_column_provenance(column, index_name=index_name),
                },
            )
            row[output_column] = source_row[column.original_name]

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame.from_records(list(rows.values()))
    sort_columns = [
        column
        for column in (
            "mode_index",
            "source_index",
            "frequency_ghz",
            "row_index",
            index_name,
        )
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns).reset_index(drop=True)
    return result


def _drop_native_mask_surface_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "metadata" not in frame.columns:
        return frame
    keep = [
        not (isinstance(metadata, Mapping) and "mask_margin_nm" in metadata)
        for metadata in frame["metadata"]
    ]
    return frame.loc[keep].reset_index(drop=True)


def _sample_metadata(
    source_row: Any, sample_columns: list[str], row_offset: int
) -> dict[str, Any]:
    row: dict[str, Any] = {"row_index": row_offset}
    if not sample_columns:
        return row

    sample_column = sample_columns[0]
    sample_value = source_row[sample_column]
    row["sample_column"] = sample_column
    row["sample_value"] = sample_value

    alias = _sample_column_alias(sample_column)
    if alias is not None:
        row[alias] = _coerce_sample_value(
            sample_value, integer=alias.endswith("_index")
        )
    return row


def _sample_column_alias(column: str) -> str | None:
    normalized = column.strip().lower()
    if normalized in {"m", "mode"}:
        return "mode_index"
    if normalized == "i":
        return "source_index"
    if normalized.startswith("f") and "ghz" in normalized:
        return "frequency_ghz"
    return None


def _coerce_sample_value(value: Any, *, integer: bool) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if integer and np.isfinite(numeric):
        return round(numeric)
    return numeric


def _inverse_q_values(values: Any) -> pd.Series:
    import pandas as pd

    numeric = cast("pd.Series", pd.to_numeric(values, errors="coerce")).fillna(
        float("nan")
    )
    return numeric.map(lambda value: inverse_q_from_q(float_or_nan(value)))


def _fraction_values(values: Any, totals: Any) -> pd.Series:
    import pandas as pd

    numeric_values = cast("pd.Series", pd.to_numeric(values, errors="coerce")).fillna(
        0.0
    )
    if np.isscalar(totals):
        total = float_or_nan(totals)
        if total == 0.0 or not np.isfinite(total):
            return numeric_values * 0.0
        return numeric_values / total

    numeric_totals = cast("pd.Series", pd.to_numeric(totals, errors="coerce")).fillna(
        float("nan")
    )
    result = numeric_values / numeric_totals
    return result.where((numeric_totals != 0.0) & np.isfinite(numeric_totals), 0.0)


def _sample_group_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("mode_index", "source_index", "frequency_ghz", "sample_value")
        if column in frame.columns
    ]


def empty_domain_energy_summary() -> pd.DataFrame:
    """Return an empty domain-energy summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "domain_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "E_elec_j",
            "E_mag_j",
            "p_elec",
            "p_mag",
        ]
    )


def empty_surface_q_summary() -> pd.DataFrame:
    """Return an empty surface-Q summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "surface_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "interface_type",
            "p_surf",
            "q_surf",
            "inverse_q",
        ]
    )


def empty_port_epr_summary() -> pd.DataFrame:
    """Return an empty port-EPR summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "row_index",
            "sample_column",
            "sample_value",
            "mode_index",
            "port_index",
            "section",
            "source_name",
            "physical_name",
            "entry_name",
            "role",
            "attributes",
            "postprocessing_type",
            "p_port",
            "abs_p_port",
            "abs_p_port_fraction",
        ]
    )
