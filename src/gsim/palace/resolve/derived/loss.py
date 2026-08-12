"""Resolve Palace EPR tables into loss and inverse-Q summaries.

This module owns the mode-indexed loss-derived table boundary inside Resolve.
It keeps row matching, rate columns, and empty table contracts with the public
domain/surface/budget summary functions because those helpers are local policy
for the same loss semantic. Source/sample grouped budgets stay in
``loss_grouped`` because that is a separate aggregation policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from gsim.palace._shared import optional_float, optional_int
from gsim.palace.resolve.derived.epr_math import (
    float_or_nan,
    inverse_q_from_q,
    numeric_or_default,
    q_from_inverse_q,
    sum_numeric_column,
)
from gsim.palace.resolve.derived.materials import (
    empty_dielectric_interface_summary,
)
from gsim.palace.results.eigenmode import Eigenmodes
from gsim.palace.results.loss import (
    DOMAIN_LOSS_COLUMNS,
    LOSS_BUDGET_COLUMNS,
    LOSS_CHANNEL_BUDGET_COLUMNS,
    SURFACE_LOSS_COLUMNS,
)

_SURFACE_INTERFACE_CONTEXT_KEYS = (
    "surface_attribute",
    "preset_name",
    "preset_source",
    "loss_channel",
    "source_entry_name",
    "thickness",
    "permittivity",
    "loss_tangent",
)


def summarize_domain_loss(
    domain_energy: pd.DataFrame,
    domain_materials: pd.DataFrame,
    *,
    modes: Eigenmodes | pd.DataFrame | None = None,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Combine domain EPR rows with effective material loss parameters.

    ``domain-E.csv`` contains participation values, while ``config.json``
    contains the effective Palace material loss tangent. This helper keeps
    those primitive loaders separate and derives ``inverse_q = p_elec *
    loss_tangent`` only in this Resolve-derived layer.
    """
    import pandas as pd

    if domain_energy.empty:
        return empty_domain_loss_summary()

    rows: list[dict[str, Any]] = []
    for _, domain_row in domain_energy.iterrows():
        row = dict(domain_row)
        material_row = matching_domain_material_row(domain_row, domain_materials)
        loss_tangent = numeric_or_default(
            None if material_row is None else material_row.get("loss_tangent"),
            default=0.0,
        )
        p_elec = numeric_or_default(domain_row.get("p_elec"), default=0.0)
        inverse_q = p_elec * loss_tangent
        mode_index = optional_int(domain_row.get("mode_index"))
        resolved_frequency = frequency_for_mode(
            mode_index,
            modes=modes,
            frequency_ghz=frequency_ghz,
        )
        if resolved_frequency is None:
            resolved_frequency = optional_float(domain_row.get("frequency_ghz"))

        row.update(
            {
                "frequency_ghz": resolved_frequency,
                "material_attribute": (
                    None
                    if material_row is None
                    else material_row.get("material_attribute")
                ),
                "material_attributes": (
                    ()
                    if material_row is None
                    else material_row.get("material_attributes", ())
                ),
                "material_name": (
                    None if material_row is None else material_row.get("material_name")
                ),
                "material_permittivity": (
                    None if material_row is None else material_row.get("permittivity")
                ),
                "material_loss_tangent": loss_tangent,
                "material_conductivity": (
                    None if material_row is None else material_row.get("conductivity")
                ),
                "material_permeability": (
                    None if material_row is None else material_row.get("permeability")
                ),
                "material_axes": (
                    None if material_row is None else material_row.get("material_axes")
                ),
                "loss_tangent": loss_tangent,
                "inverse_q": inverse_q,
                "q_equivalent": q_from_inverse_q(inverse_q),
                **rate_columns_for_frequency(
                    frequency_ghz=resolved_frequency,
                    inverse_q=inverse_q,
                ),
            }
        )
        rows.append(row)

    if not rows:
        return empty_domain_loss_summary()
    return ordered_dataframe(pd.DataFrame.from_records(rows), DOMAIN_LOSS_COLUMNS)


def summarize_surface_loss(
    surface_q: pd.DataFrame,
    dielectric_interfaces: pd.DataFrame | None = None,
    *,
    modes: Eigenmodes | pd.DataFrame | None = None,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Combine surface-Q rows with configured dielectric interface parameters.

    Palace already reports the effective surface ``Q_surf``. This helper keeps
    that solver result authoritative, adds configured interface metadata, and
    derives rate/T1 columns only when mode frequency is available.
    """
    import pandas as pd

    if surface_q.empty:
        return empty_surface_loss_summary()

    interface_frame = (
        dielectric_interfaces
        if dielectric_interfaces is not None
        else empty_dielectric_interface_summary()
    )
    rows: list[dict[str, Any]] = []
    for _, surface_row in surface_q.iterrows():
        row = dict(surface_row)
        interface_row = matching_dielectric_interface_row(
            surface_row,
            interface_frame,
        )
        inverse_q = numeric_or_default(surface_row.get("inverse_q"), default=0.0)
        mode_index = optional_int(surface_row.get("mode_index"))
        resolved_frequency = frequency_for_mode(
            mode_index,
            modes=modes,
            frequency_ghz=frequency_ghz,
        )
        if resolved_frequency is None:
            resolved_frequency = optional_float(surface_row.get("frequency_ghz"))
        row.update(
            {
                "frequency_ghz": resolved_frequency,
                "q_equivalent": q_from_inverse_q(inverse_q),
                "surface_attributes": (),
                **dict.fromkeys(_SURFACE_INTERFACE_CONTEXT_KEYS),
                **rate_columns_for_frequency(
                    frequency_ghz=resolved_frequency,
                    inverse_q=inverse_q,
                ),
            }
        )
        if interface_row is not None:
            row.update(
                {key: interface_row.get(key) for key in _SURFACE_INTERFACE_CONTEXT_KEYS}
            )
            row["surface_attributes"] = interface_row.get("surface_attributes", ())
        rows.append(row)

    if not rows:
        return empty_surface_loss_summary()
    return ordered_dataframe(pd.DataFrame.from_records(rows), SURFACE_LOSS_COLUMNS)


def summarize_loss_budget(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    *,
    modes: Eigenmodes | pd.DataFrame | None = None,
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Summarize per-mode bulk/domain and surface inverse-Q contributions."""
    import pandas as pd

    surface_loss = select_primary_surface_loss_rows(surface_loss)
    if domain_loss.empty and surface_loss.empty:
        return empty_loss_budget_summary()

    mode_indices = loss_mode_indices(domain_loss, surface_loss, modes)
    if not mode_indices and frequency_ghz is not None:
        mode_indices = (None,)
    if not mode_indices:
        return empty_loss_budget_summary()

    rows: list[dict[str, Any]] = []
    for mode_index in mode_indices:
        domain_rows = rows_for_mode(domain_loss, mode_index)
        surface_rows = rows_for_mode(surface_loss, mode_index)
        domain_inverse_q = sum_numeric_column(domain_rows, "inverse_q")
        surface_inverse_q = sum_numeric_column(surface_rows, "inverse_q")
        total_inverse_q = domain_inverse_q + surface_inverse_q
        q_eig = q_for_mode(mode_index, modes=modes)
        inverse_q_eig = inverse_q_from_q(q_eig)
        eig_with_surface_inverse_q = inverse_q_eig + surface_inverse_q
        resolved_frequency = frequency_for_mode(
            mode_index,
            modes=modes,
            frequency_ghz=frequency_ghz,
        )
        rows.append(
            {
                "mode_index": mode_index,
                "frequency_ghz": resolved_frequency,
                "q_eig": q_eig,
                "inverse_q_eig": inverse_q_eig,
                "domain_inverse_q_sum": domain_inverse_q,
                "surface_inverse_q_sum": surface_inverse_q,
                "total_inverse_q_sum": total_inverse_q,
                "eig_with_surface_inverse_q_sum": eig_with_surface_inverse_q,
                "q_total": q_from_inverse_q(total_inverse_q),
                "q_eig_with_surface": q_from_inverse_q(eig_with_surface_inverse_q),
                "domain_vs_eig_relative_error": relative_error(
                    domain_inverse_q,
                    inverse_q_eig,
                ),
                **rate_columns_for_frequency(
                    frequency_ghz=resolved_frequency,
                    inverse_q=total_inverse_q,
                ),
            }
        )

    return ordered_dataframe(pd.DataFrame.from_records(rows), LOSS_BUDGET_COLUMNS)


def summarize_loss_channel_budget(loss_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize caller-tagged loss channels without inferring taxonomy."""
    if (
        loss_rows.empty
        or "loss_channel" not in loss_rows
        or "inverse_q" not in loss_rows
    ):
        return empty_loss_channel_budget_summary()

    frame = select_primary_surface_loss_rows(loss_rows)
    frame = frame.loc[frame["loss_channel"].notna()].copy()
    if frame.empty:
        return empty_loss_channel_budget_summary()

    groups = tuple(
        column
        for column in (
            "source_index",
            "mode_index",
            "sample_column",
            "sample_value",
            "frequency_ghz",
            "source_entry_name",
        )
        if column in frame.columns and frame[column].notna().any()
    )
    sum_columns = [*groups, "loss_channel"]
    budget = frame.groupby(sum_columns, dropna=False)["inverse_q"].sum().reset_index()
    if groups:
        totals = budget.groupby(list(groups), dropna=False)["inverse_q"].transform(
            "sum"
        )
    else:
        totals = budget["inverse_q"].sum()
    budget["loss_fraction"] = budget["inverse_q"].div(totals).fillna(0.0)
    return ordered_dataframe(budget, LOSS_CHANNEL_BUDGET_COLUMNS)


def select_primary_surface_loss_rows(loss_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep all Surface EPR rows for source-aware surface budgeting."""
    return loss_rows


def rate_columns_for_frequency(
    *,
    frequency_ghz: float | None,
    inverse_q: float,
) -> dict[str, float]:
    """Return gamma and T1 columns derived from frequency and inverse-Q."""
    if frequency_ghz is None:
        return {}
    frequency = float(frequency_ghz)
    if frequency <= 0.0 or not np.isfinite(frequency):
        return {}
    gamma_hz = frequency * 1.0e9 * inverse_q
    gamma_rad_per_s = 2.0 * np.pi * gamma_hz
    return {
        "gamma_rad_per_s": gamma_rad_per_s,
        "gamma_per_us": gamma_rad_per_s / 1.0e6,
        "gamma_hz": gamma_hz,
        "gamma_mhz": gamma_hz / 1.0e6,
        "t1_us": float("inf") if gamma_rad_per_s <= 0.0 else 1.0e6 / gamma_rad_per_s,
    }


def validate_positive_frequency_ghz(frequency_ghz: float) -> None:
    """Validate an explicit frequency used for loss-rate derivation."""
    frequency = float(frequency_ghz)
    if frequency <= 0.0 or not np.isfinite(frequency):
        msg = f"frequency_ghz must be a positive finite value, got {frequency_ghz!r}"
        raise ValueError(msg)


def tuple_of_ints(value: Any) -> tuple[int, ...]:
    """Return a tuple of integer attributes from Palace sidecar values."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        try:
            return (int(value),)
        except ValueError:
            return ()
    if isinstance(value, (int, float)):
        if np.isfinite(float(value)):
            return (int(value),)
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()


def relative_error(estimate: float, reference: float) -> float:
    """Return the relative error between an estimate and finite reference."""
    if reference == 0.0 or not np.isfinite(reference):
        return float("nan")
    return (estimate - reference) / reference


def ordered_dataframe(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return a frame ordered by known report columns followed by extras."""
    ordered = [column for column in columns if column in frame.columns]
    extra = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, [*ordered, *extra]]


def empty_domain_loss_summary() -> pd.DataFrame:
    """Return an empty domain loss summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(columns=DOMAIN_LOSS_COLUMNS)


def empty_surface_loss_summary() -> pd.DataFrame:
    """Return an empty surface loss summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(columns=SURFACE_LOSS_COLUMNS)


def empty_loss_budget_summary() -> pd.DataFrame:
    """Return an empty loss-budget summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(columns=LOSS_BUDGET_COLUMNS)


def empty_loss_channel_budget_summary() -> pd.DataFrame:
    """Return an empty loss-channel budget summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(columns=LOSS_CHANNEL_BUDGET_COLUMNS)


def matching_domain_material_row(
    domain_row: Any,
    domain_materials: pd.DataFrame,
) -> dict[str, Any] | None:
    """Return the material row matching a domain EPR row."""
    if domain_materials.empty:
        return None

    domain_index = optional_int(domain_row.get("domain_index"))
    if domain_index is not None and "domain_index" in domain_materials.columns:
        for _, material_row in domain_materials.iterrows():
            if optional_int(material_row.get("domain_index")) == domain_index:
                return dict(material_row)

    attributes = tuple_of_ints(domain_row.get("attributes"))
    if not attributes:
        return None

    if "material_attribute" in domain_materials.columns:
        for _, material_row in domain_materials.iterrows():
            attribute = optional_int(material_row.get("material_attribute"))
            if attribute in attributes:
                return dict(material_row)

    if "material_attributes" in domain_materials.columns:
        wanted = set(attributes)
        for _, material_row in domain_materials.iterrows():
            material_attributes = set(
                tuple_of_ints(material_row.get("material_attributes"))
            )
            if wanted & material_attributes:
                return dict(material_row)

    return None


def matching_dielectric_interface_row(
    surface_row: Any,
    dielectric_interfaces: pd.DataFrame,
) -> dict[str, Any] | None:
    """Return the dielectric-interface row matching a surface EPR row."""
    if dielectric_interfaces.empty:
        return None

    surface_index = optional_int(surface_row.get("surface_index"))
    if surface_index is not None and "surface_index" in dielectric_interfaces.columns:
        for _, interface_row in dielectric_interfaces.iterrows():
            if optional_int(interface_row.get("surface_index")) == surface_index:
                return dict(interface_row)

    attributes = tuple_of_ints(surface_row.get("attributes"))
    if not attributes:
        return None

    if "surface_attribute" in dielectric_interfaces.columns:
        for _, interface_row in dielectric_interfaces.iterrows():
            attribute = optional_int(interface_row.get("surface_attribute"))
            if attribute in attributes:
                return dict(interface_row)

    if "surface_attributes" in dielectric_interfaces.columns:
        wanted = set(attributes)
        for _, interface_row in dielectric_interfaces.iterrows():
            interface_attributes = set(
                tuple_of_ints(interface_row.get("surface_attributes"))
            )
            if wanted & interface_attributes:
                return dict(interface_row)

    return None


def frequency_for_mode(
    mode_index: int | None,
    *,
    modes: Eigenmodes | pd.DataFrame | None,
    frequency_ghz: float | None,
) -> float | None:
    """Return the resolved mode frequency in GHz."""
    if frequency_ghz is not None:
        return float(frequency_ghz)
    if mode_index is None or modes is None:
        return None

    frame = modes.dataframe if isinstance(modes, Eigenmodes) else modes
    if frame.empty or "mode_index" not in frame.columns:
        return None
    frequency_column = (
        "freq_real_ghz"
        if "freq_real_ghz" in frame.columns
        else "frequency_ghz"
        if "frequency_ghz" in frame.columns
        else None
    )
    if frequency_column is None:
        return None

    matches = frame.loc[frame["mode_index"].map(optional_int) == mode_index]
    if matches.empty:
        return None
    frequency = float_or_nan(matches.iloc[0][frequency_column])
    return frequency if np.isfinite(frequency) else None


def q_for_mode(
    mode_index: int | None,
    *,
    modes: Eigenmodes | pd.DataFrame | None,
) -> float:
    """Return the solver Q for a mode, or ``nan`` when unavailable."""
    if mode_index is None or modes is None:
        return float("nan")
    frame = modes.dataframe if isinstance(modes, Eigenmodes) else modes
    if frame.empty or "mode_index" not in frame.columns:
        return float("nan")
    q_column = (
        "q"
        if "q" in frame.columns
        else "q_factor"
        if "q_factor" in frame.columns
        else None
    )
    if q_column is None:
        return float("nan")

    matches = frame.loc[frame["mode_index"].map(optional_int) == mode_index]
    if matches.empty:
        return float("nan")
    return float_or_nan(matches.iloc[0][q_column])


def loss_mode_indices(
    domain_loss: pd.DataFrame,
    surface_loss: pd.DataFrame,
    modes: Eigenmodes | pd.DataFrame | None,
) -> tuple[int | None, ...]:
    """Return sorted mode indices represented by loss rows or eigenmode data."""
    mode_indices: set[int] = set()
    for frame in (domain_loss, surface_loss):
        if "mode_index" not in frame.columns:
            continue
        for value in frame["mode_index"]:
            mode_index = optional_int(value)
            if mode_index is not None:
                mode_indices.add(mode_index)

    if modes is not None:
        frame = modes.dataframe if isinstance(modes, Eigenmodes) else modes
        if "mode_index" in frame.columns:
            for value in frame["mode_index"]:
                mode_index = optional_int(value)
                if mode_index is not None:
                    mode_indices.add(mode_index)

    return tuple(sorted(mode_indices))


def rows_for_mode(frame: pd.DataFrame, mode_index: int | None) -> pd.DataFrame:
    """Return rows belonging to one mode, or all rows when no mode column exists."""
    if frame.empty:
        return frame
    if mode_index is None or "mode_index" not in frame.columns:
        return frame
    return cast(
        "pd.DataFrame", frame.loc[frame["mode_index"].map(optional_int) == mode_index]
    )


__all__ = [
    "DOMAIN_LOSS_COLUMNS",
    "LOSS_BUDGET_COLUMNS",
    "SURFACE_LOSS_COLUMNS",
    "select_primary_surface_loss_rows",
    "summarize_domain_loss",
    "summarize_loss_budget",
    "summarize_loss_channel_budget",
    "summarize_surface_loss",
    "validate_positive_frequency_ghz",
]
