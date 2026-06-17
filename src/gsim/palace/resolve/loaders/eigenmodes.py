"""Resolve Palace Eigenmode CSV artifacts into typed data.

This module owns the primitive Eigenmode loaders: final ``eig.csv`` rows, AMR
iteration history, and convergence summaries. It returns ``Eigenmodes`` typed
data and dataframes consumed by Eigenmode reports.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from gsim.palace.resolve.sources.path_utils import resolve_palace_output_dir
from gsim.palace.results.eigenmode import Eigenmodes

if TYPE_CHECKING:
    import pandas as pd

_ITERATION_DIR_RE = re.compile(r"iteration(\d+)$")
_EIG_CSV_RE = re.compile(r"eig.*\.csv$", re.IGNORECASE)


def load_eigenmodes(source: str | Path | dict) -> Eigenmodes:
    """Load Palace ``eig.csv`` as a normalized Eigenmode result table.

    Args:
        source: CSV path, simulation directory, Palace output directory, or
            results dict returned by ``EigenmodeSim.run()`` / ``run_local()``.

    Returns:
        :class:`Eigenmodes` with stable columns:
        ``mode_index``, ``freq_real_ghz``, ``freq_imag_ghz``, ``q``,
        ``error_backward``, and ``error_absolute``.
    """
    import pandas as pd

    csv_path = _resolve_eig_csv(source)
    raw = pd.read_csv(csv_path, skipinitialspace=True)
    raw.columns = [str(column).strip() for column in raw.columns]
    if raw.empty:
        msg = f"Palace eigenmode CSV is empty: {csv_path}"
        raise ValueError(msg)
    columns = [str(column) for column in raw.columns]

    mode_col = _find_eigenmode_column(columns, ("m", "mode"), default=columns[0])
    real_col = _find_eigenmode_column(
        columns,
        ("re{f", "re(f", "real"),
        default=columns[1] if len(columns) > 1 else None,
    )
    imag_col = _find_eigenmode_column(
        columns,
        ("im{f", "im(f", "imag"),
        default=columns[2] if len(columns) > 2 else None,
    )
    q_col = _find_eigenmode_column(columns, ("q",), default=None)
    backward_error_col = _find_eigenmode_column(
        columns,
        ("bkwd", "backward"),
        default=None,
    )
    absolute_error_col = _find_eigenmode_column(
        columns,
        ("abs", "absolute"),
        default=None,
    )

    if real_col is None:
        msg = f"Palace eigenmode CSV has no real-frequency column: {csv_path}"
        raise ValueError(msg)

    frame = pd.DataFrame(
        {
            "mode_index": raw[mode_col].astype(float).round().astype(int),
            "freq_real_ghz": raw[real_col].astype(float),
            "freq_imag_ghz": (
                raw[imag_col].astype(float)
                if imag_col is not None
                else np.zeros(len(raw), dtype=float)
            ),
            "q": (
                raw[q_col].astype(float)
                if q_col is not None
                else np.full(len(raw), np.nan)
            ),
            "error_backward": (
                raw[backward_error_col].astype(float)
                if backward_error_col is not None
                else np.full(len(raw), np.nan)
            ),
            "error_absolute": (
                raw[absolute_error_col].astype(float)
                if absolute_error_col is not None
                else np.full(len(raw), np.nan)
            ),
        }
    )
    frame.attrs.update(
        {
            "csv_path": str(csv_path),
            "source_columns": tuple(columns),
        }
    )
    return Eigenmodes(source_path=csv_path, dataframe=frame)


def load_eigenmode_history(
    source: str | Path,
    *,
    include_final: bool = True,
) -> pd.DataFrame:
    """Load Palace Eigenmode AMR history from ``iteration*/eig.csv`` files.

    ``source`` may be a simulation directory or the ``results/palace`` directory.
    The returned frame includes source visibility columns and convergence
    deltas suitable for notebook reports.
    """
    import pandas as pd

    base = Path(source)
    if base.is_file():
        msg = "load_eigenmode_history() expects a simulation/output directory."
        raise ValueError(msg)

    output_dir = resolve_palace_output_dir(base)
    pass_frames: list[pd.DataFrame] = []
    last_table: pd.DataFrame | None = None

    for iteration_dir, iteration_index in _iteration_dirs(output_dir):
        csv_path = iteration_dir / "eig.csv"
        if not csv_path.exists():
            continue
        modes = load_eigenmodes(csv_path)
        pass_frames.append(
            eigenmodes_to_history_frame(
                modes,
                iteration_index=iteration_index,
                label=f"Pass {iteration_index}",
                is_final=False,
                source_kind="iteration",
                source_iteration=iteration_index,
            )
        )
        last_table = modes.dataframe

    if include_final:
        final_csv_path = output_dir / "eig.csv"
        if final_csv_path.exists():
            final_modes = load_eigenmodes(final_csv_path)
            if last_table is None or not _eigenmode_tables_match(
                last_table,
                final_modes.dataframe,
            ):
                iteration_index = (
                    1 if not pass_frames else _next_eigenmode_iteration(pass_frames)
                )
                pass_frames.append(
                    eigenmodes_to_history_frame(
                        final_modes,
                        iteration_index=iteration_index,
                        label="Final",
                        is_final=True,
                        source_kind="final",
                        source_iteration=None,
                    )
                )

    if not pass_frames:
        msg = f"No Palace eigenmode eig.csv files found under {output_dir}"
        raise FileNotFoundError(msg)

    return add_eigenmode_convergence_columns(pd.concat(pass_frames, ignore_index=True))


def summarize_eigenmode_history(
    history: pd.DataFrame,
    *,
    include_imaginary: bool = False,
) -> pd.DataFrame:
    """Build per-pass summary rows from an Eigenmode history frame."""
    frame = add_eigenmode_convergence_columns(history)
    if frame.empty:
        return frame

    summary = (
        frame.groupby(
            ["iteration_index", "label", "is_final"],
            sort=True,
            dropna=False,
        )
        .agg(
            n_modes=("mode_index", "nunique"),
            max_abs_delta_to_previous_mhz=("abs_delta_to_previous_mhz", "max"),
            max_abs_relative_delta_to_previous_percent=(
                "abs_relative_delta_to_previous_percent",
                "max",
            ),
            max_abs_delta_to_final_mhz=("abs_delta_to_final_mhz", "max"),
            max_abs_relative_delta_to_final_percent=(
                "abs_relative_delta_to_final_percent",
                "max",
            ),
            max_abs_imaginary_relative_delta_to_previous_percent=(
                "abs_imaginary_relative_delta_to_previous_percent",
                "max",
            ),
        )
        .reset_index()
    )
    if include_imaginary:
        summary["hfss_max_delta_freq_percent"] = summary[
            [
                "max_abs_relative_delta_to_previous_percent",
                "max_abs_imaginary_relative_delta_to_previous_percent",
            ]
        ].max(axis=1)
    else:
        summary["hfss_max_delta_freq_percent"] = summary[
            "max_abs_relative_delta_to_previous_percent"
        ]
    return summary.sort_values("iteration_index").reset_index(drop=True)


def _resolve_eig_csv(source: str | Path | dict) -> Path:
    if isinstance(source, dict):
        explicit = source.get("eig.csv")
        if explicit is not None:
            return Path(explicit)
        matches = [
            Path(value)
            for name, value in source.items()
            if str(name).endswith(".csv") and _EIG_CSV_RE.fullmatch(str(name))
        ]
        if matches:
            return matches[0]
        msg = f"Results dict has no eigenmode CSV entry: {list(source)}"
        raise FileNotFoundError(msg)

    path = Path(source)
    if path.is_file():
        return path
    found = _find_eig_csv(path)
    if found is None:
        msg = f"eig.csv not found in {path} or its subdirectories"
        raise FileNotFoundError(msg)
    return found


def _find_eig_csv(base: Path) -> Path | None:
    candidates = [
        base / "eig.csv",
        base / "results" / "palace" / "eig.csv",
        base / "palace" / "eig.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not base.exists():
        return None
    matches = [path for path in base.rglob("*.csv") if _EIG_CSV_RE.fullmatch(path.name)]
    return matches[0] if matches else None


def _find_eigenmode_column(
    columns: list[str] | tuple[str, ...],
    patterns: tuple[str, ...],
    *,
    default: str | None,
) -> str | None:
    normalized_patterns = tuple(pattern.lower() for pattern in patterns)
    for column in columns:
        lower = column.lower().strip()
        if any(pattern in lower for pattern in normalized_patterns):
            return column
    return default


def eigenmodes_to_history_frame(
    eigenmodes: Eigenmodes,
    *,
    iteration_index: int,
    label: str,
    is_final: bool,
    source_kind: str,
    source_iteration: int | None,
) -> pd.DataFrame:
    """Return one Eigenmode table annotated with AMR source metadata."""
    frame = eigenmodes.to_report_dataframe()
    frame.insert(0, "source_path", str(eigenmodes.source_path))
    frame.insert(0, "source_iteration", source_iteration)
    frame.insert(0, "source_kind", source_kind)
    frame.insert(0, "is_final", is_final)
    frame.insert(0, "label", label)
    frame.insert(0, "iteration_index", iteration_index)
    return cast("pd.DataFrame", frame)


def add_eigenmode_convergence_columns(history: pd.DataFrame) -> pd.DataFrame:
    """Add pass-to-pass and pass-to-final convergence columns."""
    frame = history.copy()
    if frame.empty:
        return frame

    frame = frame.sort_values(["mode_index", "iteration_index"]).reset_index(drop=True)
    grouped = frame.groupby("mode_index", sort=True)

    previous_frequency = grouped["frequency_ghz"].shift(1)
    final_frequency = grouped["frequency_ghz"].transform("last")
    previous_imaginary_frequency = grouped["imaginary_frequency_ghz"].shift(1)
    final_imaginary_frequency = grouped["imaginary_frequency_ghz"].transform("last")

    frame["delta_to_previous_ghz"] = frame["frequency_ghz"] - previous_frequency
    frame["delta_to_previous_mhz"] = frame["delta_to_previous_ghz"] * 1.0e3
    frame["abs_delta_to_previous_mhz"] = frame["delta_to_previous_mhz"].abs()
    frame["relative_delta_to_previous_percent"] = (
        frame["delta_to_previous_ghz"] / previous_frequency.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_previous_percent"] = frame[
        "relative_delta_to_previous_percent"
    ].abs()

    frame["delta_to_final_ghz"] = frame["frequency_ghz"] - final_frequency
    frame["delta_to_final_mhz"] = frame["delta_to_final_ghz"] * 1.0e3
    frame["abs_delta_to_final_mhz"] = frame["delta_to_final_mhz"].abs()
    frame["relative_delta_to_final_percent"] = (
        frame["delta_to_final_ghz"] / final_frequency.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_final_percent"] = frame[
        "relative_delta_to_final_percent"
    ].abs()

    frame["imaginary_delta_to_previous_ghz"] = (
        frame["imaginary_frequency_ghz"] - previous_imaginary_frequency
    )
    frame["imaginary_delta_to_previous_mhz"] = (
        frame["imaginary_delta_to_previous_ghz"] * 1.0e3
    )
    frame["abs_imaginary_delta_to_previous_mhz"] = frame[
        "imaginary_delta_to_previous_mhz"
    ].abs()
    frame["imaginary_relative_delta_to_previous_percent"] = (
        frame["imaginary_delta_to_previous_ghz"] / previous_imaginary_frequency.abs()
    ) * 1.0e2
    frame["abs_imaginary_relative_delta_to_previous_percent"] = frame[
        "imaginary_relative_delta_to_previous_percent"
    ].abs()

    frame["imaginary_delta_to_final_ghz"] = (
        frame["imaginary_frequency_ghz"] - final_imaginary_frequency
    )
    frame["imaginary_delta_to_final_mhz"] = (
        frame["imaginary_delta_to_final_ghz"] * 1.0e3
    )
    frame["abs_imaginary_delta_to_final_mhz"] = frame[
        "imaginary_delta_to_final_mhz"
    ].abs()
    frame["imaginary_relative_delta_to_final_percent"] = (
        frame["imaginary_delta_to_final_ghz"] / final_imaginary_frequency.abs()
    ) * 1.0e2
    frame["abs_imaginary_relative_delta_to_final_percent"] = frame[
        "imaginary_relative_delta_to_final_percent"
    ].abs()
    return frame.sort_values(["iteration_index", "mode_index"]).reset_index(drop=True)


def _next_eigenmode_iteration(pass_frames: list[pd.DataFrame]) -> int:
    max_index = 0
    for frame in pass_frames:
        max_index = max(max_index, int(cast("Any", frame["iteration_index"].max())))
    return max_index + 1


def _eigenmode_tables_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = [
        "mode_index",
        "freq_real_ghz",
        "freq_imag_ghz",
        "q",
        "error_backward",
        "error_absolute",
    ]
    return (
        left.loc[:, columns]
        .reset_index(drop=True)
        .equals(right.loc[:, columns].reset_index(drop=True))
    )


def find_eigenmode_iteration_csvs(source: str | Path) -> tuple[Path, ...]:
    """Return AMR iteration ``eig.csv`` files below a Palace output tree."""
    path = Path(source)
    if path.is_file() or not path.exists():
        return ()
    try:
        output_dir = resolve_palace_output_dir(path)
    except FileNotFoundError:
        return ()
    return tuple(
        iteration_dir / "eig.csv"
        for iteration_dir, _ in _iteration_dirs(output_dir)
        if (iteration_dir / "eig.csv").exists()
    )


def empty_eigenmode_history() -> pd.DataFrame:
    """Return an empty Eigenmode AMR history frame with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "iteration_index",
            "label",
            "is_final",
            "source_kind",
            "source_iteration",
            "source_path",
            "mode_index",
            "frequency_ghz",
            "imaginary_frequency_ghz",
            "q_factor",
            "backward_error",
            "absolute_error",
        ]
    )


def empty_eigenmode_pass_summary() -> pd.DataFrame:
    """Return an empty Eigenmode AMR pass summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "iteration_index",
            "label",
            "is_final",
            "n_modes",
            "max_abs_delta_to_previous_mhz",
            "max_abs_relative_delta_to_previous_percent",
            "max_abs_delta_to_final_mhz",
            "max_abs_relative_delta_to_final_percent",
            "max_abs_imaginary_relative_delta_to_previous_percent",
            "hfss_max_delta_freq_percent",
        ]
    )


def eigenmode_history_source(source: str | Path | dict, eig_csv_path: Path) -> Path:
    """Return the directory used to discover Eigenmode AMR history sidecars."""
    if isinstance(source, dict):
        return eig_csv_path.parent

    path = Path(source)
    return path.parent if path.is_file() else path


def _iteration_dirs(output_dir: Path) -> tuple[tuple[Path, int], ...]:
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    dirs: list[tuple[Path, int]] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = _ITERATION_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        dirs.append((path, int(match.group(1))))
    return tuple(sorted(dirs, key=lambda item: item[1]))
