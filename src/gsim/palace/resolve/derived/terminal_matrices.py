"""Derived terminal-matrix typed data and convergence tables.

This module is the semantic boundary between primitive Palace terminal CSV
loading and Electrostatic report assembly. ``resolve.loaders.terminal`` owns
raw CSV parsing and label resolution; this module wraps those primitive frames
as ``TerminalMatrix`` typed data and derives AMR convergence tables. Problem
report policy stays in ``resolve.assembly``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gsim.palace.resolve.loaders.terminal import (
    TERMINAL_MATRIX_SPECS,
    find_terminal_matrix_final_csv,
    iteration_dirs,
    normalize_terminal_matrix_kind,
    read_terminal_matrix_csv,
    resolve_terminal_matrix_csv,
    resolve_terminal_matrix_labels,
)
from gsim.palace.resolve.sources.path_utils import resolve_palace_output_dir
from gsim.palace.results.electrostatic import TerminalMatrix

if TYPE_CHECKING:
    import pandas as pd


def load_terminal_matrix(
    source: str | Path | dict,
    matrix_kind: str = "C",
    *,
    index_map_path: str | Path | None = None,
    terminal_names: tuple[str, ...] | list[str] | None = None,
    display_scale: float | None = None,
    display_unit: str | None = None,
) -> TerminalMatrix:
    """Load a Palace electrostatic terminal matrix with named terminals."""
    kind = normalize_terminal_matrix_kind(matrix_kind)
    csv_path = resolve_terminal_matrix_csv(source, kind)
    return load_terminal_matrix_from_csv(
        csv_path,
        kind,
        label_source=source,
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        display_scale=display_scale,
        display_unit=display_unit,
    )


def load_terminal_matrix_from_csv(
    csv_path: Path,
    matrix_kind: str,
    *,
    label_source: str | Path | dict,
    index_map_path: str | Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    display_scale: float | None,
    display_unit: str | None,
) -> TerminalMatrix:
    """Wrap one resolved terminal-matrix CSV as ``TerminalMatrix`` typed data."""
    spec = TERMINAL_MATRIX_SPECS[matrix_kind]
    raw = read_terminal_matrix_csv(csv_path)
    labels = resolve_terminal_matrix_labels(
        label_source,
        raw.shape[0],
        index_map_path=index_map_path,
        terminal_names=terminal_names,
    )
    matrix = cast("pd.DataFrame", raw.copy())
    matrix.index = labels
    matrix.columns = labels
    matrix.attrs.update(
        {
            "matrix_kind": matrix_kind,
            "source_unit": spec["source_unit"],
            "csv_path": str(csv_path),
        }
    )
    return TerminalMatrix(
        source_path=csv_path,
        matrix_kind=matrix_kind,
        dataframe=matrix,
        terminal_names=tuple(labels),
        source_unit=spec["source_unit"],
        display_scale=(
            float(spec["display_scale"])
            if display_scale is None
            else float(display_scale)
        ),
        display_unit=spec["display_unit"] if display_unit is None else display_unit,
    )


def load_terminal_matrix_history(
    source: str | Path,
    matrix_kind: str = "C",
    *,
    index_map_path: str | Path | None = None,
    terminal_names: tuple[str, ...] | list[str] | None = None,
    include_final: bool = True,
    display_scale: float | None = None,
    display_unit: str | None = None,
) -> pd.DataFrame:
    """Load Palace electrostatic terminal matrices across AMR passes."""
    import pandas as pd

    label_source = Path(source)
    if label_source.is_file():
        msg = "load_terminal_matrix_history() expects a simulation/output directory."
        raise ValueError(msg)
    output_dir = resolve_palace_output_dir(label_source)

    kind = normalize_terminal_matrix_kind(matrix_kind)
    pass_frames: list[pd.DataFrame] = []
    last_raw: pd.DataFrame | None = None

    for iteration_dir, pass_index in iteration_dirs(output_dir):
        csv_path = iteration_dir / str(TERMINAL_MATRIX_SPECS[kind]["file_name"])
        if not csv_path.exists():
            continue
        matrix = load_terminal_matrix_from_csv(
            csv_path,
            kind,
            label_source=label_source,
            index_map_path=index_map_path,
            terminal_names=terminal_names,
            display_scale=display_scale,
            display_unit=display_unit,
        )
        pass_frames.append(
            terminal_matrix_to_history_frame(
                matrix,
                pass_index=pass_index,
                label=f"Pass {pass_index}",
                is_final=False,
            )
        )
        last_raw = read_terminal_matrix_csv(csv_path)

    if include_final:
        final_csv_path = find_terminal_matrix_final_csv(output_dir, kind)
        if final_csv_path is not None:
            final_raw = read_terminal_matrix_csv(final_csv_path)
            if last_raw is None or not terminal_matrices_match(last_raw, final_raw):
                matrix = load_terminal_matrix_from_csv(
                    final_csv_path,
                    kind,
                    label_source=label_source,
                    index_map_path=index_map_path,
                    terminal_names=terminal_names,
                    display_scale=display_scale,
                    display_unit=display_unit,
                )
                pass_index = (
                    1
                    if not pass_frames
                    else next_terminal_matrix_pass_index(pass_frames)
                )
                pass_frames.append(
                    terminal_matrix_to_history_frame(
                        matrix,
                        pass_index=pass_index,
                        label="Final",
                        is_final=True,
                    )
                )

    if not pass_frames:
        csv_name = TERMINAL_MATRIX_SPECS[kind]["file_name"]
        msg = f"No Palace electrostatic {csv_name} files found under {output_dir}"
        raise FileNotFoundError(msg)

    return add_terminal_matrix_convergence_columns(
        pd.concat(pass_frames, ignore_index=True)
    )


def terminal_matrix_history_source(source: str | Path | dict, csv_path: Path) -> Path:
    """Return the directory used to discover terminal-matrix AMR history."""
    if isinstance(source, dict):
        return csv_path.parent

    path = Path(source)
    return path.parent if path.is_file() else path


def find_terminal_matrix_iteration_csvs(
    source: str | Path,
    matrix_kind: str,
) -> tuple[Path, ...]:
    """Find AMR iteration CSVs for one terminal-matrix kind."""
    path = Path(source)
    if path.is_file() or not path.exists():
        return ()
    try:
        output_dir = resolve_palace_output_dir(path)
    except FileNotFoundError:
        return ()
    kind = normalize_terminal_matrix_kind(matrix_kind)
    csv_name = str(TERMINAL_MATRIX_SPECS[kind]["file_name"])
    return tuple(
        iteration_dir / csv_name
        for iteration_dir, _ in iteration_dirs(output_dir)
        if (iteration_dir / csv_name).exists()
    )


def summarize_terminal_matrix_history(history: pd.DataFrame) -> pd.DataFrame:
    """Build per-pass summary rows from terminal matrix history."""
    frame = add_terminal_matrix_convergence_columns(history)
    if frame.empty:
        return frame

    summary = (
        frame.groupby(
            ["matrix_kind", "pass_index", "label", "is_final", "display_unit"],
            sort=True,
        )
        .agg(
            n_elements=("element", "nunique"),
            n_diagonal_elements=("is_diagonal", "sum"),
            max_abs_value=("value_si", lambda column: column.abs().max()),
            max_abs_display_value=("display_value", lambda column: column.abs().max()),
            max_abs_delta_to_previous=("abs_delta_to_previous_si", "max"),
            max_abs_display_delta_to_previous=(
                "abs_display_delta_to_previous",
                "max",
            ),
            max_abs_relative_delta_to_previous_percent=(
                "abs_relative_delta_to_previous_percent",
                "max",
            ),
            max_abs_delta_to_final=("abs_delta_to_final_si", "max"),
            max_abs_display_delta_to_final=("abs_display_delta_to_final", "max"),
            max_abs_relative_delta_to_final_percent=(
                "abs_relative_delta_to_final_percent",
                "max",
            ),
        )
        .reset_index()
    )
    summary["n_off_diagonal_elements"] = (
        summary["n_elements"] - summary["n_diagonal_elements"]
    )
    return summary.sort_values(["matrix_kind", "pass_index"]).reset_index(drop=True)


def terminal_matrix_to_history_frame(
    matrix: TerminalMatrix,
    *,
    pass_index: int,
    label: str,
    is_final: bool,
) -> pd.DataFrame:
    """Return one terminal matrix annotated with AMR pass metadata."""
    frame = matrix.to_long_dataframe()
    frame.insert(0, "is_final", is_final)
    frame.insert(0, "label", label)
    frame.insert(0, "pass_index", pass_index)
    return frame


def add_terminal_matrix_convergence_columns(history: pd.DataFrame) -> pd.DataFrame:
    """Add pass-to-pass and pass-to-final terminal-matrix convergence columns."""
    frame = history.copy()
    if frame.empty:
        return frame

    frame = frame.sort_values(
        ["matrix_kind", "row_index", "column_index", "pass_index"]
    ).reset_index(drop=True)
    grouped = frame.groupby(["matrix_kind", "row_index", "column_index"], sort=True)

    previous_value = grouped["value_si"].shift(1)
    final_value = grouped["value_si"].transform("last")
    display_scale = frame["display_scale"]

    frame["delta_to_previous_si"] = frame["value_si"] - previous_value
    frame["abs_delta_to_previous_si"] = frame["delta_to_previous_si"].abs()
    frame["relative_delta_to_previous_percent"] = (
        frame["delta_to_previous_si"] / previous_value.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_previous_percent"] = frame[
        "relative_delta_to_previous_percent"
    ].abs()

    frame["delta_to_final_si"] = frame["value_si"] - final_value
    frame["abs_delta_to_final_si"] = frame["delta_to_final_si"].abs()
    frame["relative_delta_to_final_percent"] = (
        frame["delta_to_final_si"] / final_value.abs()
    ) * 1.0e2
    frame["abs_relative_delta_to_final_percent"] = frame[
        "relative_delta_to_final_percent"
    ].abs()

    frame["display_delta_to_previous"] = frame["delta_to_previous_si"] * display_scale
    frame["abs_display_delta_to_previous"] = frame["display_delta_to_previous"].abs()
    frame["display_delta_to_final"] = frame["delta_to_final_si"] * display_scale
    frame["abs_display_delta_to_final"] = frame["display_delta_to_final"].abs()
    return frame.sort_values(["pass_index", "row_index", "column_index"]).reset_index(
        drop=True
    )


def next_terminal_matrix_pass_index(pass_frames: list[pd.DataFrame]) -> int:
    """Return the next pass index after existing terminal-matrix pass frames."""
    max_index = 0
    for frame in pass_frames:
        max_index = max(max_index, int(cast("Any", frame["pass_index"].max())))
    return max_index + 1


def terminal_matrices_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return whether two terminal matrix tables are identical."""
    return left.equals(right)


def empty_terminal_matrix_history() -> pd.DataFrame:
    """Return an empty terminal-matrix AMR history with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "pass_index",
            "label",
            "is_final",
            "matrix_kind",
            "matrix_csv_path",
            "row_index",
            "column_index",
            "row_terminal",
            "column_terminal",
            "element",
            "is_diagonal",
            "value_si",
            "source_unit",
            "display_value",
            "display_unit",
            "display_scale",
            "delta_to_previous_si",
            "abs_delta_to_previous_si",
            "relative_delta_to_previous_percent",
            "abs_relative_delta_to_previous_percent",
            "delta_to_final_si",
            "abs_delta_to_final_si",
            "relative_delta_to_final_percent",
            "abs_relative_delta_to_final_percent",
            "display_delta_to_previous",
            "abs_display_delta_to_previous",
            "display_delta_to_final",
            "abs_display_delta_to_final",
        ]
    )


def empty_terminal_matrix_pass_summary() -> pd.DataFrame:
    """Return an empty terminal-matrix AMR pass summary with stable columns."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "matrix_kind",
            "pass_index",
            "label",
            "is_final",
            "display_unit",
            "n_elements",
            "n_diagonal_elements",
            "max_abs_value",
            "max_abs_display_value",
            "max_abs_delta_to_previous",
            "max_abs_display_delta_to_previous",
            "max_abs_relative_delta_to_previous_percent",
            "max_abs_delta_to_final",
            "max_abs_display_delta_to_final",
            "max_abs_relative_delta_to_final_percent",
            "n_off_diagonal_elements",
        ]
    )


__all__ = [
    "add_terminal_matrix_convergence_columns",
    "empty_terminal_matrix_history",
    "empty_terminal_matrix_pass_summary",
    "find_terminal_matrix_iteration_csvs",
    "load_terminal_matrix",
    "load_terminal_matrix_from_csv",
    "load_terminal_matrix_history",
    "next_terminal_matrix_pass_index",
    "summarize_terminal_matrix_history",
    "terminal_matrices_match",
    "terminal_matrix_history_source",
    "terminal_matrix_to_history_frame",
]
