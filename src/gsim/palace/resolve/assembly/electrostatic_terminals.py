"""Terminal-matrix report helpers for Electrostatic Palace reports.

This module adapts raw Electrostatic terminal-matrix typed data to the report
composition layer. It records source status rows, handles optional matrices,
and synthesizes AMR history views when only the final matrix is available.
It does not define ``TerminalMatrix`` semantics, choose visualizers, or compose
the final ``ElectrostaticReport``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsim.palace.resolve.assembly.common import report_source_row
from gsim.palace.resolve.derived.terminal_matrices import (
    add_terminal_matrix_convergence_columns,
    find_terminal_matrix_iteration_csvs,
    load_terminal_matrix,
    load_terminal_matrix_history,
    summarize_terminal_matrix_history,
    terminal_matrix_history_source,
    terminal_matrix_to_history_frame,
)
from gsim.palace.resolve.derived.terminal_matrices import (
    empty_terminal_matrix_history as terminal_empty_matrix_history,
)
from gsim.palace.resolve.derived.terminal_matrices import (
    empty_terminal_matrix_pass_summary as terminal_empty_matrix_pass_summary,
)
from gsim.palace.resolve.loaders.terminal import (
    TERMINAL_MATRIX_SPECS,
    normalize_terminal_matrix_kind,
    read_terminal_matrix_csv,
)
from gsim.palace.resolve.sources.path_utils import find_file
from gsim.palace.results.electrostatic import TerminalMatrixConvergence

if TYPE_CHECKING:
    import pandas as pd

    from gsim.palace.results.electrostatic import TerminalMatrix


@dataclass(frozen=True, kw_only=True)
class ElectrostaticTerminalReportParts:
    """Terminal-matrix typed data ready for Electrostatic report assembly."""

    terminal_c: TerminalMatrix
    terminal_cm: TerminalMatrix | None
    terminal_cinv: TerminalMatrix | None
    terminal_c_convergence: TerminalMatrixConvergence
    terminal_cm_convergence: TerminalMatrixConvergence
    terminal_cinv_convergence: TerminalMatrixConvergence
    source_rows: tuple[dict[str, Any], ...]


def load_electrostatic_terminal_report_parts(
    source: str | Path | dict,
    *,
    index_map_path: Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    include_history: bool,
) -> ElectrostaticTerminalReportParts:
    """Load the complete terminal-matrix section for an Electrostatic report."""
    source_rows: list[dict[str, Any]] = []

    terminal_c = load_terminal_matrix_for_report(
        source,
        "C",
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        source_rows=source_rows,
        required=True,
    )
    if terminal_c is None:
        msg = "Required electrostatic terminal-C.csv not found"
        raise FileNotFoundError(msg)
    terminal_cm = load_terminal_matrix_for_report(
        source,
        "Cm",
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        source_rows=source_rows,
        required=False,
    )
    terminal_cinv = load_terminal_matrix_for_report(
        source,
        "Cinv",
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        source_rows=source_rows,
        required=False,
    )

    terminal_c_history = load_terminal_matrix_history_for_report(
        source,
        terminal_c,
        include_history=include_history,
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        source_rows=source_rows,
    )
    terminal_cm_history = load_terminal_matrix_history_for_report(
        source,
        terminal_cm,
        include_history=include_history,
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        source_rows=source_rows,
    )
    terminal_cinv_history = load_terminal_matrix_history_for_report(
        source,
        terminal_cinv,
        include_history=include_history,
        index_map_path=index_map_path,
        terminal_names=terminal_names,
        source_rows=source_rows,
    )

    return ElectrostaticTerminalReportParts(
        terminal_c=terminal_c,
        terminal_cm=terminal_cm,
        terminal_cinv=terminal_cinv,
        terminal_c_convergence=TerminalMatrixConvergence(
            matrix_kind="C",
            history=terminal_c_history,
            pass_summary=terminal_matrix_pass_summary(terminal_c_history),
        ),
        terminal_cm_convergence=TerminalMatrixConvergence(
            matrix_kind="Cm",
            history=terminal_cm_history,
            pass_summary=terminal_matrix_pass_summary(terminal_cm_history),
        ),
        terminal_cinv_convergence=TerminalMatrixConvergence(
            matrix_kind="Cinv",
            history=terminal_cinv_history,
            pass_summary=terminal_matrix_pass_summary(terminal_cinv_history),
        ),
        source_rows=tuple(source_rows),
    )


def load_terminal_matrix_for_report(
    source: str | Path | dict,
    matrix_kind: str,
    *,
    index_map_path: Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    source_rows: list[dict[str, Any]],
    required: bool,
) -> TerminalMatrix | None:
    """Load one terminal matrix and append its report source-status row."""
    kind = normalize_terminal_matrix_kind(matrix_kind)
    csv_name = str(TERMINAL_MATRIX_SPECS[kind]["file_name"])
    csv_path = find_optional_terminal_matrix_csv(source, kind)
    if csv_path is None or not csv_path.exists():
        source_rows.append(
            report_source_row(
                csv_name,
                csv_path,
                required=required,
                present=False,
                loaded=False,
                message="not found",
            )
        )
        if required:
            msg = f"Required electrostatic {csv_name} not found"
            raise FileNotFoundError(msg)
        return None

    resolved_terminal_names = terminal_names
    if resolved_terminal_names is None and index_map_path is None:
        terminal_count = len(read_terminal_matrix_csv(csv_path))
        resolved_terminal_names = tuple(
            f"T{index}" for index in range(1, terminal_count + 1)
        )

    matrix = load_terminal_matrix(
        {csv_name: csv_path, "palace_index_map.json": index_map_path}
        if index_map_path is not None
        else csv_path,
        kind,
        index_map_path=index_map_path,
        terminal_names=resolved_terminal_names,
    )
    source_rows.append(
        report_source_row(
            csv_name,
            csv_path,
            required=required,
            present=True,
            loaded=True,
            message="loaded",
        )
    )
    return matrix


def load_terminal_matrix_history_for_report(
    source: str | Path | dict,
    matrix: TerminalMatrix | None,
    *,
    include_history: bool,
    index_map_path: Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
    source_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Load one terminal matrix AMR history and append source-status rows."""
    import pandas as pd

    if matrix is None:
        return empty_terminal_matrix_history()

    csv_name = str(TERMINAL_MATRIX_SPECS[matrix.matrix_kind]["file_name"])
    history_source = terminal_matrix_history_source(source, matrix.source_path)
    iteration_paths = find_terminal_matrix_iteration_csvs(
        history_source,
        matrix.matrix_kind,
    )
    source_rows.append(
        report_source_row(
            f"iteration*/{csv_name}",
            None,
            required=False,
            present=bool(iteration_paths),
            loaded=include_history and bool(iteration_paths),
            message=(
                f"loaded {len(iteration_paths)} AMR iteration files"
                if include_history and iteration_paths
                else "history disabled"
                if not include_history
                else f"no AMR iteration {csv_name} files found"
            ),
        )
    )
    if not include_history:
        return empty_terminal_matrix_history()

    try:
        return load_terminal_matrix_history(
            history_source,
            matrix.matrix_kind,
            index_map_path=index_map_path,
            terminal_names=terminal_names or matrix.terminal_names,
            include_final=True,
        )
    except (FileNotFoundError, ValueError):
        final_history = terminal_matrix_to_history_frame(
            matrix,
            pass_index=1,
            label="Final",
            is_final=True,
        )
        return add_terminal_matrix_convergence_columns(pd.DataFrame(final_history))


def terminal_matrix_pass_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Return an empty or summarized terminal-matrix pass table."""
    return (
        empty_terminal_matrix_pass_summary()
        if history.empty
        else summarize_terminal_matrix_history(history)
    )


def find_optional_terminal_matrix_csv(
    source: str | Path | dict,
    matrix_kind: str,
) -> Path | None:
    """Find a terminal-matrix CSV without raising when it is absent."""
    csv_name = str(TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    if isinstance(source, dict):
        explicit = source.get(csv_name)
        if explicit is not None:
            return Path(explicit)
        for value in source.values():
            path = Path(value)
            root = path.parent if path.suffix else path
            found = find_file(root, csv_name) if root.exists() else None
            if found is not None:
                return found
        return None

    path = Path(source)
    if path.is_file():
        if path.name == csv_name:
            return path
        root = path.parent
    else:
        root = path
    if not root.exists():
        return None
    return find_file(root, csv_name)


def empty_terminal_matrix_history() -> pd.DataFrame:
    """Return the empty terminal-matrix history dataframe contract."""
    return terminal_empty_matrix_history()


def empty_terminal_matrix_pass_summary() -> pd.DataFrame:
    """Return the empty terminal-matrix pass-summary dataframe contract."""
    return terminal_empty_matrix_pass_summary()


__all__ = [
    "ElectrostaticTerminalReportParts",
    "empty_terminal_matrix_history",
    "empty_terminal_matrix_pass_summary",
    "find_optional_terminal_matrix_csv",
    "load_electrostatic_terminal_report_parts",
    "load_terminal_matrix_for_report",
    "load_terminal_matrix_history_for_report",
    "terminal_matrix_pass_summary",
]
