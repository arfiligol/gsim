"""Primitive Palace Electrostatic terminal-matrix CSV loading.

This loader owns raw terminal CSV resolution, CSV parsing, and optional
terminal-label resolution. It intentionally does not construct ``TerminalMatrix``
typed data, derive AMR convergence tables, assemble reports, or render figures;
those responsibilities live in ``resolve.derived``, ``resolve.assembly``,
``results``, and ``display`` respectively.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pandas as pd

from gsim.palace.resolve.loaders.index_maps import load_postprocessing_index_map
from gsim.palace.resolve.sources.path_utils import find_file

_ITERATION_DIR_RE = re.compile(r"iteration(\d+)$")
_TERMINAL_MATRIX_COLUMN_RE = re.compile(r"\[i\]\[(?P<index>\d+)\]")

TERMINAL_MATRIX_SPECS = {
    "C": {
        "file_name": "terminal-C.csv",
        "source_unit": "F",
        "display_scale": 1.0e15,
        "display_unit": "fF",
    },
    "Cm": {
        "file_name": "terminal-Cm.csv",
        "source_unit": "F",
        "display_scale": 1.0e15,
        "display_unit": "fF",
    },
    "Cinv": {
        "file_name": "terminal-Cinv.csv",
        "source_unit": "1/F",
        "display_scale": 1.0,
        "display_unit": "1/F",
    },
}


def normalize_terminal_matrix_kind(matrix_kind: str) -> str:
    """Normalize a terminal-matrix kind alias to Palace's canonical label."""
    aliases = {
        "c": "C",
        "capacitance": "C",
        "mutual": "Cm",
        "cm": "Cm",
        "c_m": "Cm",
        "inverse": "Cinv",
        "cinv": "Cinv",
        "c_inv": "Cinv",
    }
    key = matrix_kind.strip()
    normalized = TERMINAL_MATRIX_SPECS.get(key)
    if normalized is not None:
        return key
    alias = aliases.get(key.lower())
    if alias is not None:
        return alias
    allowed = ", ".join(TERMINAL_MATRIX_SPECS)
    msg = (
        f"Unknown Palace terminal matrix kind {matrix_kind!r}; "
        f"expected one of {allowed}."
    )
    raise ValueError(msg)


def resolve_terminal_matrix_csv(source: str | Path | dict, matrix_kind: str) -> Path:
    """Resolve the final terminal-matrix CSV path for one matrix kind."""
    csv_name = str(TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    if isinstance(source, dict):
        csv_val = source.get(csv_name)
        if csv_val is None:
            msg = f"Results dict has no {csv_name!r} entry"
            raise FileNotFoundError(msg)
        return Path(csv_val)

    path = Path(source)
    if path.is_file():
        return path
    found = find_file(path, csv_name)
    if found is None:
        msg = f"{csv_name} not found in {path} or its subdirectories"
        raise FileNotFoundError(msg)
    return found


def find_terminal_matrix_final_csv(base: Path, matrix_kind: str) -> Path | None:
    """Return a final terminal-matrix CSV under common Palace output locations."""
    csv_name = str(TERMINAL_MATRIX_SPECS[matrix_kind]["file_name"])
    candidates = [
        base / csv_name,
        base / "results" / "palace" / csv_name,
        base / "palace" / csv_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_terminal_matrix_labels(
    source: str | Path | dict,
    terminal_count: int,
    *,
    index_map_path: str | Path | None,
    terminal_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Return terminal labels for one Palace terminal matrix."""
    if terminal_names is not None:
        labels = tuple(str(name) for name in terminal_names)
    else:
        labels = terminal_names_from_index_map(
            source,
            terminal_count,
            index_map_path=index_map_path,
        )

    if len(labels) != terminal_count:
        msg = (
            f"Terminal label count ({len(labels)}) does not match Palace matrix "
            f"size ({terminal_count})."
        )
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = f"Terminal labels must be unique: {labels!r}"
        raise ValueError(msg)
    return labels


def terminal_names_from_index_map(
    source: str | Path | dict,
    terminal_count: int,
    *,
    index_map_path: str | Path | None,
) -> tuple[str, ...]:
    """Return terminal labels from ``palace_index_map.json`` when available."""
    index_map = load_postprocessing_index_map(source, index_map_path=index_map_path)
    labels_by_index: dict[int, str] = {}
    for index in range(1, terminal_count + 1):
        entries = index_map.entry_for_index("Boundaries.Terminal", index)
        if entries is None:
            labels_by_index[index] = f"T{index}"
            continue
        terminal_name = entries.extra.get("terminal_name")
        labels_by_index[index] = (
            str(terminal_name)
            if terminal_name is not None
            else entries.primary_physical_name
        )
    return tuple(labels_by_index[index] for index in range(1, terminal_count + 1))


def parse_terminal_matrix_column_index(column: str) -> int:
    """Parse a Palace terminal-matrix column index from a CSV header."""
    match = _TERMINAL_MATRIX_COLUMN_RE.search(column.strip())
    if match is None:
        msg = f"Could not parse Palace electrostatic matrix column: {column!r}"
        raise ValueError(msg)
    return int(match.group("index"))


def iteration_dirs(output_dir: Path) -> tuple[tuple[Path, int], ...]:
    """Return Palace AMR iteration directories sorted by pass index."""
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


def read_terminal_matrix_csv(csv_path: Path) -> pd.DataFrame:
    """Read a Palace terminal matrix CSV and normalize header whitespace."""
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty or len(frame.columns) < 2:
        msg = f"Palace electrostatic matrix CSV is empty or incomplete: {csv_path}"
        raise ValueError(msg)

    row_column = frame.columns[0]
    row_indices = frame[row_column].astype(float).round().astype(int).tolist()
    matrix_columns = list(frame.columns[1:])
    column_indices = [
        parse_terminal_matrix_column_index(column) for column in matrix_columns
    ]

    if len(set(row_indices)) != len(row_indices):
        msg = f"Duplicate Palace matrix row indices in {csv_path}: {row_indices}"
        raise ValueError(msg)
    if len(set(column_indices)) != len(column_indices):
        msg = f"Duplicate Palace matrix column indices in {csv_path}: {column_indices}"
        raise ValueError(msg)
    if len(row_indices) != len(column_indices):
        msg = (
            f"Palace electrostatic matrix must be square; found "
            f"{len(row_indices)} rows and {len(column_indices)} columns in {csv_path}"
        )
        raise ValueError(msg)

    expected_indices = set(range(1, len(row_indices) + 1))
    if set(row_indices) != expected_indices or set(column_indices) != expected_indices:
        msg = (
            "Palace electrostatic matrix indices must be contiguous and 1-based "
            f"in {csv_path}"
        )
        raise ValueError(msg)

    matrix = frame[matrix_columns].astype(float)
    matrix.index = row_indices
    matrix.columns = column_indices
    return cast(
        "pd.DataFrame", matrix.sort_index().reindex(sorted(column_indices), axis=1)
    )


__all__ = [
    "TERMINAL_MATRIX_SPECS",
    "find_terminal_matrix_final_csv",
    "iteration_dirs",
    "normalize_terminal_matrix_kind",
    "parse_terminal_matrix_column_index",
    "read_terminal_matrix_csv",
    "resolve_terminal_matrix_csv",
    "resolve_terminal_matrix_labels",
    "terminal_names_from_index_map",
]
