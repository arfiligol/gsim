"""Resolve generic Palace indexed CSV files into typed indexed tables.

Palace postprocessing files encode entity indices in column names. This module
loads those files, attaches index-map provenance, and returns ``IndexedCsv``
typed data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gsim.palace.resolve.loaders.index_maps import load_postprocessing_index_map
from gsim.palace.resolve.sources.path_utils import resolve_report_csv
from gsim.palace.results.indexed import IndexedCsv, IndexedCsvColumn

_CSV_INDEX_SECTIONS = {
    "domain-E.csv": "Domains.Postprocessing.Energy",
    "surface-Q.csv": "Boundaries.Postprocessing.Dielectric",
    "port-EPR.csv": "Boundaries.Postprocessing.SurfaceFlux",
}
_INDEXED_CSV_COLUMN_RE = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z0-9_]*)\[(?P<index>\d+)\](?P<suffix>.*)$"
)


def load_indexed_csv(
    source: str | Path | dict,
    csv_name: str | None = None,
    *,
    section: str | None = None,
    index_map_path: str | Path | None = None,
    rename_columns: bool = True,
) -> IndexedCsv:
    """Load a Palace indexed CSV and annotate columns from the index map.

    Palace postprocessing reports use columns such as ``E_elec[1] (J)`` and
    ``p_surf[2]``. This loader preserves the numeric report contract while
    linking each index back to the generated ``palace_index_map.json`` artifact.

    Args:
        source: CSV path, simulation directory, Palace output directory, or
            results dict.
        csv_name: CSV name when ``source`` is a directory or results dict.
        section: Palace index-map section. If omitted, common Palace report
            filenames infer the section.
        index_map_path: Optional explicit ``palace_index_map.json`` path.
        rename_columns: If true, indexed columns with known physical names are
            renamed from ``quantity[1]`` to ``quantity[physical-name]``.

    Returns:
        :class:`IndexedCsv` containing the DataFrame and column provenance rows.
    """
    import pandas as pd

    csv_path = resolve_report_csv(
        source,
        csv_name,
        candidate_names=_CSV_INDEX_SECTIONS,
    )
    resolved_section = section or _section_for_csv(csv_path.name)
    index_map = load_postprocessing_index_map(source, index_map_path=index_map_path)
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()

    rename_map: dict[str, str] = {}
    column_rows: list[IndexedCsvColumn] = []
    for column in frame.columns:
        parsed = _parse_indexed_csv_column(column)
        if parsed is None:
            continue
        quantity, index, suffix = parsed
        entry = index_map.entry_for_index(resolved_section, index)
        physical_name = None if entry is None else entry.primary_physical_name
        renamed = (
            f"{quantity}[{physical_name}]{suffix}"
            if rename_columns and physical_name is not None
            else column
        )
        if renamed != column:
            rename_map[column] = renamed
        column_rows.append(
            IndexedCsvColumn(
                original_name=column,
                renamed_name=renamed,
                quantity=quantity,
                index=index,
                section=resolved_section,
                physical_name=physical_name,
                entry_name=None if entry is None else entry.entry_name,
                role=None if entry is None else entry.role,
                attributes=() if entry is None else entry.attributes,
                metadata={} if entry is None else dict(entry.metadata),
                extra={} if entry is None else dict(entry.extra),
            )
        )

    if rename_map:
        frame = frame.rename(columns=rename_map)
    return IndexedCsv(
        source_path=csv_path,
        section=resolved_section,
        dataframe=frame,
        columns=tuple(column_rows),
    )


def _section_for_csv(csv_name: str) -> str:
    """Return the index-map section associated with a Palace CSV name."""
    try:
        return _CSV_INDEX_SECTIONS[csv_name]
    except KeyError:
        msg = (
            f"Cannot infer Palace index-map section for {csv_name!r}; "
            "pass section= explicitly."
        )
        raise ValueError(msg) from None


def _parse_indexed_csv_column(col: str) -> tuple[str, int, str] | None:
    """Parse ``quantity[index] suffix`` Palace report columns."""
    match = _INDEXED_CSV_COLUMN_RE.fullmatch(col.strip())
    if match is None:
        return None
    return (
        match.group("prefix"),
        int(match.group("index")),
        match.group("suffix"),
    )


def indexed_column_provenance(
    column: IndexedCsvColumn,
    *,
    index_name: str,
) -> dict[str, Any]:
    """Return physical-name provenance for one indexed Palace CSV column."""
    metadata = dict(column.metadata)
    source_name = _metadata_string(metadata, "source_name")
    if source_name is None:
        source_name = _metadata_string(metadata, "source_entry_name")
    if source_name is None:
        source_name = _metadata_string(metadata, "source")
    if source_name is None:
        source_name = (
            column.physical_name or column.entry_name or f"Index {column.index}"
        )
    physical_name = _metadata_string(metadata, "physical_name")
    if physical_name is None:
        physical_name = column.physical_name
    row: dict[str, Any] = {
        index_name: column.index,
        "section": column.section,
        "source_name": source_name,
        "physical_name": physical_name,
        "entry_name": column.entry_name,
        "role": column.role,
        "attributes": column.attributes,
    }
    if column.metadata:
        row["metadata"] = dict(column.metadata)
    interface_type = interface_type_for_column(column)
    if interface_type is not None:
        row["interface_type"] = interface_type
    postprocessing_type = column.extra.get("Type")
    if postprocessing_type is not None and interface_type is None:
        row["postprocessing_type"] = postprocessing_type
    return row


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    """Return a non-empty string metadata value for one key."""
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def interface_type_for_column(column: IndexedCsvColumn) -> str | None:
    """Return the dielectric interface type encoded by an indexed CSV column."""
    entry_type = column.extra.get("Type")
    if isinstance(entry_type, str) and entry_type in {"MA", "MS", "SA", "Default"}:
        return entry_type

    for name in (column.physical_name, column.entry_name):
        if not name:
            continue
        prefix = name.split(":", 1)[0]
        if prefix in {"MA", "MS", "SA"}:
            return prefix
    return None
