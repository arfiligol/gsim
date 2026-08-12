"""Optional report-table loading helpers for Palace report composition.

Problem report loaders use these helpers for optional indexed Palace CSVs such
as ``domain-E.csv``, ``surface-Q.csv``, and ``port-EPR.csv``. The helpers only
decide source status and invoke the supplied primitive loader; they do not own
the data model for any problem type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsim.palace.resolve.assembly.common import report_source_row
from gsim.palace.resolve.sources.path_utils import find_file, resolve_report_csv

if TYPE_CHECKING:
    import pandas as pd


def load_optional_report_table(
    source: str | Path | dict,
    csv_name: str,
    *,
    loader: Any,
    empty_factory: Any,
    index_map_path: Path | None,
    index_map_present: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one optional indexed Palace report table with source status."""
    csv_path = find_optional_report_csv(source, csv_name)
    if csv_path is None or not csv_path.exists():
        return empty_factory(), report_source_row(
            csv_name,
            csv_path,
            required=False,
            present=False,
            loaded=False,
            message="not found",
        )

    if not index_map_present:
        return empty_factory(), report_source_row(
            csv_name,
            csv_path,
            required=False,
            present=True,
            loaded=False,
            message="missing palace_index_map.json",
        )

    report_source = {
        csv_name: csv_path,
        "palace_index_map.json": index_map_path,
    }
    loaded = loader(report_source, index_map_path=index_map_path)
    return loaded, report_source_row(
        csv_name,
        csv_path,
        required=False,
        present=True,
        loaded=True,
        message="loaded",
    )


def find_optional_report_csv(source: str | Path | dict, csv_name: str) -> Path | None:
    """Find an optional report CSV without raising when absent."""
    try:
        return resolve_report_csv(source, csv_name)
    except (FileNotFoundError, ValueError):
        pass

    if isinstance(source, dict):
        for value in source.values():
            path = Path(value)
            root = path.parent if path.suffix else path
            found = find_file(root, csv_name) if root.exists() else None
            if found is not None:
                return found
        return None

    path = Path(source)
    root = path.parent if path.is_file() else path
    if not root.exists():
        return None
    return find_file(root, csv_name)


__all__ = ["find_optional_report_csv", "load_optional_report_table"]
