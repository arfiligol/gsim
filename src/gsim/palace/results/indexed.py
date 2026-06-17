"""Typed Palace indexed CSV data.

Palace postprocessing CSV files often encode mesh entity indices in column
names, such as ``p_surf[2]`` or ``E_elec[1]``. This module stores the decoded
column identity together with the loaded dataframe so higher-level reports can
show physical names without reparsing CSV headers.

This module does not discover files, parse Palace JSON sidecars, or compose
problem reports. Those responsibilities belong to ``gsim.palace.resolve`` and
``gsim.palace.results``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class IndexedCsvColumn:
    """Mapping from one Palace indexed CSV column to mesh identity."""

    original_name: str
    renamed_name: str
    quantity: str
    index: int
    section: str
    physical_name: str | None = None
    entry_name: str | None = None
    role: str | None = None
    attributes: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly audit row."""
        row: dict[str, Any] = {
            "original_name": self.original_name,
            "renamed_name": self.renamed_name,
            "quantity": self.quantity,
            "index": self.index,
            "section": self.section,
        }
        if self.physical_name is not None:
            row["physical_name"] = self.physical_name
        if self.entry_name is not None:
            row["entry_name"] = self.entry_name
        if self.role is not None:
            row["role"] = self.role
        if self.attributes:
            row["attributes"] = list(self.attributes)
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        if self.extra:
            row["extra"] = dict(self.extra)
        return row


@dataclass(frozen=True)
class IndexedCsv:
    """Palace indexed CSV with optional physical-name column labels."""

    source_path: Path
    section: str
    dataframe: pd.DataFrame
    columns: tuple[IndexedCsvColumn, ...]

    @property
    def column_map(self) -> tuple[dict[str, Any], ...]:
        """Return JSON-friendly column provenance rows."""
        return tuple(column.to_dict() for column in self.columns)


__all__ = ["IndexedCsv", "IndexedCsvColumn"]
