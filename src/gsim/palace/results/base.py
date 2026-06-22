"""Small base classes for dataframe-backed Palace Typed Data.

Typed Data classes own simulation-result semantics. These bases only remove
repeated dataframe mechanics: copy-on-wrap, CSV persistence, and table-only
``visualize()`` output. They do not define plot semantics or require
``figures()`` hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from gsim.palace.display import DisplayValue

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class DataFrameResult:
    """Typed result backed by one canonical dataframe.

    Subclasses decide what the rows mean. This base only gives reviewers the
    common storage contract: incoming frames are copied, exported frames are
    copied, and CSV persistence writes that canonical table.
    """

    dataframe: pd.DataFrame

    def __post_init__(self) -> None:
        """Copy the incoming frame so wrappers do not mutate loader output."""
        object.__setattr__(self, "dataframe", self.dataframe.copy())

    @property
    def empty(self) -> bool:
        """Return whether the wrapped result table has no rows."""
        return bool(self.dataframe.empty)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the canonical result table."""
        return self.dataframe.copy()

    def save_csv(self, path: str | Path) -> Path:
        """Write the canonical result table as CSV and return the path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(output_path, index=False)
        return output_path


class NamedTableResult(DataFrameResult):
    """Table-only Typed Data with one named notebook display table."""

    table_name: ClassVar[str]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the only default table this result owns."""
        return {f"{self.table_name}_table": self.to_dataframe()}

    def visualize(self) -> dict[str, DisplayValue]:
        """Expose table-only data through the common Typed Data display API."""
        return cast("dict[str, DisplayValue]", self.tables())


__all__ = ["DataFrameResult", "NamedTableResult"]
