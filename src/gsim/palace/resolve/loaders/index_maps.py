"""Resolve Palace postprocessing index-map sidecars.

Index maps connect Palace integer postprocessing indices back to physical names,
roles, and mesh attributes. This module returns dataframe summaries used by
primitive loaders and problem reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from gsim.palace.mesh.postprocessing import PostprocessingIndexMap

from gsim.palace._shared import (
    as_mapping as _as_mapping,
)
from gsim.palace._shared import (
    optional_int as _optional_int,
)
from gsim.palace._shared import (
    optional_str as _optional_str,
)
from gsim.palace._shared import (
    optional_str_pair as _optional_str_pair,
)
from gsim.palace.resolve.sources.run_artifacts import find_postprocessing_index_map


def load_postprocessing_index_map(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None = None,
) -> PostprocessingIndexMap:
    """Load ``palace_index_map.json`` as a postprocessing index map.

    Args:
        source: Simulation directory, Palace output directory, results dict, or
            any path near the generated index-map artifact.
        index_map_path: Optional explicit ``palace_index_map.json`` path.

    Returns:
        :class:`gsim.palace.mesh.postprocessing.PostprocessingIndexMap`.
    """
    from gsim.palace.mesh.postprocessing import (
        PostprocessingIndexEntry,
        PostprocessingIndexMap,
    )

    path = (
        Path(index_map_path)
        if index_map_path is not None
        else find_postprocessing_index_map(source)
    )
    if path is None or not path.exists():
        msg = "palace_index_map.json not found"
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text())
    entries = tuple(
        PostprocessingIndexEntry(
            section=str(row["section"]),
            index=int(row["index"]),
            entry_name=str(row.get("entry_name", row.get("physical_name", ""))),
            role=str(row.get("role", "")),
            attributes=tuple(int(v) for v in row.get("attributes", ())),
            entity_tags=tuple(int(v) for v in row.get("entity_tags", ())),
            physical_names=tuple(str(v) for v in row.get("physical_names", ())),
            dimension=_optional_int(row.get("dimension")),
            source=_optional_str(row.get("source")),
            interface_of=_optional_str_pair(row.get("interface_of")),
            exterior_of=_optional_str(row.get("exterior_of")),
            metadata=_as_mapping(row.get("metadata")),
            extra={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "section",
                    "index",
                    "entry_name",
                    "role",
                    "attributes",
                    "entity_tags",
                    "physical_names",
                    "dimension",
                    "source",
                    "interface_of",
                    "exterior_of",
                    "metadata",
                }
            },
        )
        for row in data.get("entries", ())
    )
    return PostprocessingIndexMap(
        entries=entries,
        schema_version=int(data.get("schema_version", 1)),
    )


def postprocessing_index_map_to_dataframe(
    index_map: PostprocessingIndexMap,
) -> pd.DataFrame:
    """Return a flat dataframe summary of a postprocessing index map."""
    import pandas as pd

    rows = [
        {"schema_version": index_map.schema_version, **row}
        for row in index_map.to_rows()
    ]
    if not rows:
        return empty_postprocessing_index_map_dataframe()
    return pd.DataFrame.from_records(rows)


def empty_postprocessing_index_map_dataframe() -> pd.DataFrame:
    """Return the empty dataframe contract for index-map report tables."""
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "schema_version",
            "section",
            "index",
            "entry_name",
            "role",
            "attributes",
            "physical_names",
            "dimension",
            "source",
            "interface_of",
            "exterior_of",
            "metadata",
        ]
    )


def find_optional_postprocessing_index_map_path(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None,
) -> Path | None:
    """Return an explicit or discovered index-map path when available."""
    if index_map_path is not None:
        return Path(index_map_path)
    return find_postprocessing_index_map(source)


def load_optional_postprocessing_index_map(
    source: str | Path | dict,
    *,
    index_map_path: str | Path | None,
) -> PostprocessingIndexMap | None:
    """Load an optional postprocessing index map without raising if absent."""
    resolved_index_map_path = find_optional_postprocessing_index_map_path(
        source,
        index_map_path=index_map_path,
    )
    if resolved_index_map_path is None:
        return None
    return load_postprocessing_index_map(
        source,
        index_map_path=resolved_index_map_path,
    )


__all__ = [
    "empty_postprocessing_index_map_dataframe",
    "find_optional_postprocessing_index_map_path",
    "load_optional_postprocessing_index_map",
    "load_postprocessing_index_map",
    "postprocessing_index_map_to_dataframe",
]
