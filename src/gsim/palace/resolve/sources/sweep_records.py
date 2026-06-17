"""Serialize Palace sweep benchmark records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gsim.palace._shared import json_ready


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sweep records to CSV with a deterministic column order."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def write_records_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sweep records to JSONL with JSON-safe values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(json_ready(row), sort_keys=True) + "\n")


__all__ = ["write_records_csv", "write_records_jsonl"]
