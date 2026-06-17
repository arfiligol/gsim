"""Filesystem search helpers for Palace artifact resolution.

Resolve loaders use these helpers to find solver outputs in canonical Palace
run folders and explicit cloud result mappings. The module owns path-search
policy only; it does not parse files or create typed data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

_ITERATION_DIR_RE = re.compile(r"iteration(\d+)$")


def find_file(base: Path, name: str) -> Path | None:
    """Find ``name`` in canonical Palace run-folder locations."""
    candidates = [
        base / name,
        base / "results" / "palace" / name,
        base / "metadata" / name,
        base / "input" / name,
        base / "palace" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_report_csv(
    source: str | Path | Mapping[str, str | Path],
    csv_name: str | None,
    *,
    candidate_names: Iterable[str] = (),
) -> Path:
    """Resolve a Palace report CSV from a direct path, directory, or result dict.

    Args:
        source: Direct CSV path, result directory, Palace output directory, or
            cloud/local results dict.
        csv_name: Expected CSV filename. When omitted for a results dict,
            ``candidate_names`` must identify exactly one available entry.
        candidate_names: Candidate filenames used only when ``csv_name`` is
            omitted for a results dict.

    Returns:
        Path to the resolved CSV file.
    """
    if isinstance(source, Mapping):
        resolved_name = csv_name
        if resolved_name is None:
            candidates = [
                name for name in candidate_names if source.get(name) is not None
            ]
            if len(candidates) != 1:
                msg = (
                    "Pass csv_name= when a results dict contains zero or multiple "
                    "matching report CSVs."
                )
                raise ValueError(msg)
            resolved_name = candidates[0]
        csv_val = source.get(resolved_name)
        if csv_val is None:
            msg = f"Results dict has no {resolved_name!r} entry"
            raise FileNotFoundError(msg)
        return Path(csv_val)

    path = Path(source)
    if path.is_file():
        if csv_name is not None and path.name != csv_name:
            msg = f"CSV path {path} does not match csv_name={csv_name!r}"
            raise ValueError(msg)
        return path
    if csv_name is None:
        msg = "Pass csv_name= when source is a directory."
        raise ValueError(msg)
    found = find_file(path, csv_name)
    if found is None:
        msg = f"{csv_name} not found in {path} or its subdirectories"
        raise FileNotFoundError(msg)
    return found


def resolve_palace_output_dir(base: Path) -> Path:
    """Resolve a directory that may contain Palace final or AMR outputs."""
    candidates = [
        base,
        base / "results" / "palace",
        base / "palace",
    ]
    for candidate in candidates:
        if candidate.exists() and (
            (candidate / "eig.csv").exists()
            or any(
                path.is_dir() and _ITERATION_DIR_RE.fullmatch(path.name)
                for path in candidate.iterdir()
            )
        ):
            return candidate
    if not base.exists():
        raise FileNotFoundError(base)
    return base


__all__ = ["find_file", "resolve_palace_output_dir", "resolve_report_csv"]
