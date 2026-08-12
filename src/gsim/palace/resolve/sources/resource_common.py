"""Shared resource parsing helpers for Palace resolve modules."""

from __future__ import annotations

import re
from typing import Any

_RESOURCE_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_resource_memory_bytes(value: Any) -> float | None:
    """Parse Palace or Slurm memory strings into bytes."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(
        rf"(?P<number>{_RESOURCE_NUMBER_RE})\s*(?P<unit>[KMGTPE]?)(?:i?B?)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    numeric = float(match.group("number"))
    unit = match.group("unit").upper()
    scale = {
        "": 1.0,
        "K": 1024.0,
        "M": 1024.0**2,
        "G": 1024.0**3,
        "T": 1024.0**4,
        "P": 1024.0**5,
        "E": 1024.0**6,
    }[unit]
    return numeric * scale
