"""Supported Palace configuration version targets.

Palace changes its accepted configuration schema across releases. The version
model keeps that target explicit in notebook code and in run metadata, while
schema validation remains a separate final check after mesh-derived config
assembly.
"""

from __future__ import annotations

from typing import Literal

PalaceConfigVersion = Literal["0.15.0", "0.16.0", "0.16.1"]
SUPPORTED_PALACE_CONFIG_VERSIONS: tuple[PalaceConfigVersion, ...] = (
    "0.15.0",
    "0.16.0",
    "0.16.1",
)
DEFAULT_PALACE_CONFIG_VERSION: PalaceConfigVersion = "0.16.0"


def normalize_palace_config_version(version: str) -> PalaceConfigVersion:
    """Return a supported Palace config version or fail fast."""
    normalized = str(version).removeprefix("v")
    if normalized == "0.15.0":
        return "0.15.0"
    if normalized == "0.16.0":
        return "0.16.0"
    if normalized == "0.16.1":
        return "0.16.1"
    supported = ", ".join(SUPPORTED_PALACE_CONFIG_VERSIONS)
    raise ValueError(
        f"Unsupported Palace config version {version!r}. "
        f"Supported versions: {supported}."
    )


__all__ = [
    "DEFAULT_PALACE_CONFIG_VERSION",
    "SUPPORTED_PALACE_CONFIG_VERSIONS",
    "PalaceConfigVersion",
    "normalize_palace_config_version",
]
