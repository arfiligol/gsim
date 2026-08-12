"""Versioned Palace configuration validation.

The mesh/config generator assembles the final Palace JSON from simulation
state, mesh groups, ports, materials, and typed solver fragments. This module
performs the last-mile schema check against vendored official Palace schemas so
unsupported version-specific keys fail before a solver run starts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validators
from referencing import Registry, Resource

from gsim.palace.models.versions import (
    PalaceConfigVersion,
    normalize_palace_config_version,
)


class PalaceConfigValidationError(ValueError):
    """Raised when an assembled Palace config does not match its target schema."""


def _json_path(path: Iterable[str | int]) -> str:
    """Render a JSON-path-like location for a schema error."""
    parts = list(path)
    if not parts:
        return "$"
    rendered = "$"
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def _truncate_value(value: object, *, max_chars: int = 240) -> str:
    """Render a value within the configured diagnostic length."""
    rendered = repr(value)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[: max_chars - 3]}..."


@lru_cache
def schema_root_path(version: PalaceConfigVersion) -> Path:
    """Return the local directory containing schemas for a Palace version."""
    normalized = normalize_palace_config_version(version)
    return Path(str(files("gsim.palace.config_schemas").joinpath(normalized)))


@lru_cache
def palace_config_schema(version: PalaceConfigVersion) -> dict[str, Any]:
    """Load the top-level vendored Palace config schema."""
    schema_path = schema_root_path(version) / "config-schema.json"
    return json.loads(schema_path.read_text())


@lru_cache
def palace_subschema_properties(
    version: PalaceConfigVersion,
    *,
    section: str,
    nested: str | None = None,
) -> frozenset[str]:
    """Return schema property names for a top-level or nested config section."""
    normalized = normalize_palace_config_version(version)
    schema_path = schema_root_path(normalized) / "config" / f"{section.lower()}.json"
    schema = json.loads(schema_path.read_text())
    properties = schema["properties"]
    if nested is not None:
        properties = properties[nested]["properties"]
    return frozenset(str(key) for key in properties)


@lru_cache
def _palace_config_validator(version: PalaceConfigVersion) -> Any:
    """Build and cache the JSON-schema validator for one Palace version."""
    normalized = normalize_palace_config_version(version)
    schema_path = schema_root_path(normalized) / "config-schema.json"
    schema = dict(palace_config_schema(normalized))
    schema["$id"] = schema_path.as_uri()
    registry = Registry().with_resource(
        schema_path.as_uri(),
        Resource.from_contents(schema),
    )
    for child_schema_path in sorted(
        (schema_root_path(normalized) / "config").glob("*.json")
    ):
        child_schema = json.loads(child_schema_path.read_text())
        child_schema["$id"] = child_schema_path.as_uri()
        registry = registry.with_resource(
            child_schema_path.as_uri(),
            Resource.from_contents(child_schema),
        )
    validator_cls = validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema, registry=registry)


def validate_palace_config(
    config: dict[str, Any],
    *,
    palace_version: str,
) -> None:
    """Validate an assembled Palace config against a supported version schema."""
    normalized = normalize_palace_config_version(palace_version)
    validator = _palace_config_validator(normalized)
    errors = sorted(
        validator.iter_errors(config),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return

    first: ValidationError = errors[0]
    path = _json_path(first.absolute_path)
    value = _truncate_value(first.instance)
    raise PalaceConfigValidationError(
        "Palace config validation failed "
        f"for target version {normalized} at {path}: {first.message}. "
        f"Invalid value: {value}"
    )


__all__ = [
    "PalaceConfigValidationError",
    "palace_config_schema",
    "palace_subschema_properties",
    "schema_root_path",
    "validate_palace_config",
]
