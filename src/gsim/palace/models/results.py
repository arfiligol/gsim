"""Validation model for Palace simulations.

Mesh generation returns ``gsim.palace.mesh.generator.MeshResult`` directly;
Resolve/results modules own typed solver outputs.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, ConfigDict, Field


def _supports_color_output() -> bool:
    """Return True when stdout is an interactive terminal supporting ANSI."""
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class ValidationResult(BaseModel):
    """Result of simulation configuration validation.

    Attributes:
        valid: Whether the configuration is valid
        errors: List of error messages
        warnings: List of warning messages
    """

    model_config = ConfigDict(validate_assignment=True)

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        """Returns True if the validation passed, False otherwise."""
        return self.valid

    def __str__(self) -> str:
        """Returns a formatted string representation of the validation results."""
        lines = []
        if self.valid:
            lines.append("Validation: PASSED")
        else:
            failed = "Validation: FAILED"
            if _supports_color_output():
                failed = f"\033[31m{failed}\033[0m"
            lines.append(failed)
        if self.errors:
            lines.append("Errors:")
            lines.extend([f"  - {e}" for e in self.errors])
        if self.warnings:
            lines.append("Warnings:")
            lines.extend([f"  - {w}" for w in self.warnings])
        return "\n".join(lines)


__all__ = ["ValidationResult"]
