"""Pydantic models for Palace EM simulation configuration.

This module provides Pydantic v2 models for configuring Palace simulations,
offering validation, serialization, and a clean API.

Submodules:
    - stack: MaterialConfig (Layer/Stack are in gsim.common.stack)
    - ports: PortConfig, CPWPortConfig, PalacePort, TerminalConfig, WavePortConfig
    - regions: ActivatedRegion
    - simulation_layers: SimulationLayerCatalog
    - mesh: MeshConfig
    - numerical: NumericalConfig
    - problems: DrivenConfig, EigenmodeConfig, ElectrostaticConfig, etc.
    - results: ValidationResult
"""

from __future__ import annotations

from gsim.palace.models.mesh import MeshConfig
from gsim.palace.models.numerical import NumericalConfig
from gsim.palace.models.pec import PECBlockConfig
from gsim.palace.models.ports import (
    CPWPortConfig,
    PalacePort,
    PortConfig,
    PortGeometry,
    PortType,
    TerminalConfig,
    WavePortConfig,
)
from gsim.palace.models.problems import (
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    MagnetostaticConfig,
    TransientConfig,
)
from gsim.palace.models.regions import ActivatedRegion, ActivatedRegionRole
from gsim.palace.models.results import ValidationResult
from gsim.palace.models.simulation_layers import (
    SimulationLayer,
    SimulationLayerCatalog,
)
from gsim.palace.models.sources import CurrentSourceConfig, CurrentSourceElementConfig
from gsim.palace.models.stack import MaterialConfig
from gsim.palace.models.versions import (
    DEFAULT_PALACE_CONFIG_VERSION,
    SUPPORTED_PALACE_CONFIG_VERSIONS,
    PalaceConfigVersion,
)

__all__ = [
    "DEFAULT_PALACE_CONFIG_VERSION",
    "SUPPORTED_PALACE_CONFIG_VERSIONS",
    "ActivatedRegion",
    "ActivatedRegionRole",
    "CPWPortConfig",
    "CurrentSourceConfig",
    "CurrentSourceElementConfig",
    "DrivenConfig",
    "EigenmodeConfig",
    "ElectrostaticConfig",
    "MagnetostaticConfig",
    "MaterialConfig",
    "MeshConfig",
    "NumericalConfig",
    "PECBlockConfig",
    "PalaceConfigVersion",
    "PalacePort",
    "PortConfig",
    "PortGeometry",
    "PortType",
    "SimulationLayer",
    "SimulationLayerCatalog",
    "TerminalConfig",
    "TransientConfig",
    "ValidationResult",
    "WavePortConfig",
]
