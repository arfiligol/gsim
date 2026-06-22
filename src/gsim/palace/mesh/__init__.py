"""3D mesh generation and reportable mesh artifacts for Palace EM simulation.

This module provides mesh generation directly from gdsfactory components
and Palace API data structures. The package root is intentionally limited to
notebook-facing mesh controls and artifact helpers; lower-level manifest row
models and typing aliases live in their owner modules.

Usage:
    from gsim.palace.mesh import generate_mesh, MeshConfig

    # Quick presets
    config = MeshConfig.coarse()   # Fast iteration
    config = MeshConfig.default()  # Balanced
    config = MeshConfig.fine()     # High accuracy

    # Or customize with overrides
    config = MeshConfig.coarse(margin=100.0, fmax=50e9)

    # Or full manual control
    config = MeshConfig(refined_mesh_size=3.0, max_mesh_size=200.0)

    result = generate_mesh(
        component=c,
        stack=stack,
        ports=ports,
        output_dir="./sim_output",
        refined_mesh_size=config.refined_mesh_size,
        max_mesh_size=config.max_mesh_size,
        margin_x=config.effective_margin_x,
        margin_y=config.effective_margin_y,
        fmax=config.fmax,
    )
"""

from __future__ import annotations

from gsim.palace.mesh.generator import MeshResult, generate_mesh
from gsim.palace.mesh.manifest import (
    MeshManifest,
)
from gsim.palace.mesh.postprocessing import (
    SurfaceFluxSpec,
    build_dielectric_interface_specs_from_assignments,
    build_dielectric_interface_specs_from_material_kinds,
    build_postprocessing_config_from_manifest,
    build_surface_epr_dielectric_specs,
)
from gsim.palace.mesh.surface_epr import (
    build_interface_surface_catalog,
)
from gsim.palace.models.mesh import MeshConfig

__all__ = [
    "MeshConfig",
    "MeshManifest",
    "MeshResult",
    "SurfaceFluxSpec",
    "build_dielectric_interface_specs_from_assignments",
    "build_dielectric_interface_specs_from_material_kinds",
    "build_interface_surface_catalog",
    "build_postprocessing_config_from_manifest",
    "build_surface_epr_dielectric_specs",
    "generate_mesh",
]
