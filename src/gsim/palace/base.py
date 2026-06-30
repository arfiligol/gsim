"""Shared runtime API for Palace simulation classes.

This module provides the common simulation lifecycle used by
``DrivenSim``, ``EigenmodeSim``, ``ElectrostaticSim``, and
``MagnetostaticSim``: geometry assignment, output management, mesh/config
generation, cloud submission, local Palace execution, and handoff-package
generation.

The base model does not own problem-specific physics settings, report
composition, local process command details, or notebook display. Subclasses
define the problem configuration, ``gsim.palace.run`` owns concrete execution
implementations, ``gsim.palace.resolve`` parses generated artifacts into typed
result data for review/report workflows, and typed report objects expose
notebook display helpers.
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.models import (
    ActivatedRegion,
    ActivatedRegionRole,
    CPWPortConfig,
    DrivenConfig,
    EigenmodeConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PalaceConfigVersion,
    PortConfig,
    SimulationLayerCatalog,
    TerminalConfig,
    WavePortConfig,
)
from gsim.palace.models.ports import PalaceDirectionInput
from gsim.palace.models.results import ValidationResult
from gsim.palace.models.versions import (
    DEFAULT_PALACE_CONFIG_VERSION,
    normalize_palace_config_version,
)
from gsim.palace.run.handoff import generate_palace_handoff_package
from gsim.palace.run.local import run_palace_local
from gsim.palace.run_folder import palace_run_folder, prepare_palace_run_folder
from gsim.palace.run_stage import PalaceRunHandle

if TYPE_CHECKING:
    from gdsfactory.component import Component
    from gdsfactory.technology import LayerStack as GfLayerStack

    from gsim.palace.handoff import (
        PalaceSlurmHandoffResult,
        PalaceSlurmProfileResolution,
    )
    from gsim.palace.mesh.generator import MeshResult
    from gsim.palace.mesh.postprocessing import PostprocessingConfig
    from gsim.palace.results.driven import SParams
    from gsim.palace.run_folder import PalaceRunFolder

logger = logging.getLogger(__name__)


def _default_refinement_config() -> dict[str, Any]:
    return {
        "Tol": 1.0e-2,
        "MaxIts": 0,
        "MaxSize": 0,
        "UpdateFraction": 0.7,
        "Nonconformal": False,
        "UniformLevels": 0,
        "Boxes": [],
        "Spheres": [],
    }


class PalaceSimBase(BaseModel):
    """Pydantic base model for all Palace simulation classes.

    This base owns the common simulation lifecycle and shared Pydantic state.
    Problem-specific subclasses add solver configuration and selection fields
    such as ports, terminals, or current sources.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    geometry: Geometry | None = None
    stack: LayerStack | None = None
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    palace_version: PalaceConfigVersion = DEFAULT_PALACE_CONFIG_VERSION
    refinement: dict[str, Any] = Field(default_factory=_default_refinement_config)
    output_formats: dict[str, Any] = Field(default_factory=dict)

    _output_dir: Path | None = PrivateAttr(default=None)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, float] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _simulation_layers: SimulationLayerCatalog | None = PrivateAttr(default=None)
    _activated_regions: dict[str, ActivatedRegion] = PrivateAttr(default_factory=dict)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)
    _last_postprocessing_config: Any = PrivateAttr(default=None)
    _job_id: str | None = PrivateAttr(default=None)

    if TYPE_CHECKING:
        # Subclasses provide these problem-specific fields. Keeping these
        # annotations inside TYPE_CHECKING gives shared methods useful types
        # without making Pydantic treat them as base-model fields.
        driven: DrivenConfig
        eigenmode: EigenmodeConfig
        ports: list[PortConfig]
        cpw_ports: list[CPWPortConfig]
        wave_ports: list[WavePortConfig]
        terminals: list[TerminalConfig]
        simulation_type: Literal[
            "driven", "eigenmode", "electrostatic", "magnetostatic"
        ]
        absorbing_boundary: bool

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    def set_output_dir(self, path: str | Path) -> None:
        """Set the output directory for mesh and config files.

        Args:
            path: Directory path for output files

        Example:
            >>> sim.set_output_dir("./palace-sim")
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path | None:
        """Get the current output directory."""
        return self._output_dir

    # -------------------------------------------------------------------------
    # Geometry methods
    # -------------------------------------------------------------------------

    def set_geometry(self, component: Component) -> None:
        """Set the gdsfactory component for simulation.

        Args:
            component: gdsfactory Component to simulate

        Example:
            >>> sim.set_geometry(my_component)
        """
        from gsim.common import Geometry

        self.geometry = Geometry(component=component)

    def set_simulation_layers(
        self, catalog: SimulationLayerCatalog | Mapping[str, Any] | None
    ) -> None:
        """Set PDK-declared simulation-only layers for authored solver sheets.

        The catalog identifies GDS layers that carry solver boundary geometry.
        It does not turn those polygons into material geometry and it does not
        configure ports by itself; ports still need ``add_port()`` or
        ``add_cpw_port()`` declarations.
        """
        self._simulation_layers = (
            None if catalog is None else SimulationLayerCatalog.model_validate(catalog)
        )

    @property
    def component(self) -> Component | None:
        """Get the current component (for backward compatibility)."""
        return self.geometry.component if self.geometry else None

    # -------------------------------------------------------------------------
    # Stack methods
    # -------------------------------------------------------------------------

    def set_stack(
        self,
        stack: LayerStack | GfLayerStack | None = None,
        *,
        yaml_path: str | Path | None = None,
        air_above: float | None = None,
        air_below: float | None = None,
        substrate_thickness: float = 2.0,
        include_substrate: bool = False,
        add_oxide_dielectric: bool = True,
        add_passivation_dielectric: bool = True,
        **kwargs,
    ) -> None:
        """Configure the layer stack.

        Three modes of use:

         1. **Active PDK** (default — auto-detects IHP, QPDK, etc.)::

             sim.set_stack(substrate_thickness=2.0)

        2. **YAML file**::

               sim.set_stack(yaml_path="custom_stack.yaml")

        3. **Custom stack** (advanced — pass a hand-built LayerStack)::

               sim.set_stack(my_layer_stack)

        Args:
            stack: Custom gsim or gdsfactory LayerStack. Direct inputs are
                copied into simulation-owned state and bypass lazy PDK
                extraction.
            yaml_path: Path to custom YAML stack file.
            air_above: Deprecated and ignored. Use set_airbox().
            air_below: Deprecated and ignored. Use set_airbox().
            substrate_thickness: Thickness below z=0 in um.
            include_substrate: Include lossy silicon substrate.
            add_oxide_dielectric: Add synthetic oxide background dielectric.
            add_passivation_dielectric: Add synthetic passivation dielectric.
            **kwargs: Additional args passed to extract_layer_stack.

        Example:
            >>> sim.set_stack(substrate_thickness=2.0)
        """
        if stack is not None:
            self.stack = self._copy_stack_input(stack)
            self._stack_kwargs = {"_prebuilt": True}
            return

        if air_above is not None or air_below is not None:
            logger.warning(
                "set_stack(air_above/air_below) is deprecated and ignored. "
                "Use set_airbox(margin_x=..., margin_y=..., "
                "z_above=..., z_below=...)."
            )

        self._stack_kwargs = {
            "yaml_path": yaml_path,
            "substrate_thickness": substrate_thickness,
            "include_substrate": include_substrate,
            "add_oxide_dielectric": add_oxide_dielectric,
            "add_passivation_dielectric": add_passivation_dielectric,
            **kwargs,
        }
        # Stack will be resolved lazily during mesh() or simulate()
        self.stack = None

    def _copy_stack_input(self, stack: LayerStack | GfLayerStack) -> LayerStack:
        """Return a simulation-owned copy of a direct stack input."""
        if isinstance(stack, LayerStack):
            return stack.model_copy(deep=True)

        if hasattr(stack, "layers"):
            from gsim.common.stack.extractor import extract_layer_stack

            converted = extract_layer_stack(
                stack,
                pdk_name=str(getattr(stack, "name", "custom")),
                include_substrate=True,
                add_oxide_dielectric=False,
                add_passivation_dielectric=False,
            )
            converted.dielectrics = []
            return converted

        raise TypeError(
            "set_stack(stack=...) expects a gsim.common.LayerStack or "
            "gdsfactory LayerStack input."
        )

    def set_airbox(
        self,
        *,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
    ) -> None:
        """Configure an explicit weak-priority airbox for meshing.

        This helper centralizes airbox controls that were previously split
        across mesh margin arguments and stack air thickness parameters.
        The resulting airbox is created as a dedicated dielectric volume
        with the weakest boolean-priority in mesh construction.

        Args:
            margin_x: Airbox x-margin around the design (um).
                Defaults to ``0.0`` when omitted.
            margin_y: Airbox y-margin around the design (um).
                Defaults to ``0.0`` when omitted.
            z_above: Airbox extension above the stack top (um).
                Defaults to ``0.0`` when omitted.
            z_below: Airbox extension below the stack bottom (um).
                Defaults to ``0.0`` when omitted.
        """
        if self._activated_regions:
            raise ValueError(
                "Explicit activated regions cannot be mixed with set_airbox(). "
                "Use activate_substrate(), activate_inter_die_vacuum(), and "
                "activate_outer_vacuum() without airbox configuration."
            )

        mx = 0.0 if margin_x is None else margin_x
        my = 0.0 if margin_y is None else margin_y
        za = 0.0 if z_above is None else z_above
        zb = 0.0 if z_below is None else z_below

        if mx < 0:
            raise ValueError("margin_x must be >= 0")
        if my < 0:
            raise ValueError("margin_y must be >= 0")
        if za < 0 or zb < 0:
            raise ValueError("z_above and z_below must be >= 0")

        # Keep mesh margin controls in sync for domain/port extents.
        mesh_config = getattr(self, "mesh_config", None)
        if mesh_config is not None:
            mesh_config.margin_x = mx
            mesh_config.margin_y = my

        # Store explicit airbox expansion for generator plumbing.
        self._airbox_config = {
            "margin_x": mx,
            "margin_y": my,
            "z_above": za,
            "z_below": zb,
        }

    def _apply_airbox_overrides(
        self,
        *,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
    ) -> None:
        """Route mesh-time airbox kwargs through set_airbox()."""
        if self._activated_regions:
            if z_above is not None or z_below is not None:
                raise ValueError(
                    "Explicit activated regions cannot be mixed with mesh-time "
                    "airbox z_above/z_below kwargs."
                )
            return

        if (
            margin_x is None
            and margin_y is None
            and z_above is None
            and z_below is None
        ):
            return

        current = self._airbox_config or {}
        self.set_airbox(
            margin_x=margin_x if margin_x is not None else current.get("margin_x"),
            margin_y=margin_y if margin_y is not None else current.get("margin_y"),
            z_above=z_above if z_above is not None else current.get("z_above"),
            z_below=z_below if z_below is not None else current.get("z_below"),
        )

    def activate_substrate(
        self,
        layer: str,
        *,
        die: str | None = None,
        margin_x: float = 0.0,
        margin_y: float = 0.0,
        material: str | None = None,
    ) -> None:
        """Activate a stack substrate layer as an explicit Palace region."""
        self._activate_region(
            layer,
            role="substrate",
            die=die,
            margin_x=margin_x,
            margin_y=margin_y,
            material=material,
        )

    def activate_inter_die_vacuum(
        self,
        layer: str = "D0_TO_D1_GAP",
        *,
        lower_die: str = "D0",
        upper_die: str = "D1",
        margin_x: float = 0.0,
        margin_y: float = 0.0,
        material: str | None = None,
    ) -> None:
        """Activate the inter-die vacuum gap as an explicit Palace region."""
        self._activate_region(
            layer,
            role="inter_die_vacuum",
            lower_die=lower_die,
            upper_die=upper_die,
            margin_x=margin_x,
            margin_y=margin_y,
            material=material,
        )

    def activate_outer_vacuum(
        self,
        layer: str = "OUTER_VACUUM",
        *,
        margin_x: float = 0.0,
        margin_y: float = 0.0,
        z_above: float = 0.0,
        z_below: float = 0.0,
        material: str | None = None,
    ) -> None:
        """Activate the outer vacuum as an explicit Palace region."""
        self._activate_region(
            layer,
            role="outer_vacuum",
            margin_x=margin_x,
            margin_y=margin_y,
            z_above=z_above,
            z_below=z_below,
            material=material,
        )

    def _activate_region(
        self,
        layer: str,
        *,
        role: ActivatedRegionRole,
        die: str | None = None,
        lower_die: str | None = None,
        upper_die: str | None = None,
        margin_x: float = 0.0,
        margin_y: float = 0.0,
        z_above: float = 0.0,
        z_below: float = 0.0,
        material: str | None = None,
    ) -> None:
        """Store sim-owned region activation intent."""
        if self._airbox_config:
            raise ValueError(
                "Explicit activated regions cannot be mixed with set_airbox(). "
                "Remove airbox configuration before activating stack regions."
            )

        region = ActivatedRegion(
            layer=layer,
            role=role,
            die=die,
            lower_die=lower_die,
            upper_die=upper_die,
            margin_x=margin_x,
            margin_y=margin_y,
            z_above=z_above,
            z_below=z_below,
            material=material,
        )
        existing = self._activated_regions.get(region.layer)
        if existing is not None and existing.role != region.role:
            raise ValueError(
                f"Stack layer '{region.layer}' is already activated as "
                f"{existing.role!r}."
            )
        self._activated_regions[region.layer] = region

    def _activated_region_values(self) -> tuple[ActivatedRegion, ...]:
        """Return activated regions in deterministic layer-name order."""
        return tuple(
            self._activated_regions[layer] for layer in sorted(self._activated_regions)
        )

    def _reject_explicit_region_airbox_config(self, mesh_config: MeshConfig) -> None:
        """Reject stored mesh airbox expansion in explicit-region mode."""
        if self._activated_regions and mesh_config.airbox_margin > 0:
            raise ValueError(
                "Explicit activated regions cannot be mixed with mesh airbox "
                "controls. Configure region margins with activate_substrate(), "
                "activate_inter_die_vacuum(), or activate_outer_vacuum()."
            )

    # -------------------------------------------------------------------------
    # Material methods
    # -------------------------------------------------------------------------

    def set_material(
        self,
        name: str,
        *,
        material_type: (
            Literal["conductor", "dielectric", "semiconductor"] | None
        ) = None,
        conductivity: float | None = None,
        permittivity: float | None = None,
        loss_tangent: float | None = None,
    ) -> None:
        """Override or add material properties.

        Args:
            name: Material name
            material_type: Material type (conductor, dielectric, semiconductor)
            conductivity: Conductivity in S/m (for conductors)
            permittivity: Relative permittivity (for dielectrics)
            loss_tangent: Dielectric loss tangent

        Example:
            >>> sim.set_material(
            ...     "aluminum", material_type="conductor", conductivity=3.8e7
            ... )
            >>> sim.set_material("sio2", material_type="dielectric", permittivity=3.9)
        """
        # Determine type if not provided
        resolved_type = material_type
        if resolved_type is None:
            if conductivity is not None and conductivity > 1e4:
                resolved_type = "conductor"
            elif permittivity is not None:
                resolved_type = "dielectric"
            else:
                resolved_type = "dielectric"

        self.materials[name] = MaterialConfig(
            type=resolved_type,
            conductivity=conductivity,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
        )

    def add_pec(
        self,
        *,
        gds_layer: tuple[int, int],
        from_layer: str,
        to_layer: str,
    ) -> None:
        """Add a PEC block between two stack layers.

        The PEC block is a user-drawn polygon on a GDS layer that gets
        extruded between ``from_layer.zmin`` and ``to_layer.zmax`` and
        treated as a PEC boundary. This is a common practice in commercial
        EM solvers for connecting ground planes across metal layers at
        port boundaries.

        Args:
            gds_layer: GDS layer tuple where the PEC polygon is drawn.
            from_layer: Stack layer name — extrusion starts at this layer's zmin.
            to_layer: Stack layer name — extrusion ends at this layer's zmax.

        Example:
            >>> sim.add_pec(
            ...     gds_layer=(65000, 0), from_layer="metal1", to_layer="topmetal2"
            ... )
        """
        from gsim.palace.models.pec import PECBlockConfig

        self._pec_blocks.append(
            PECBlockConfig(
                gds_layer=gds_layer,
                from_layer=from_layer,
                to_layer=to_layer,
            )
        )

    def set_numerical(
        self,
        *,
        order: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: Literal[
            "AMS",
            "BoomerAMG",
            "SuperLU",
            "MUMPS",
            "STRUMPACK",
            "Jacobi",
            "Default",
        ] = "Default",
        preconditioner: Literal["Default", "AMS", "BoomerAMG"] = "Default",
        device: Literal["CPU", "GPU", "Debug"] = "CPU",
    ) -> None:
        """Configure numerical solver parameters.

        Args:
            order: Finite element order (1-4)
            tolerance: Linear solver tolerance
            max_iterations: Maximum solver iterations
            solver_type: Linear solver type
            preconditioner: Preconditioner type
            device: Compute device (CPU or GPU)

        Example:
            >>> sim.set_numerical(order=3, tolerance=1e-8)
        """
        self.numerical = NumericalConfig(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
        )

    def set_palace_version(self, version: str) -> None:
        """Set the target Palace configuration schema version.

        The target version controls config schema validation and is recorded in
        handoff/local-run metadata. It does not change the mesh-derived
        ``Domains`` or ``Boundaries`` ownership in the config generator.
        """
        self.palace_version = normalize_palace_config_version(version)

    def set_refinement(
        self,
        *,
        tol: float | None = None,
        max_its: int | None = None,
        max_size: float | None = None,
        update_fraction: float | None = None,
        nonconformal: bool | None = None,
        max_nc_levels: int | None = None,
        maximum_imbalance: float | None = None,
        save_adapt_iterations: bool | None = None,
        save_adapt_mesh: bool | None = None,
        uniform_levels: int | None = None,
        serial_uniform_levels: int | None = None,
        boxes: Sequence[Mapping[str, Any]] | None = None,
        spheres: Sequence[Mapping[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure Palace ``Model.Refinement`` through a JSON fragment."""
        refinement = _default_refinement_config()
        updates = {
            "Tol": tol,
            "MaxIts": max_its,
            "MaxSize": max_size,
            "UpdateFraction": update_fraction,
            "Nonconformal": nonconformal,
            "MaxNCLevels": max_nc_levels,
            "MaximumImbalance": maximum_imbalance,
            "SaveAdaptIterations": save_adapt_iterations,
            "SaveAdaptMesh": save_adapt_mesh,
            "UniformLevels": uniform_levels,
            "SerialUniformLevels": serial_uniform_levels,
            "Boxes": None if boxes is None else [dict(item) for item in boxes],
            "Spheres": None if spheres is None else [dict(item) for item in spheres],
        }
        refinement.update(
            {key: value for key, value in updates.items() if value is not None}
        )
        if extra:
            refinement.update(deepcopy(dict(extra)))
        self.refinement = refinement

    def set_linear_solver(
        self,
        *,
        type: Literal[  # noqa: A002
            "AMS",
            "BoomerAMG",
            "SuperLU",
            "MUMPS",
            "STRUMPACK",
            "Jacobi",
            "Default",
        ]
        | None = None,
        ksp_type: str | None = None,
        tol: float | None = None,
        max_its: int | None = None,
        mg_max_levels: int | None = None,
        mg_cycle_its: int | None = None,
        mg_smooth_its: int | None = None,
        div_free_tol: float | None = None,
        div_free_max_its: int | None = None,
        estimator_tol: float | None = None,
        estimator_max_its: int | None = None,
        estimator_mg: bool | None = None,
        ams_max_its: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure Palace ``Solver.Linear`` through a JSON fragment."""
        numerical = (
            self.numerical.model_copy(
                update={
                    "solver_type": type,
                    "preconditioner": "Default",
                    "linear_solver": None,
                }
            )
            if type is not None
            else self.numerical
        )
        linear = numerical.to_linear_solver_config(palace_version=self.palace_version)
        updates = {
            "Type": type,
            "KSPType": ksp_type,
            "Tol": tol,
            "MaxIts": max_its,
            "MGMaxLevels": mg_max_levels,
            "MGCycleIts": mg_cycle_its,
            "MGSmoothIts": mg_smooth_its,
            "DivFreeTol": div_free_tol,
            "DivFreeMaxIts": div_free_max_its,
            "EstimatorTol": estimator_tol,
            "EstimatorMaxIts": estimator_max_its,
            "EstimatorMG": estimator_mg,
            "AMSMaxIts": ams_max_its,
        }
        linear.update(
            {key: value for key, value in updates.items() if value is not None}
        )
        if extra:
            linear.update(deepcopy(dict(extra)))
        self.numerical.linear_solver = linear

    def set_output_formats(
        self,
        *,
        paraview: bool = True,
        grid_function: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure Palace ``Problem.OutputFormats``."""
        output_formats: dict[str, Any] = {
            "Paraview": paraview,
            "GridFunction": grid_function,
        }
        if extra:
            output_formats.update(deepcopy(dict(extra)))
        self.output_formats = output_formats

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_stack(self) -> LayerStack:
        """Resolve the layer stack from PDK, YAML, or custom object.

        Returns:
            LayerStack object for mesh generation
        """
        # If a custom stack was given via set_stack(layer_stack), use it
        if self.stack is not None and self._stack_kwargs.get("_prebuilt"):
            # Apply material overrides
            for name, props in self.materials.items():
                self.stack.materials[name] = props.to_dict()
            return self.stack

        from gsim.common.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        legacy_stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)

        # Restore yaml_path for potential re-resolution
        self._stack_kwargs["yaml_path"] = yaml_path

        # Apply material overrides
        for name, props in self.materials.items():
            legacy_stack.materials[name] = props.to_dict()

        # Keep stack free of synthetic air regions; Palace airboxes are
        # generated explicitly from set_airbox().
        legacy_stack.dielectrics = [
            d
            for d in legacy_stack.dielectrics
            if str(d.get("material", "")).lower() != "air"
        ]

        # Store the LayerStack
        self.stack = legacy_stack

        return legacy_stack

    def _build_mesh_config(
        self,
        preset: Literal["coarse", "default", "fine"] | None,
        refined_mesh_size: float | None,
        max_mesh_size: float | None,
        margin: float | None,
        airbox_margin: float | None,
        fmax: float | None,
        planar_conductors: bool | None,
        show_gui: bool,
        margin_x: float | None = None,
        margin_y: float | None = None,
        auto_size: bool = False,
        cells_per_feature: int = 2,
        curve_fit_mode: Literal["line", "spline", "bspline"] | None = None,
        curve_fit_layers: list[str] | None = None,
        curve_fit_tolerance_um: float | None = None,
        curve_fit_min_points: int | None = None,
        curve_fit_corner_angle_deg: float | None = None,
        high_order_elements: bool | None = None,
        high_order_order: int | None = None,
        high_order_optimize: bool | None = None,
        surface_epr_inset_margins_um: Sequence[float] | None = None,
    ) -> MeshConfig:
        """Build mesh config from preset with optional overrides.

        When ``auto_size`` is True and ``refined_mesh_size`` is not explicitly
        provided, ``refined_mesh_size`` is scaled down to
        ``min(preset_size, min_conductor_feature / cells_per_feature)``.

        When ``auto_size`` is False (the default), presets use their literal
        ``refined_mesh_size``. If a small conductor feature is detected that
        may be under-resolved by the preset, a warning is emitted suggesting
        ``auto_size=True``.
        """
        if preset is None or preset == "default":
            mesh_config = MeshConfig.default()
        elif preset == "coarse":
            mesh_config = MeshConfig.coarse()
        elif preset == "fine":
            mesh_config = MeshConfig.fine()
        else:
            raise ValueError(
                f"Unknown preset {preset!r}; "
                "expected one of 'coarse', 'default', 'fine', or None"
            )

        # Auto-size is explicit opt-in. Presets are "honest" speed tiers by
        # default; when auto_size=True the refined_mesh_size shrinks to
        # min(preset, min_conductor_feature / cells_per_feature). When
        # auto_size=False we just warn if a small feature risks being
        # under-resolved.
        if auto_size:
            if refined_mesh_size is None:
                component = self.geometry.component if self.geometry else None
                if component is not None:
                    from gsim.palace.mesh.auto_size import auto_refined_mesh_size

                    stack = self._resolve_stack()
                    scaled = auto_refined_mesh_size(
                        component,
                        stack,
                        preset_size=mesh_config.refined_mesh_size,
                        cells_per_feature=cells_per_feature,
                    )
                    if scaled < mesh_config.refined_mesh_size:
                        logger.info(
                            "Auto-sizing refined_mesh_size: %.3f -> %.3f um "
                            "(min conductor feature / %d)",
                            mesh_config.refined_mesh_size,
                            scaled,
                            cells_per_feature,
                        )
                        mesh_config.refined_mesh_size = scaled
        else:
            component = self.geometry.component if self.geometry else None
            if component is not None:
                from gsim.palace.mesh.auto_size import min_conductor_feature_size

                stack = self._resolve_stack()
                feature = min_conductor_feature_size(component, stack)
                if (
                    feature is not None
                    and math.isfinite(feature)
                    and feature < mesh_config.refined_mesh_size / 2
                ):
                    logger.warning(
                        "Small conductor feature detected (%.3f um) may be "
                        "under-resolved by refined_mesh_size=%.3f um. "
                        "Pass auto_size=True to scale the mesh down.",
                        feature,
                        mesh_config.refined_mesh_size,
                    )

        existing_config = getattr(self, "mesh_config", None)

        # Preserve curve-fit settings from sim.mesh_config unless overridden.
        if existing_config is not None:
            mesh_config.curve_fit_mode = existing_config.curve_fit_mode
            mesh_config.curve_fit_layers = list(existing_config.curve_fit_layers)
            mesh_config.curve_fit_tolerance_um = existing_config.curve_fit_tolerance_um
            mesh_config.curve_fit_min_points = existing_config.curve_fit_min_points
            mesh_config.curve_fit_corner_angle_deg = (
                existing_config.curve_fit_corner_angle_deg
            )
            mesh_config.high_order_elements = existing_config.high_order_elements
            mesh_config.high_order_order = existing_config.high_order_order
            mesh_config.high_order_optimize = existing_config.high_order_optimize
            mesh_config.surface_epr_inset_margins_um = (
                existing_config.surface_epr_inset_margins_um
            )

        # Preserve planar_conductors from sim.mesh_config if not
        # explicitly provided via sim.mesh(planar_conductors=...)
        if planar_conductors is None:
            if existing_config is not None:
                mesh_config.planar_conductors = existing_config.planar_conductors
        else:
            mesh_config.planar_conductors = planar_conductors

        # Apply overrides
        if refined_mesh_size is not None:
            mesh_config.refined_mesh_size = refined_mesh_size
        if max_mesh_size is not None:
            mesh_config.max_mesh_size = max_mesh_size
        if margin is not None:
            mesh_config.margin = margin
        if margin_x is not None:
            mesh_config.margin_x = margin_x
        if margin_y is not None:
            mesh_config.margin_y = margin_y
        if airbox_margin is not None:
            raise ValueError(
                "airbox_margin has been removed from mesh()/preview(). "
                "Use set_airbox(margin_x=..., margin_y=..., "
                "z_above=..., z_below=...)."
            )
        if fmax is not None:
            mesh_config.fmax = fmax
        if curve_fit_mode is not None:
            mesh_config.curve_fit_mode = curve_fit_mode
        if curve_fit_layers is not None:
            mesh_config.curve_fit_layers = curve_fit_layers
        if curve_fit_tolerance_um is not None:
            mesh_config.curve_fit_tolerance_um = curve_fit_tolerance_um
        if curve_fit_min_points is not None:
            mesh_config.curve_fit_min_points = curve_fit_min_points
        if curve_fit_corner_angle_deg is not None:
            mesh_config.curve_fit_corner_angle_deg = curve_fit_corner_angle_deg
        if high_order_elements is not None:
            mesh_config.high_order_elements = high_order_elements
        if high_order_order is not None:
            mesh_config.high_order_order = high_order_order
        if high_order_optimize is not None:
            mesh_config.high_order_optimize = high_order_optimize
        if surface_epr_inset_margins_um is not None:
            mesh_config.surface_epr_inset_margins_um = tuple(
                surface_epr_inset_margins_um
            )
        mesh_config.show_gui = show_gui

        return mesh_config

    # -------------------------------------------------------------------------
    # Post-mesh validation
    # -------------------------------------------------------------------------

    def validate_mesh(
        self,
        *,
        material_overlay: Any | None = None,
    ) -> ValidationResult:
        """Validate the generated mesh and config before cloud submission.

        Checks that physical groups are correctly assigned after meshing:
        conductor surfaces, dielectric volumes, ports, and absorbing boundary.
        Also verifies the generated config.json structure.

        Call after mesh() and before run().

        Returns:
            ValidationResult with validation status and messages when valid.

        Raises:
            RuntimeError: If mesh/config validation fails.

        Example:
            >>> sim.mesh(preset="coarse")
            >>> result = sim.validate_mesh()
            >>> print(result)
        """
        from gsim.palace.mesh.validation import validate_mesh as _validate_mesh

        result = _validate_mesh(self, material_overlay=material_overlay)
        if not result.valid:
            raise RuntimeError(f"Mesh validation failed:\n{result}")
        return result

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        """Print the layer stack table.

        Example:
            >>> sim.show_stack()
        """
        from gsim.common.stack import print_stack_table

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            print_stack_table(self.stack)

    def plot_stack(self) -> None:
        """Plot the layer stack visualization.

        Example:
            >>> sim.plot_stack()
        """
        from gsim.common.stack import plot_stack

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            plot_stack(self.stack)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_mesh(
        self,
        output: str | Path | None = None,
        show_groups: list[str] | None = None,
        interactive: bool = True,
        style: Literal["wireframe", "solid"] = "wireframe",
        transparent_groups: list[str] | None = None,
    ) -> None:
        """Plot the mesh using PyVista.

        Requires mesh() to be called first.

        Args:
            output: Output PNG path (only used if interactive=False)
            show_groups: List of group name patterns to show (None = all).
                Example: ["metal", "P"] to show metal layers and ports.
            interactive: If True, open interactive 3D viewer.
                If False, save static PNG to output path.
            style: ``"wireframe"`` (edges only) or ``"solid"`` (coloured
                surfaces per physical group).
            transparent_groups: Group names rendered at low opacity in
                *solid* mode.  Ignored in *wireframe* mode.

        Raises:
            ValueError: If output_dir not set or mesh file doesn't exist

        Example:
            >>> sim.mesh(preset="default")
            >>> sim.plot_mesh(show_groups=["metal", "P"])
            >>> sim.plot_mesh(style="solid", transparent_groups=["Absorbing_boundary"])
        """
        from gsim.viz import plot_mesh as _plot_mesh

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_path = self._output_dir / "palace.msh"
        if not mesh_path.exists():
            raise ValueError(f"Mesh file not found: {mesh_path}. Call mesh() first.")

        # Default output path if not interactive
        if output is None and not interactive:
            output = self._output_dir / "mesh.png"

        _plot_mesh(
            msh_path=mesh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
            style=style,
            transparent_groups=transparent_groups,
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings_list = []

        # Check geometry
        if self.geometry is None:
            errors.append("No component set. Call set_geometry(component) first.")

        # Check stack
        if self.stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        # Check ports

        has_ports = bool(self.ports) or bool(self.cpw_ports) or bool(self.wave_ports)
        if not has_ports:
            if self.simulation_type == "driven":
                warnings_list.append(
                    "No ports configured. Call add_port(), add_cpw_port(),"
                    " or add_wave_port()."
                )
            elif self.simulation_type == "eigenmode":
                warnings_list.append(
                    "No ports configured. Eigenmode findsallmodes without port loading."
                )
        else:
            # Validate port configurations
            for port in self.ports:
                if port.geometry == "inplane" and port.layer is None:
                    errors.append(f"Port '{port.name}': inplane ports require 'layer'")
                if port.geometry == "via" and (
                    port.from_layer is None or port.to_layer is None
                ):
                    errors.append(
                        f"Port '{port.name}': via ports require "
                        "'from_layer' and 'to_layer'"
                    )

            # Validate CPW ports
            errors.extend(
                f"CPW port '{cpw.name}': 'layer' is required"
                for cpw in self.cpw_ports
                if not cpw.layer
            )
            # Validate wave ports
            errors.extend(
                f"Wave port '{wp.name}': 'layer' is required"
                for wp in self.wave_ports
                if not wp.layer
            )

        # Validate excitation port if specified
        if self.simulation_type == "driven" and self.driven.excitation_port is not None:
            port_names = [p.name for p in self.ports]
            cpw_names = [cpw.name for cpw in self.cpw_ports]
            all_port_names = port_names + cpw_names
            if self.driven.excitation_port not in all_port_names:
                errors.append(
                    f"Excitation port '{self.driven.excitation_port}' not found. "
                    f"Available: {all_port_names}"
                )

        if self.simulation_type == "electrostatic" and len(self.terminals) < 2:
            # Electrostatic requires at least 2 terminals
            errors.append(
                "Electrostatic simulation requires at least 2 terminals. "
                "Call add_terminal() to add terminals."
            )
        if self.simulation_type == "electrostatic":
            # Validate terminal configurations
            errors.extend(
                f"Terminal '{terminal.name}': 'layer' is required"
                for terminal in self.terminals
                if not terminal.layer
            )

        if self.simulation_type == "magnetostatic":
            current_sources = getattr(self, "current_sources", []) or []
            if not current_sources:
                errors.append(
                    "Magnetostatic simulation requires at least 1 current source. "
                    "Call add_current_source() to add a source."
                )
            for source in current_sources:
                if source.elements:
                    errors.extend(
                        f"Current source '{source.name}' element {element_index}: "
                        "'layer' is required"
                        for element_index, element in enumerate(
                            source.elements, start=1
                        )
                        if not element.layer
                    )
                elif not source.layer:
                    errors.append(
                        f"Current source '{source.name}': 'layer' is required"
                    )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _find_gf_port(self, port_name: str):
        """Find a gdsfactory port by name."""
        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        for p in component.ports:
            if p.name == port_name:
                return p

        raise ValueError(
            f"Port '{port_name}' not found on component. "
            f"Available ports: {[p.name for p in component.ports]}"
        )

    def _configure_ports_on_component(self, stack: LayerStack) -> None:  # noqa: ARG002
        """Configure ports on the component using legacy functions."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
            configure_wave_port,
        )

        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        # Configure regular ports
        for port_config in self.ports or []:
            if self.simulation_type != "driven":
                port_config.excited = False
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = self._find_gf_port(port_config.name)

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available ports: {[p.name for p in component.ports]}"
                )

            if port_config.geometry == "inplane" and port_config.layer is not None:
                configure_inplane_port(
                    gf_port,
                    layer=port_config.layer,
                    length=port_config.length or gf_port.width,
                    direction=port_config.direction,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                    offset=port_config.offset,
                    generate_sheet=port_config.generate_sheet,
                )
            elif port_config.geometry == "via" and (
                port_config.from_layer is not None and port_config.to_layer is not None
            ):
                configure_via_port(
                    gf_port,
                    from_layer=port_config.from_layer,
                    to_layer=port_config.to_layer,
                    direction=port_config.direction,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                    offset=port_config.offset,
                )

            # Clear any stale reactive metadata from previous simulations.
            # kfactory Info is mapping-like (get/setitem) but has no pop/delitem.
            gf_port.info["resistance"] = None
            gf_port.info["inductance"] = None
            gf_port.info["capacitance"] = None

            # Attach RLC values to port info for downstream consumers.
            if port_config.resistance is not None:
                gf_port.info["resistance"] = port_config.resistance
            if port_config.inductance is not None:
                gf_port.info["inductance"] = port_config.inductance
            if port_config.capacitance is not None:
                gf_port.info["capacitance"] = port_config.capacitance

        # Configure CPW ports
        for cpw_config in self.cpw_ports or []:
            # Find the single gdsfactory port at the signal center
            gf_port = self._find_gf_port(cpw_config.name)

            if gf_port is None:
                raise ValueError(
                    f"CPW port '{cpw_config.name}' not found on component. "
                    f"Available: {[p.name for p in component.ports]}"
                )

            configure_cpw_port(
                gf_port,
                layer=cpw_config.layer,
                s_width=cpw_config.s_width,
                gap_width=cpw_config.gap_width,
                length=cpw_config.length,
                impedance=cpw_config.impedance,
                excited=cpw_config.excited,
                offset=cpw_config.offset,
                generate_sheet=cpw_config.generate_sheet,
            )
        # Configure wave ports
        for port_config in self.wave_ports:
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = self._find_gf_port(port_config.name)

            if port_config.layer is not None:
                configure_wave_port(
                    gf_port,
                    layer=port_config.layer,
                    z_margin=port_config.z_margin,
                    lateral_margin=port_config.lateral_margin,
                    max_size=port_config.max_size,
                    excited=port_config.excited,
                    mode=port_config.mode,
                    offset=port_config.offset,
                )

        self._configured_ports = True

    def _generate_mesh_internal(
        self,
        output_dir: Path,
        mesh_config: MeshConfig,
        ports: list,
        driven_config: Any,
        model_name: str,
        verbose: bool,
        write_config: bool = True,
        periodic_axis: str | None = None,
        decimate_tolerance: float | None = None,
        gmsh_verbosity: int = 0,
    ) -> MeshResult:
        """Internal mesh generation."""
        from gsim.palace.mesh.generator import generate_mesh

        component = self.geometry.component if self.geometry else None

        # Get effective fmax from driven config if mesh doesn't specify
        effective_fmax = mesh_config.fmax
        if driven_config is not None and mesh_config.fmax == 100e9:
            effective_fmax = driven_config.fmax

        # Resolve stack
        stack = self._resolve_stack()
        self._reject_explicit_region_airbox_config(mesh_config)
        airbox_cfg = self._airbox_config or {}
        domain_margin_x = airbox_cfg.get("margin_x", mesh_config.effective_margin_x)
        domain_margin_y = airbox_cfg.get("margin_y", mesh_config.effective_margin_y)

        if verbose:
            logger.info("Generating mesh in %s", output_dir)

        airbox_cfg = getattr(self, "_airbox_config", {})
        domain_margin_x = airbox_cfg.get("margin_x", mesh_config.effective_margin_x)
        domain_margin_y = airbox_cfg.get("margin_y", mesh_config.effective_margin_y)

        mesh_result = generate_mesh(
            component=component,
            stack=stack,
            ports=ports,
            output_dir=output_dir,
            model_name=model_name,
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            margin_x=domain_margin_x,
            margin_y=domain_margin_y,
            air_margin=mesh_config.airbox_margin,
            airbox_margin_x=airbox_cfg.get("margin_x"),
            airbox_margin_y=airbox_cfg.get("margin_y"),
            airbox_z_above=airbox_cfg.get("z_above"),
            airbox_z_below=airbox_cfg.get("z_below"),
            fmax=effective_fmax,
            show_gui=mesh_config.show_gui,
            simulation_type=self.simulation_type,
                driven_config=driven_config,
                eigenmode_config=self.eigenmode,
                magnetostatic_config=getattr(self, "magnetostatic", None),
                numerical_config=self.numerical,
                refinement_config=self.refinement,
                problem_output_formats=self.output_formats or None,
                write_config=write_config,
            planar_conductors=mesh_config.planar_conductors,
            pec_blocks=self._pec_blocks or None,
            absorbing_boundary=self.absorbing_boundary,
            current_sources=getattr(self, "current_sources", None) or [],
            periodic_axis=periodic_axis,
            merge_via_distance=mesh_config.merge_via_distance,
            curve_fit_mode=mesh_config.curve_fit_mode,
            curve_fit_layers=mesh_config.curve_fit_layers,
            curve_fit_tolerance_um=mesh_config.curve_fit_tolerance_um,
            curve_fit_min_points=mesh_config.curve_fit_min_points,
            curve_fit_corner_angle_deg=mesh_config.curve_fit_corner_angle_deg,
            high_order_elements=mesh_config.high_order_elements,
            high_order_order=mesh_config.high_order_order,
            high_order_optimize=mesh_config.high_order_optimize,
            verbosity=gmsh_verbosity,
            decimate_tolerance=decimate_tolerance,
            surface_epr_inset_margins_um=mesh_config.surface_epr_inset_margins_um,
            simulation_layers=self._simulation_layers,
            activated_regions=self._activated_region_values(),
        )

        # Store mesh_result for deferred config generation
        self._last_mesh_result = mesh_result
        self._last_ports = ports

        return mesh_result

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        """Get ports for preview."""
        from gsim.palace.ports import extract_ports

        component = self.geometry.component if self.geometry else None
        self._configure_ports_on_component(stack)
        return extract_ports(component, stack)

    # -------------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------------

    def preview(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
        airbox_margin: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = True,
        auto_size: bool = False,
        cells_per_feature: int = 2,
        curve_fit_mode: Literal["line", "spline", "bspline"] | None = None,
        curve_fit_layers: list[str] | None = None,
        curve_fit_tolerance_um: float | None = None,
        curve_fit_min_points: int | None = None,
        curve_fit_corner_angle_deg: float | None = None,
        high_order_elements: bool | None = None,
        high_order_order: int | None = None,
        high_order_optimize: bool | None = None,
        decimate_tolerance: float | None = None,
        surface_epr_inset_margins_um: Sequence[float] | None = None,
    ) -> None:
        """Preview the mesh without running simulation.

        Opens the gmsh GUI to visualize the mesh interactively.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um)
            max_mesh_size: Max mesh size in air/dielectric (um)
            margin: XY margin around design (um)
            margin_x: X-axis margin (um). Overrides margin for X.
            margin_y: Y-axis margin (um). Overrides margin for Y.
            z_above: Airbox extension above stack top (um).
            z_below: Airbox extension below stack bottom (um).
            airbox_margin: Deprecated. Use set_airbox().
            fmax: Max frequency for mesh sizing (Hz)
            planar_conductors: Treat conductors as 2D PEC surfaces
            show_gui: Show gmsh GUI for interactive preview
            auto_size: If True, scale refined_mesh_size down to the smallest
                conductor feature / cells_per_feature. Off by default.
            cells_per_feature: Target cells across the smallest conductor
                feature when auto_size=True. Default 2.
            curve_fit_mode: Patterned dielectric boundary mode (line/spline/bspline).
            curve_fit_layers: Layer names where spline fitting is applied.
            curve_fit_tolerance_um: Point merge tolerance before fitting.
            curve_fit_min_points: Min contour points required to fit curves.
            curve_fit_corner_angle_deg: Turn-angle threshold used to split
                sharp corners from smooth curve-fit segments.
            high_order_elements: Enable high-order geometric mesh elements.
            high_order_order: Polynomial order for high-order elements.
            high_order_optimize: Run gmsh high-order optimization after meshing.
            decimate_tolerance: Relative tolerance for polygon decimation
                (None = no decimation; typical 0.001-0.01).
            surface_epr_inset_margins_um: Surface EPR inset margins in um.
                0 means total; positive values define generated finite-shell
                inset partitions.

        Example:
            >>> sim.preview(preset="fine", planar_conductors=True, show_gui=True)
        """
        from gsim.palace.mesh.generator import generate_mesh

        component = self.geometry.component if self.geometry else None

        # Validate configuration
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        self._apply_airbox_overrides(
            margin_x=margin_x,
            margin_y=margin_y,
            z_above=z_above,
            z_below=z_below,
        )

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            margin_x=margin_x,
            margin_y=margin_y,
            airbox_margin=airbox_margin,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
            auto_size=auto_size,
            cells_per_feature=cells_per_feature,
            curve_fit_mode=curve_fit_mode,
            curve_fit_layers=curve_fit_layers,
            curve_fit_tolerance_um=curve_fit_tolerance_um,
            curve_fit_min_points=curve_fit_min_points,
            curve_fit_corner_angle_deg=curve_fit_corner_angle_deg,
            high_order_elements=high_order_elements,
            high_order_order=high_order_order,
            high_order_optimize=high_order_optimize,
            surface_epr_inset_margins_um=surface_epr_inset_margins_um,
        )

        # Resolve stack
        stack = self._resolve_stack()
        self._reject_explicit_region_airbox_config(mesh_config)

        # Get ports
        ports = self._get_ports_for_preview(stack)

        airbox_cfg = self._airbox_config or {}
        domain_margin_x = airbox_cfg.get("margin_x", mesh_config.effective_margin_x)
        domain_margin_y = airbox_cfg.get("margin_y", mesh_config.effective_margin_y)

        # Generate mesh in temp directory
        airbox_cfg = getattr(self, "_airbox_config", {})
        domain_margin_x = airbox_cfg.get("margin_x", mesh_config.effective_margin_x)
        domain_margin_y = airbox_cfg.get("margin_y", mesh_config.effective_margin_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=ports,
                output_dir=tmpdir,
                refined_mesh_size=mesh_config.refined_mesh_size,
                max_mesh_size=mesh_config.max_mesh_size,
                margin_x=domain_margin_x,
                margin_y=domain_margin_y,
                air_margin=mesh_config.airbox_margin,
                airbox_margin_x=airbox_cfg.get("margin_x"),
                airbox_margin_y=airbox_cfg.get("margin_y"),
                airbox_z_above=airbox_cfg.get("z_above"),
                airbox_z_below=airbox_cfg.get("z_below"),
                fmax=mesh_config.fmax,
                show_gui=True,
                simulation_type=self.simulation_type,
                driven_config=self.driven,
                eigenmode_config=self.eigenmode,
                electrostatic_config=getattr(self, "electrostatic", None),
                magnetostatic_config=getattr(self, "magnetostatic", None),
                numerical_config=self.numerical,
                refinement_config=self.refinement,
                problem_output_formats=self.output_formats or None,
                terminals=getattr(self, "terminals", None) or [],
                current_sources=getattr(self, "current_sources", None) or [],
                planar_conductors=mesh_config.planar_conductors,
                pec_blocks=self._pec_blocks or None,
                absorbing_boundary=self.absorbing_boundary,
                merge_via_distance=mesh_config.merge_via_distance,
                curve_fit_mode=mesh_config.curve_fit_mode,
                curve_fit_layers=mesh_config.curve_fit_layers,
                curve_fit_tolerance_um=mesh_config.curve_fit_tolerance_um,
                curve_fit_min_points=mesh_config.curve_fit_min_points,
                curve_fit_corner_angle_deg=mesh_config.curve_fit_corner_angle_deg,
                high_order_elements=mesh_config.high_order_elements,
                high_order_order=mesh_config.high_order_order,
                high_order_optimize=mesh_config.high_order_optimize,
                decimate_tolerance=decimate_tolerance,
                surface_epr_inset_margins_um=mesh_config.surface_epr_inset_margins_um,
                simulation_layers=self._simulation_layers,
                activated_regions=self._activated_region_values(),
            )

    # -------------------------------------------------------------------------
    # Mesh generation
    # -------------------------------------------------------------------------

    def mesh(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        margin_x: float | None = None,
        margin_y: float | None = None,
        z_above: float | None = None,
        z_below: float | None = None,
        airbox_margin: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = False,
        model_name: str = "palace",
        verbose: bool = True,
        verbosity: int = 0,
        auto_size: bool = False,
        cells_per_feature: int = 2,
        periodic_axis: str | None = None,
        decimate_tolerance: float | None = None,
        merge_via_distance: float | None = None,
        curve_fit_mode: Literal["line", "spline", "bspline"] | None = None,
        curve_fit_layers: list[str] | None = None,
        curve_fit_tolerance_um: float | None = None,
        curve_fit_min_points: int | None = None,
        curve_fit_corner_angle_deg: float | None = None,
        high_order_elements: bool | None = None,
        high_order_order: int | None = None,
        high_order_optimize: bool | None = None,
        surface_epr_inset_margins_um: Sequence[float] | None = None,
    ) -> MeshResult:
        """Generate the mesh for Palace simulation.

        Only generates the mesh file (palace.msh). Config is generated
        separately with write_config().

        Requires set_output_dir() to be called first.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um), overrides preset
            max_mesh_size: Max mesh size in air/dielectric (um), overrides preset
            margin: XY margin around design (um), overrides preset
            margin_x: X-axis margin (um). Overrides margin for X.
            margin_y: Y-axis margin (um). Overrides margin for Y.
            z_above: Airbox extension above stack top (um).
            z_below: Airbox extension below stack bottom (um).
            airbox_margin: Deprecated. Use set_airbox().
            fmax: Max frequency for mesh sizing (Hz), overrides preset
            planar_conductors: Treat conductors as 2D PEC surfaces
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files
            verbose: Print progress messages
            verbosity: Gmsh OCC verbosity level (0=silent, 1-99=increasing)
            auto_size: If True, scale refined_mesh_size down to the smallest
                conductor feature / cells_per_feature. Off by default so presets
                use their literal refined_mesh_size.
            cells_per_feature: Target cells across the smallest conductor
                feature when auto_size=True. Default 2.
            periodic_axis: Optional periodic axis ("x" or "y") for periodic
                meshing constraints on opposite domain sides.
            decimate_tolerance: Relative tolerance for polygon decimation
                (None = no decimation; typical 0.001-0.01). Reduces vertex
                count on curved geometry before meshing.
            merge_via_distance: Max gap (um) between nearby via polygons to
                merge before extrusion. Pass 0 to disable merging (keep
                each via as a separate volume). Defaults to the preset's
                value (typically 2.0 um).
            curve_fit_mode: Patterned dielectric boundary mode (line/spline/bspline).
            curve_fit_layers: Layer names where spline fitting is applied.
            curve_fit_tolerance_um: Point merge tolerance before fitting.
            curve_fit_min_points: Min contour points required to fit curves.
            curve_fit_corner_angle_deg: Turn-angle threshold used to split
                sharp corners from smooth curve-fit segments.
            high_order_elements: Enable high-order geometric mesh elements.
            high_order_order: Polynomial order for high-order elements.
            high_order_optimize: Run gmsh high-order optimization after meshing.
            surface_epr_inset_margins_um: Surface EPR inset margins in um.
                0 means total; positive values define generated finite-shell
                inset partitions.

        Returns:
            MeshResult with mesh path and generated manifest

        Raises:
            ValueError: If output_dir not set or configuration is invalid

        Example:
            >>> sim.set_output_dir("./sim")
            >>> result = sim.mesh(preset="fine", planar_conductors=True)
            >>> print(f"Mesh saved to: {result.mesh_path}")
        """
        from gsim.palace.ports import extract_ports

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        component = self.geometry.component if self.geometry else None

        self._apply_airbox_overrides(
            margin_x=margin_x,
            margin_y=margin_y,
            z_above=z_above,
            z_below=z_below,
        )

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            margin_x=margin_x,
            margin_y=margin_y,
            airbox_margin=airbox_margin,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
            auto_size=auto_size,
            cells_per_feature=cells_per_feature,
            curve_fit_mode=curve_fit_mode,
            curve_fit_layers=curve_fit_layers,
            curve_fit_tolerance_um=curve_fit_tolerance_um,
            curve_fit_min_points=curve_fit_min_points,
            curve_fit_corner_angle_deg=curve_fit_corner_angle_deg,
            high_order_elements=high_order_elements,
            high_order_order=high_order_order,
            high_order_optimize=high_order_optimize,
            surface_epr_inset_margins_um=surface_epr_inset_margins_um,
        )

        if merge_via_distance is not None:
            mesh_config.merge_via_distance = merge_via_distance

        # Validate configuration
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        output_dir = self._output_dir

        # Resolve stack and configure ports
        stack = self._resolve_stack()
        self._configure_ports_on_component(stack)

        # Extract ports
        palace_ports = extract_ports(component, stack)

        # Generate mesh (config is written separately by simulate() or write_config())
        result = self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
            driven_config=self.driven,
            model_name=model_name,
            verbose=verbose,
            write_config=False,
            periodic_axis=periodic_axis,
            decimate_tolerance=decimate_tolerance,
            gmsh_verbosity=verbosity,
        )

        # Post-mesh summary: nodes, tets, refined / max sizes (in um).
        stats = result.mesh_stats or {}
        node_count = stats.get("nodes")
        tet_count = stats.get("tetrahedra")
        if node_count is not None and tet_count is not None:
            logger.info(
                "Mesh: %s nodes \u00b7 %s tets \u00b7 refined=%.3g \u00b5m \u00b7 "
                "max=%.3g \u00b5m",
                f"{node_count:,}",
                f"{tet_count:,}",
                mesh_config.refined_mesh_size,
                mesh_config.max_mesh_size,
            )
            if node_count > 75_000:
                logger.warning(
                    "Mesh has %s nodes (>75,000) \u2014 simulation may be slow.",
                    f"{node_count:,}",
                )

        return result

    def write_config(
        self,
        *,
        validate_mesh: bool = True,
        photonic: bool = False,
        photononic: bool | None = None,
        postprocessing: PostprocessingConfig | None = None,
        reuse_postprocessing: bool = True,
        write_artifacts: bool = True,
        material_overlay: Any | None = None,
        hints: dict[str, Any] | None = None,
        prepare_run_folder: bool = True,
        validate_schema: bool = True,
    ) -> Path:
        """Write Palace config.json after mesh generation.

        By default this validates that the generated mesh/config include
        conductor boundaries required for EM conductor solves. For photonic
        simulations, pass ``photonic=True`` to skip this validation.

        Args:
            validate_mesh: Validate the generated mesh/config after writing.
            photonic: Skip conductor-oriented mesh validation when ``True``.
            photononic: Deprecated alias for ``photonic``.
            postprocessing: Optional typed Palace postprocessing config built
                from the mesh manifest. When provided, its domain and boundary
                fragments are merged into ``config.json`` and its index map is
                written beside the config.
            reuse_postprocessing: Reuse the last provided postprocessing config
                when ``postprocessing`` is omitted. This keeps upload/run paths
                from silently dropping a previously configured index map.
            write_artifacts: Write ``metadata/mesh_manifest.json`` and, when
                postprocessing is active, ``metadata/palace_index_map.json``.
            material_overlay: Optional PDK material overlay path, raw overlay
                mapping, or loaded overlay mapping used to resolve Palace
                material values without mutating the source layer stack.
            hints: Optional Palace config fragments merged into ``config.json``.
            prepare_run_folder: Create the canonical Palace run-folder skeleton
                before writing config and sidecars.
            validate_schema: Validate the final assembled Palace config against
                the target ``palace_version`` schema before returning.

        Returns:
            Path to the generated config.json

        Raises:
            ValueError: If mesh() hasn't been called yet

        Example:
            >>> result = sim.mesh("./sim")
            >>> config_path = sim.write_config()
            >>> # Photonic workflow without conductor boundaries
            >>> config_path = sim.write_config(photonic=True)
        """
        from gsim.palace.mesh.generator import write_config as gen_write_config
        from gsim.palace.mesh.postprocessing import (
            PostprocessingIndexMap,
            build_surface_current_index_map_from_manifest,
            build_terminal_index_map_from_manifest,
        )

        if photononic is not None:
            logger.warning(
                "write_config(photononic=...) is deprecated; use photonic=..."
            )
            photonic = photonic or photononic

        if self._last_mesh_result is None:
            raise ValueError("No mesh result. Call mesh() first.")

        if not self._last_mesh_result.groups:
            raise ValueError(
                "Mesh result has no groups data. "
                "Was mesh() called with write_config=True already?"
            )

        if postprocessing is not None:
            self._last_postprocessing_config = postprocessing
        elif reuse_postprocessing:
            postprocessing = getattr(self, "_last_postprocessing_config", None)

        domain_postprocessing_config = None
        boundary_postprocessing_config = None
        if postprocessing is not None:
            postprocessing_fragments = postprocessing.to_config()
            domain_postprocessing_config = postprocessing_fragments["domains"]
            boundary_postprocessing_config = postprocessing_fragments["boundaries"]

        stack = self._resolve_stack()
        electrostatic_config = getattr(self, "electrostatic", None)
        magnetostatic_config = getattr(self, "magnetostatic", None)
        terminals = getattr(self, "terminals", None)
        current_sources = getattr(self, "current_sources", None)
        config_hints = dict(self._hints)
        if hints:
            config_hints.update(hints)
        previous_hints = self._hints
        self._hints = config_hints
        try:
            config_path = gen_write_config(
                mesh_result=self._last_mesh_result,
                stack=stack,
                ports=self._last_ports,
                simulation_type=self.simulation_type,
                eigenmode_config=self.eigenmode,
                driven_config=self.driven,
                numerical_config=self.numerical,
                refinement_config=self.refinement,
                problem_output_formats=self.output_formats or None,
                palace_version=self.palace_version,
                validate_schema=validate_schema,
                absorbing_boundary=self.absorbing_boundary,
                hints=self._hints,
                electrostatic_config=electrostatic_config,
                magnetostatic_config=magnetostatic_config,
                terminals=terminals or [],
                current_sources=current_sources or [],
                postprocessing_config=domain_postprocessing_config,
                boundary_postprocessing_config=boundary_postprocessing_config,
                material_overlay=material_overlay,
                prepare_run_folder=prepare_run_folder,
            )
            run_folder = (
                prepare_palace_run_folder(config_path.parent)
                if prepare_run_folder
                else palace_run_folder(config_path.parent)
            )

            if write_artifacts:
                self._last_mesh_result.manifest.write_json(
                    run_folder.mesh_manifest_path
                )
                self._write_geometry_snapshot(run_folder)
                index_map_entries = []
                if postprocessing is not None:
                    index_map_entries.extend(postprocessing.index_map.entries)

                if self.simulation_type == "electrostatic":
                    config = json.loads(config_path.read_text())
                    terminal_entries = config.get("Boundaries", {}).get("Terminal", [])
                    if isinstance(terminal_entries, list):
                        terminal_names = tuple(
                            terminal.name for terminal in terminals or []
                        )
                        terminal_map = build_terminal_index_map_from_manifest(
                            self._last_mesh_result.manifest,
                            terminal_entries,
                            terminal_names=terminal_names,
                        )
                        index_map_entries.extend(terminal_map.entries)
                elif self.simulation_type == "magnetostatic":
                    config = json.loads(config_path.read_text())
                    current_entries = config.get("Boundaries", {}).get(
                        "SurfaceCurrent", []
                    )
                    if isinstance(current_entries, list):
                        current_source_names = tuple(
                            source.name for source in current_sources or []
                        )
                        current_map = build_surface_current_index_map_from_manifest(
                            self._last_mesh_result.manifest,
                            current_entries,
                            current_source_names=current_source_names,
                        )
                        index_map_entries.extend(current_map.entries)

                if index_map_entries:
                    PostprocessingIndexMap(entries=tuple(index_map_entries)).write_json(
                        run_folder.index_map_path
                    )

            # Validate mesh and config unless this is a photonic workflow.
            if not photonic and validate_mesh:
                validation = self.validate_mesh(material_overlay=material_overlay)
                if not validation.valid:
                    raise ValueError(f"Mesh validation failed:\n{validation}")

            return config_path
        finally:
            self._hints = previous_hints

    def _write_geometry_snapshot(self, run_folder: PalaceRunFolder) -> Path | None:
        """Write an optional run-local GDS snapshot for review provenance."""
        component = self.geometry.component if self.geometry is not None else None
        if component is None:
            return None
        write_gds = getattr(component, "write_gds", None)
        if write_gds is None:
            return None
        try:
            run_folder.geometry_dir.mkdir(parents=True, exist_ok=True)
            write_gds(run_folder.design_gds_path)
        except TypeError:
            try:
                write_gds(str(run_folder.design_gds_path))
            except Exception as exc:
                logger.warning("Could not write geometry snapshot: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Could not write geometry snapshot: %s", exc)
            return None
        return run_folder.design_gds_path

    # -------------------------------------------------------------------------
    # Cloud: fine-grained control
    # -------------------------------------------------------------------------

    def _prepare_upload_dir(self, *, prepare_run_folder: bool = True) -> Path:
        """Prepare a temp directory with all config/mesh files for upload.

        Ensures ``_output_dir`` is set, ``config.json`` exists, and copies
        everything to a fresh temp directory.

        Returns:
            Path to temp directory ready for upload.
        """
        import shutil

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        # Always (re)generate config.json to reflect current driven settings
        self.write_config(prepare_run_folder=prepare_run_folder)

        # Copy input files to a temp dir so we don't destroy the user's directory
        tmp = Path(tempfile.mkdtemp(prefix="palace_"))
        for item in self._output_dir.iterdir():
            dest = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return tmp

    def upload(
        self,
        *,
        verbose: bool = True,
        prepare_run_folder: bool = True,
    ) -> str:
        """Prepare config, upload to the cloud. Does NOT start execution.

        Requires :meth:`set_output_dir` and :meth:`mesh` to have been
        called first.

        Args:
            verbose: Print progress messages.
            prepare_run_folder: Create the canonical Palace run-folder skeleton
                before staging files for cloud upload.

        Returns:
            ``job_id`` string for use with :meth:`start`, :meth:`get_status`,
            or :func:`gsim.wait_for_results`.
        """
        from gsim import gcloud

        tmp = self._prepare_upload_dir(prepare_run_folder=prepare_run_folder)
        try:
            self._job_id = gcloud.upload(tmp, "palace", verbose=verbose)
        except Exception:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self._job_id

    def start(self, *, verbose: bool = True) -> None:
        """Start cloud execution for this sim's uploaded job.

        Raises:
            ValueError: If :meth:`upload` has not been called.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("Call upload() first")
        gcloud.start(self._job_id, verbose=verbose)

    def get_status(self) -> str:
        """Get the current status of this sim's cloud job.

        Returns:
            Status string (``"created"``, ``"queued"``, ``"running"``,
            ``"completed"``, ``"failed"``).

        Raises:
            ValueError: If no job has been submitted yet.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.get_status(self._job_id)

    def wait_for_results(
        self,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        parent_dir: str | Path | None = None,
    ) -> Any:
        """Wait for this sim's cloud job, download and parse results.

        Args:
            verbose: Print progress messages.
            parent_dir: Where to create the sim-data directory.

        Returns:
            Parsed result (typically ``dict[str, Path]`` of output files).

        Raises:
            ValueError: If no job has been submitted yet.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.wait_for_results(
            self._job_id, verbose=verbose, parent_dir=parent_dir
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def write_slurm_sbatch_handoff(
        self,
        profile: PalaceSlurmProfileResolution,
        *,
        job_name: str,
        script_path: str | Path = "run_palace.sbatch",
        metadata: Mapping[str, Any] | None = None,
        validate_inputs: bool = True,
        **sbatch_kwargs: Any,
    ) -> PalaceSlurmHandoffResult:
        """Write a Slurm handoff script from a resolved Slurm profile.

        This is the simulation-method wrapper for the lower-level handoff
        renderer. It keeps notebooks on the explicit ``sim`` pipeline while
        leaving Slurm schema validation and rendering in
        ``gsim.palace.handoff``.
        """
        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        from gsim.palace.handoff import write_palace_slurm_sbatch_handoff

        return write_palace_slurm_sbatch_handoff(
            self._output_dir,
            profile.to_sbatch_spec(job_name=job_name, **sbatch_kwargs),
            script_path=script_path,
            profile=profile.profile,
            metadata=metadata,
            validate_inputs=validate_inputs,
        )

    def generate_handoff_package(
        self,
        *,
        include_hashes: bool = False,
        include_results: bool = False,
        write_config: bool = True,
        validate_mesh: bool = True,
        photonic: bool = False,
        postprocessing: PostprocessingConfig | None = None,
        reuse_postprocessing: bool = True,
        write_artifacts: bool = True,
        material_overlay: Any | None = None,
        hints: dict[str, Any] | None = None,
        validate_schema: bool = True,
        status: str = "packaged",
        launcher: Mapping[str, Any] | None = None,
        script_path: str | Path | None = None,
        profile: Any | None = None,
        resources: Mapping[str, Any] | None = None,
        command: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        archive_path: str | Path | None = None,
    ) -> PalaceRunHandle:
        """Generate a Palace handoff package and return a run-stage handle.

        The package includes a fresh ``config.json`` by default, canonical
        run-folder directories, ``metadata/palace_handoff_metadata.json``,
        ``metadata/palace_handoff_archive_manifest.json``, and a tar.gz archive
        rooted at the run-folder name. It does not run Palace or submit a job.
        It also does not load typed reports; callers enter the Resolve stage
        explicitly with ``resolve_palace_result(handle.run_folder, ...)``.

        Args:
            include_hashes: Include SHA-256 checksums for present artifacts.
            include_results: Include solver result files in the archive
                manifest when they already exist.
            write_config: Write ``config.json`` before packaging. This is on by
                default so the handoff package reflects current simulation
                settings.
            validate_mesh: Forwarded to :meth:`write_config`.
            photonic: Forwarded to :meth:`write_config`.
            postprocessing: Optional Palace postprocessing config forwarded to
                :meth:`write_config`.
            reuse_postprocessing: Forwarded to :meth:`write_config`.
            write_artifacts: Forwarded to :meth:`write_config`.
            material_overlay: Forwarded to :meth:`write_config`.
            hints: Forwarded to :meth:`write_config`.
            validate_schema: Forwarded to :meth:`write_config`.
            status: Handoff metadata status string.
            launcher: Optional launcher metadata.
            script_path: Optional batch-script path included in the handoff
                package.
            profile: Optional resolved site/profile metadata.
            resources: Optional resource request metadata.
            command: Optional command metadata. When omitted, a minimal Palace
                config/mesh command shape is recorded.
            metadata: Additional JSON-friendly handoff metadata.
            archive_path: Optional target archive path recorded in metadata.

        Returns:
            :class:`gsim.palace.run_stage.PalaceRunHandle` describing the
            packaged run folder and generated handoff artifacts.

        Raises:
            ValueError: If ``set_output_dir()`` has not been called.
        """
        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        if write_config:
            self.write_config(
                validate_mesh=validate_mesh,
                photonic=photonic,
                postprocessing=postprocessing,
                reuse_postprocessing=reuse_postprocessing,
                write_artifacts=write_artifacts,
                material_overlay=material_overlay,
                hints=hints,
                prepare_run_folder=True,
                validate_schema=validate_schema,
            )
        package_metadata = dict(metadata or {})
        package_metadata["palace_config_version"] = self.palace_version
        return generate_palace_handoff_package(
            self._output_dir,
            simulation_type=self.simulation_type,
            include_hashes=include_hashes,
            include_results=include_results,
            status=status,
            launcher=launcher,
            script_path=script_path,
            profile=profile,
            resources=resources,
            command=command,
            metadata=package_metadata,
            archive_path=archive_path,
        )

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
        prepare_run_folder: bool = True,
    ) -> SParams | dict[str, Path] | str:
        """Run simulation on GDSFactory+ cloud.

        Requires mesh() to be called first. Automatically calls
        write_config() if config.json hasn't been written yet.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            wait: If ``True`` (default), block until results are ready.
                If ``False``, upload + start and return the ``job_id``.
            prepare_run_folder: Create the canonical Palace run-folder skeleton
                before staging files for cloud upload.

        Returns:
            - :class:`SParams` for :class:`DrivenSim` (driven sweeps with
              ``port-S.csv``).
            - ``dict[str, Path]`` of output files for eigenmode /
              electrostatic runs.
            - ``job_id`` string when ``wait=False``.

            Subclasses narrow this annotation to the concrete type they
            return (e.g. ``DrivenSim.run -> SParams | str``).

        Raises:
            ValueError: If output_dir not set or mesh not generated
            RuntimeError: If simulation fails

        Example:
            >>> sp = driven_sim.run()  # returns SParams
            >>> sp.s21.db  # dB magnitude of S21
            >>> results = eigen_sim.run()  # returns dict[str, Path]
            >>> print(results["eig.csv"])
        """
        self.upload(verbose=False, prepare_run_folder=prepare_run_folder)
        self.start(verbose=verbose != "quiet")
        if not wait:
            if self._job_id is None:
                msg = "job_id not set — call upload() first"
                raise RuntimeError(msg)
            return self._job_id
        return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)

    def run_local(
        self,
        *,
        palace_sif_path: str | Path | None = None,
        palace_executable: str | Path | None = None,
        use_apptainer: bool = True,
        executable_mode: Literal["wrapper", "binary"] = "wrapper",
        num_processes: int | None = None,
        num_threads: int | None = None,
        serial: bool = False,
        setup_commands: Sequence[str] | None = None,
        verbose: bool = True,
        prepare_run_folder: bool = True,
        check_runtime_version: bool = True,
    ) -> dict[str, Path]:
        """Run simulation locally using Palace.

        Requires mesh() and write_config() to be called first.
        Supports both Apptainer and direct Palace installation.
        Direct execution can also activate a caller-owned environment first,
        for example a Spack environment supplied by a PDK run profile.

        Args:
            palace_sif_path: Path to Palace Apptainer SIF file.
                Only used when ``use_apptainer=True``.
                If None, uses PALACE_SIF environment variable.
            palace_executable: Path to Palace executable.
                Only used when ``use_apptainer=False``.
                If None, uses PALACE_EXECUTABLE environment variable or "palace".
            use_apptainer: If True (default), run via Apptainer using SIF file.
                If False, run Palace executable directly.
            executable_mode: Direct Palace command style. ``"wrapper"`` expects
                the Palace wrapper script that accepts ``-np``/``-nt`` flags.
                ``"binary"`` calls the solver binary with only ``config.json``.
            num_processes: Number of MPI processes. If None (default),
                uses all available CPUs.
            num_threads: Number of OpenMP threads to use for OpenMP builds, default is 1
                or the value of OMP_NUM_THREADS in the environment
            serial: When running direct Palace, pass the Palace wrapper's
                ``-serial`` flag so smoke tests can avoid MPI launchers.
            setup_commands: Optional shell setup commands to run in the same
                shell session as the local Palace command. This is intended for
                caller-provided runtime activation such as ``spack load palace``.
                Supported only with ``use_apptainer=False``.
            verbose: Print progress messages and stream Palace output in real time
            prepare_run_folder: Create the canonical Palace run-folder skeleton
                before launching Palace.
            check_runtime_version: Try to compare the Palace executable version
                with this simulation's target ``palace_version`` before
                running. Set to ``False`` to record a mismatch without failing.

        Returns:
            Raw Palace output files as ``dict[str, Path]``. Use
            ``resolve_palace_result(...).load_report()`` for typed reports.

        Raises:
            ValueError: If output_dir not set or Palace not configured
            FileNotFoundError: If mesh, config, or Palace not found
            RuntimeError: If simulation fails

        Example:
            >>> # Using Apptainer (default)
            >>> import os
            >>> os.environ["PALACE_SIF"] = "/path/to/Palace.sif"
            >>> results = sim.run_local()
            >>>
            >>> # Using Apptainer with explicit path
            >>> results = sim.run_local(palace_sif_path="/path/to/Palace.sif")
            >>>
            >>> # Using direct Palace installation
            >>> results = sim.run_local(use_apptainer=False)
            >>>
            >>> # Using direct Palace with custom executable path
            >>> results = sim.run_local(
            ...     use_apptainer=False, palace_executable="/usr/local/bin/palace"
            ... )
            >>>
            >>> # Using caller-provided shell setup, for example Spack
            >>> results = sim.run_local(
            ...     use_apptainer=False,
            ...     setup_commands=(
            ...         "source /path/to/spack/share/spack/setup-env.sh",
            ...         "spack load palace",
            ...     ),
            ... )
            >>> resolved = resolve_palace_result(sim.output_dir, problem_type="Driven")
            >>> report = resolved.load_report(require_report=True).require_report()
        """
        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
        return run_palace_local(
            self._output_dir,
            palace_sif_path=palace_sif_path,
            palace_executable=palace_executable,
            use_apptainer=use_apptainer,
            executable_mode=executable_mode,
            num_processes=num_processes,
            num_threads=num_threads,
            serial=serial,
            setup_commands=setup_commands,
            verbose=verbose,
            prepare_run_folder=prepare_run_folder,
            palace_version=self.palace_version,
            check_runtime_version=check_runtime_version,
        )

    # -------------------------------------------------------------------------
    # Port methods
    # -------------------------------------------------------------------------

    def add_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        from_layer: str | None = None,
        to_layer: str | None = None,
        length: float | None = None,
        offset: float = 0.0,
        impedance: float = 50.0,
        resistance: float | None = None,
        inductance: float | None = None,
        capacitance: float | None = None,
        direction: PalaceDirectionInput | None = None,
        excited: bool = True,
        generate_sheet: bool = True,
        geometry: Literal["inplane", "via"] = "inplane",
    ) -> None:
        """Add a single-element lumped port.

        Args:
            name: Port name (must match component port name)
            layer: Target layer for inplane ports
            from_layer: Bottom layer for via ports
            to_layer: Top layer for via ports
            length: Port extent along direction (um)
            offset: Shift the port inward along the waveguide (um).
                Positive moves away from the boundary, into the conductor.
            impedance: Port impedance (Ohms)
            resistance: Series resistance (Ohms)
            inductance: Series inductance (H)
            capacitance: Shunt capacitance (F)
            direction: Optional Palace LumpedPort.Direction solver
                field/polarization direction. Labels such as "+X" and "-Y"
                and finite nonzero 3-vectors are normalized to unit vectors.
                Port-sheet geometry still follows the gdsfactory port
                center, width, orientation, and layer.
            excited: Whether this port is excited
            generate_sheet: If True, mesh creates the horizontal sheet from
                the gdsfactory port. If False, mesh selects a layout-authored
                horizontal sheet from the simulation layer catalog.
            geometry: Port geometry type ("inplane" or "via")

        Example:
            >>> sim.add_port("o1", layer="topmetal2", length=5.0)
            >>> sim.add_port(
            ...     "feed", from_layer="metal1", to_layer="topmetal2", geometry="via"
            ... )
        """
        # Remove existing config for this port if any
        self.ports = [p for p in self.ports if p.name != name]

        self.ports.append(
            PortConfig(
                name=name,
                layer=layer,
                from_layer=from_layer,
                to_layer=to_layer,
                length=length,
                offset=offset,
                impedance=impedance,
                resistance=resistance,
                inductance=inductance,
                capacitance=capacitance,
                direction=direction,
                excited=excited,
                generate_sheet=generate_sheet,
                geometry=geometry,
            )
        )

    def add_cpw_port(
        self,
        name: str,
        *,
        layer: str,
        s_width: float,
        gap_width: float,
        length: float = 2.0,
        offset: float | None = None,
        impedance: float = 50.0,
        excited: bool = True,
        generate_sheet: bool = True,
    ) -> None:
        """Add a coplanar waveguide (CPW) port.

        CPW ports consist of two elements (upper and lower gaps) that are
        excited with opposite E-field directions to create the CPW mode.

        Place a single gdsfactory port at the center of the signal conductor.
        The two gap element surfaces are computed from s_width and gap_width.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            s_width: Width of the signal (center) conductor (um)
            gap_width: Width of each gap between signal and ground (um)
            length: Port extent along direction (um)
            offset: Shift the port inward along the waveguide (um).
                Positive moves away from the boundary, into the conductor.
                Defaults to length/2 (port flush with conductor edge).
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            generate_sheet: If True, mesh creates the two gap sheets from the
                gdsfactory port. If False, mesh selects layout-authored
                horizontal sheets from the simulation layer catalog.

        Example:
            >>> sim.add_cpw_port(
            ...     "left", layer="topmetal2", s_width=20, gap_width=15, length=5.0
            ... )
        """
        # Remove existing CPW port with same name if any
        self.cpw_ports = [p for p in self.cpw_ports if p.name != name]

        self.cpw_ports.append(
            CPWPortConfig(
                name=name,
                layer=layer,
                s_width=s_width,
                gap_width=gap_width,
                length=length,
                offset=offset,
                impedance=impedance,
                excited=excited,
                generate_sheet=generate_sheet,
            )
        )

    def add_wave_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        z_margin: float = 0.0,
        lateral_margin: float = 0.0,
        max_size: bool = False,
        mode: int = 1,
        excited: bool = True,
        offset: float = 0.0,
    ) -> None:
        """Add a single element wave port.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            z_margin: Margin in z direction
            lateral_margin: Margin in x/y directions
              Ignores lateral margin and port_width.
            max_size: When True, automatically set z_margin and lateral_margin
                to fill the full simulation domain boundary on that side.
                Overrides z_margin and lateral_margin values.
            mode: Mode number to excite.
            excited: Whether this port is excited
            offset: Offset distance used for scattering parameter de-embedding.

        Example:
            >>> sim.add_wave_port(
            ...     "w1",
            ...     layer="topmetal2",
            ...     max_size=True,
            ...     mode=1,
            ...     excited=True,
            ...     offset=0.0,
            ... )
        """
        self.wave_ports = [p for p in self.wave_ports if p.name != name]

        self.wave_ports.append(
            WavePortConfig(
                name=name,
                layer=layer,
                z_margin=z_margin,
                lateral_margin=lateral_margin,
                max_size=max_size,
                mode=mode,
                excited=excited,
                offset=offset,
            )
        )
