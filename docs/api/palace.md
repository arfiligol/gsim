# Palace API

This API reference is curated around notebook-facing and downstream-facing
entry points. Deep modules may contain reusable implementation types, but a
symbol should be promoted to the `gsim.palace` import surface only when users
are expected to import it directly in notebooks, public fixtures, or downstream
packages. Helper-only lowering details should stay in their owning modules.

## Simulation Classes

::: gsim.palace.DrivenSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - activate_substrate
        - activate_inter_die_vacuum
        - activate_outer_vacuum
        - set_driven
        - set_palace_version
        - set_material
        - set_numerical
        - set_refinement
        - set_linear_solver
        - add_port
        - add_cpw_port
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - write_config
        - run
        - start
        - upload
        - get_status
        - wait_for_results

::: gsim.palace.EigenmodeSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - activate_substrate
        - activate_inter_die_vacuum
        - activate_outer_vacuum
        - set_eigenmode
        - set_palace_version
        - set_material
        - set_numerical
        - set_refinement
        - set_linear_solver
        - add_port
        - add_cpw_port
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run

::: gsim.palace.ElectrostaticSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - activate_substrate
        - activate_inter_die_vacuum
        - activate_outer_vacuum
        - set_electrostatic
        - set_palace_version
        - set_material
        - set_numerical
        - set_refinement
        - set_linear_solver
        - add_terminal
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run

::: gsim.palace.MagnetostaticSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - activate_substrate
        - activate_inter_die_vacuum
        - activate_outer_vacuum
        - set_magnetostatic
        - set_palace_version
        - set_material
        - set_numerical
        - set_refinement
        - set_linear_solver
        - add_current_source
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - write_config
        - run

## Advanced Port Authoring

Notebook workflows should usually configure ports through the problem-specific
simulation methods such as `add_port()`, `add_cpw_port()`, `add_wave_port()`,
and `add_terminal()`. Lower-level port geometry, extraction, and CPW/wave-port
helpers live in `gsim.palace.ports` or the port config models instead of the
root `gsim.palace` import surface.

For lumped ports, `add_port(direction=...)` describes Palace solver
field/polarization direction, not generated sheet geometry. String inputs such
as `"X"`, `"+X"`, and `"-Y"` and finite nonzero 3-vectors are normalized to
unit Cartesian vectors before writing `Boundaries.LumpedPort.Direction`.
Generated in-plane port sheets still come from the GDSFactory port `center`,
`width`, `orientation`, and layer.

For layout-authored horizontal solver sheets, pass a PDK-owned simulation layer
catalog with `sim.set_simulation_layers(...)` and declare the port with
`generate_sheet=False`. The component port layer must be a registered
simulation-only solver sheet layer; `layer=...` on `add_port()` remains the
target stack/material layer. Mesh generation selects the unique authored polygon
covering the port center and then follows the same physical-group to
`Boundaries.LumpedPort` pipeline as generated sheets. This authored-sheet path
does not apply to vertical via ports.

::: gsim.palace.models.SimulationLayerCatalog
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.ports.extract_ports
    options:
      show_source: false

::: gsim.palace.ports.configure_cpw_port
    options:
      show_source: false

::: gsim.palace.ports.configure_wave_port
    options:
      show_source: false

::: gsim.palace.models.CPWPortConfig
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.models.WavePortConfig
    options:
      show_source: false
      inherited_members: false

## Mesh Generation

These APIs control mesh construction. They should change generated geometry or
meshing behavior, not Palace postprocessing/reportability.

`sim.set_stack(...)` loads stack facts from the active PDK, YAML, a
`gsim.common.LayerStack`, or a `gdsfactory` `LayerStack`. Direct stack inputs
are copied into simulation-owned state before material overrides or mesh-time
changes are applied, so caller-owned stacks are not mutated. The no-argument
and keyword modes remain the simple stack path; they may still auto-create the
existing oxide/passivation background dielectrics when those options are left
enabled.

For PDKs that define physical simulation regions as named stack layers, activate
the regions explicitly after `set_stack(...)`. The public activation calls are
`activate_substrate(layer, *, die=None, margin_x=0.0, margin_y=0.0,
material=None)`, `activate_inter_die_vacuum(layer="D0_TO_D1_GAP", *,
lower_die="D0", upper_die="D1", margin_x=0.0, margin_y=0.0, material=None)`,
and `activate_outer_vacuum(layer="OUTER_VACUUM", *, margin_x=0.0,
margin_y=0.0, z_above=0.0, z_below=0.0, material=None)`. Region margins and
outer-vacuum z extents are non-negative and belong on these activation calls.
Layer and die names must be non-empty.

Activated regions become 3D mesh regions using the stack layer name as the
physical group and manifest identity. A flip-chip PDK can therefore keep
`D0_SUBSTRATE`, `D1_SUBSTRATE`, `D0_TO_D1_GAP`, and `OUTER_VACUUM` visible in
mesh groups, manifests, postprocessing index maps, and material-resolution
sidecars while still resolving Palace material properties from `Si` or
`vacuum`. A region-level `material=...` override is preserved as group and
manifest provenance and is the material used by `mesh.config_generator` when it
assembles `Domains.Materials`.

Explicit activated-region mode is separate from the legacy airbox workflow.
Calling `set_airbox(...)` before or after activating regions raises a hard
error. Legacy airbox controls such as `air_margin`,
`airbox_margin_x`/`airbox_margin_y`, `airbox_z_above`/`airbox_z_below`,
mesh-time `z_above`/`z_below`, and mesh-time `airbox_margin` are rejected in
explicit-region mode. Workflows that want the simple enclosing airbox should
keep using `set_stack(...)` plus `set_airbox(...)` without activating named
stack regions.

::: gsim.palace.models.ActivatedRegion
    options:
      show_source: false
      inherited_members: false

## Versioned Palace Config

`sim.set_palace_version(...)` selects the Palace configuration schema target.
The first supported targets are `0.15.0` and `0.16.0`, with `0.16.0` as the
default. `write_config(validate_schema=True)` validates the final assembled
`config.json` against the selected schema before returning.

`sim.set_refinement(...)` owns the common `Model.Refinement` fragment,
`sim.set_linear_solver(...)` owns common `Solver.Linear` settings, and
`sim.set_output_formats(...)` owns `Problem.OutputFormats`. The simpler
`sim.set_numerical(...)` remains a convenience wrapper for common order,
tolerance, solver type, and device settings. Rare Palace-native keys can be
passed as Palace JSON fragments and are checked by final schema validation.

`Domains` and `Boundaries` remain mesh-derived sections assembled by
`mesh.config_generator`. Notebook code should not pass those sections through
untyped hints; the generator rejects hints that would overwrite `Domains`,
`Boundaries`, `Problem.Type`, `Problem.Output`, or `Model.Mesh`.

::: gsim.palace.MeshConfig
    options:
      show_source: false
      inherited_members: false
      members:
        - coarse
        - default
        - fine

::: gsim.palace.generate_mesh
    options:
      show_source: false

## Mesh Artifacts And Reportability

These APIs operate on generated mesh artifacts and manifests. Use them when a
workflow needs auditable physical-name, postprocessing-index, or material
provenance sidecars without adding new mesh-generation knobs.

::: gsim.palace.mesh.MeshResult
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.mesh.MeshManifest
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.mesh.SurfaceFluxSpec
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.mesh.build_postprocessing_config_from_manifest
    options:
      show_source: false

::: gsim.palace.mesh.build_dielectric_interface_specs_from_assignments
    options:
      show_source: false

::: gsim.palace.mesh.build_dielectric_interface_specs_from_material_kinds
    options:
      show_source: false

::: gsim.palace.mesh.build_interface_surface_catalog
    options:
      show_source: false

Surface EPR interface metadata comes from generated mesh groups after full-3D
Gmsh interface discovery. Source-polygon Surface EPR bands are not a supported
production mesh input. Finite-metal B lowering emits 50 nm planar top/bottom
band/core physical groups from discovered shell interfaces; vertical sidewalls
remain total channels.

## Advanced Mesh Postprocessing Authoring

Notebook workflows should usually create dielectric-interface requests through
the assignment or material-kind builders above. Manual dielectric-interface
specs live in the postprocessing owner module instead of the
`gsim.palace.mesh` package root.

::: gsim.palace.mesh.postprocessing.DielectricInterfaceSpec
    options:
      show_source: false
      inherited_members: false

## Stack

Palace workflows usually resolve stack data through `set_stack()` or the active
PDK. Manual stack construction uses the shared `gsim.common` owner module
because stack models are reused across solver front ends.

Import manual stack models from `gsim.common.LayerStack` and
`gsim.common.Layer`, or from the deeper `gsim.common.stack` module. The
canonical API entries live on the Common API page.

## Materials

Palace config generation resolves material overlays internally when writing
solver inputs. Advanced callers that need to inspect material dispersion and
overlay provenance should use the Palace materials owner module instead of the
`gsim.palace` package root.

::: gsim.palace.materials.resolve_palace_materials_with_report
    options:
      show_source: false

## Palace Run Folder

Palace workflows use one canonical run folder. `run()`, `run_local()`, and
`generate_handoff_package()` create this structure by default:

```text
my_run/
  config.json
  palace.msh
  run_palace.sbatch
  geometry/
    design.gds
  metadata/
    mesh_manifest.json
    palace_index_map.json
    palace_material_resolution.json
    port_information.json
    palace_handoff_metadata.json
    palace_handoff_archive_manifest.json
    palace_run_metadata.json
    palace_resource_record.json
  logs/
  results/
    palace/
```

Root files are Palace execution inputs or launchers. `metadata/` stores gsim
semantic sidecars. `results/palace/` stores raw Palace solver outputs and
exists even before Palace runs. `geometry/design.gds` is an optional review
snapshot only; it is not a Palace execution input and Resolve does not load it
into reports.

`generate_handoff_package()` writes a tarball beside the run folder by default:

```text
my_run-palace.tar.gz
```

The archive root is `my_run/`, so post-run result archives can be extracted
over the same folder to fill `logs/` and `results/palace/`. `gsim.palace`
owns generic Palace result package profiles and commands; PDKs own
example-specific recommendations and site/profile defaults.

### Return Results From HPC

After the Palace job finishes, enter the remote run folder and choose the
smallest archive that supports local analysis.

| Profile | Archive suffix | Use it when | Includes | Excludes |
| --- | --- | --- | --- | --- |
| `light` | `-light.tar.gz` | You only need scalar tables, logs, metadata, and report inputs. | CSVs, logs, metadata, manifests, report inputs. | Palace field directories, VTK/BP/HDF5 field files, and meshes. |
| `with-fields` | `-with-fields.tar.gz` | Numeric postprocessing needs Palace GridFunction field output. | Everything in `light`, plus `results/**/gridfunction/**`. | ParaView output, VTK/PVTU field files, and solver `.msh` files. |
| `with-fields-and-meshes` | `-with-fields-and-meshes.tar.gz` | Local audit needs solver mesh identity or physical groups. | Everything in `with-fields`, plus solver mesh and mesh-generation identity artifacts. | ParaView output and VTK/PVTU field files by default. |
| `full` | `-full.tar.gz` | You intentionally need the complete run folder. | The whole run folder. | Nothing by default. |

Light result package:

```bash
cd <remote-run-folder>
BUNDLE_ID="$(basename "$PWD")"
tar \
  --checkpoint=1000 \
  --checkpoint-action=dot \
  -czf "../${BUNDLE_ID}-light.tar.gz" \
  --transform "s|^\.$|${BUNDLE_ID}|;s|^\./|${BUNDLE_ID}/|" \
  --exclude='./results/palace/paraview' \
  --exclude='./results/palace/gridfunction' \
  --exclude='./results/palace/iteration*/paraview' \
  --exclude='./results/palace/iteration*/gridfunction' \
  --exclude='./results/*/palace/paraview' \
  --exclude='./results/*/palace/gridfunction' \
  --exclude='./results/*/palace/iteration*/paraview' \
  --exclude='./results/*/palace/iteration*/gridfunction' \
  --exclude='*.msh' \
  --exclude='*.vtu' \
  --exclude='*.pvtu' \
  --exclude='*.vtk' \
  --exclude='*.bp' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.mesh' \
  --exclude='*.sol' \
  .
```

With fields:

```bash
cd <remote-run-folder>
BUNDLE_ID="$(basename "$PWD")"
tar \
  --checkpoint=1000 \
  --checkpoint-action=dot \
  -czf "../${BUNDLE_ID}-with-fields.tar.gz" \
  --transform "s|^\.$|${BUNDLE_ID}|;s|^\./|${BUNDLE_ID}/|" \
  --exclude='./results/palace/paraview' \
  --exclude='./results/palace/iteration*/paraview' \
  --exclude='./results/*/palace/paraview' \
  --exclude='./results/*/palace/iteration*/paraview' \
  --exclude='*.msh' \
  --exclude='*.vtu' \
  --exclude='*.pvtu' \
  --exclude='*.vtk' \
  --exclude='*.bp' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  .
```

With fields and meshes:

```bash
cd <remote-run-folder>
BUNDLE_ID="$(basename "$PWD")"
tar \
  --checkpoint=1000 \
  --checkpoint-action=dot \
  -czf "../${BUNDLE_ID}-with-fields-and-meshes.tar.gz" \
  --transform "s|^\.$|${BUNDLE_ID}|;s|^\./|${BUNDLE_ID}/|" \
  --exclude='./results/palace/paraview' \
  --exclude='./results/palace/iteration*/paraview' \
  --exclude='./results/*/palace/paraview' \
  --exclude='./results/*/palace/iteration*/paraview' \
  --exclude='*.vtu' \
  --exclude='*.pvtu' \
  --exclude='*.vtk' \
  .
```

Download and extract the archive locally from the parent folder:

```bash
scp <user>@<hpc-host>:<remote-run-parent>/<archive-name>.tar.gz .
tar -xzf <archive-name>.tar.gz
```

`generate_handoff_package()` is a Run Stage API. It returns
`PalaceRunHandle`, which records the packaged run folder and optional launcher
or archive paths. It does not load typed reports. Review/report loading starts
in the Resolve Stage:

```python
handle = sim.generate_handoff_package()
resolved = resolve_palace_result(handle.run_folder, problem_type="Driven")
bundle = resolved.load_report()
```

`run_local()` is also a Run Stage API. For direct local Palace execution,
callers may pass `setup_commands` to activate the runtime in the same shell
session as Palace, for example `source .../spack/setup-env.sh` followed by
`spack load palace`. These commands are caller-owned environment setup; `gsim`
uses them only to launch Palace and does not interpret site policy.

::: gsim.palace.PalaceRunHandle
    options:
      show_source: false
      inherited_members: false

## Resolve Pipeline

Palace result presentation has one pipeline entry point:

```python
from gsim.palace import resolve_palace_result

resolved = resolve_palace_result("build/my_run", problem_type="Driven")
bundle = resolved.load_report(require_report=True)
report = bundle.require_report()
report.show_all_results()
```

`gsim.palace.resolve` loads Palace artifacts and orchestrates report assembly.
`gsim.palace.results` owns semantic Typed Data and Problem Type Report models:
S-parameters, Eigenmodes, terminal matrices, loss, benchmark data, and the
report objects that aggregate them. `gsim.palace.display` owns only general
table/plot helper primitives.

Pass a literal `problem_type` when review code needs concrete static typing.
For example, `problem_type="Driven"` lets type checkers follow
`bundle.require_report()` to `DrivenReport`, then to typed data such as
`report.sparams`. When the problem type is inferred from files or stored in a
dynamic `str | None`, static tools can only prove the common report contract.

::: gsim.palace.resolve_palace_result
    options:
      show_source: false

::: gsim.palace.PalaceResolvedResult
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.PalaceResultBundle
    options:
      show_source: false
      inherited_members: false

## Typed Data

::: gsim.palace.SParams
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.results.Eigenmodes
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.results.TerminalMatrix
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.results.ReportLoss
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.results.SimulationBenchmark
    options:
      show_source: false
      inherited_members: false

## Problem Type Reports

::: gsim.palace.DrivenReport
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.EigenmodeReport
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.ElectrostaticReport
    options:
      show_source: false
      inherited_members: false

## Advanced Return Models

Notebook workflows usually inspect the objects returned by simulation methods
directly. Callers that need explicit type imports for mesh or validation return
values should use the owner module, not the root `gsim.palace` import surface.

::: gsim.palace.mesh.MeshResult
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.models.results.ValidationResult
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.resolve.derived.materials.load_dielectric_interface_summary
    options:
      show_source: false

## Advanced Result Details

These helpers expose lower-level Palace result tables, pass histories, return
models, and indexed CSV provenance. They are reusable implementation APIs, but
they live in the `resolve.loaders` and `resolve.derived` owner modules rather
than the root notebook-facing simulation API or the high-level
`gsim.palace.resolve` facade.

::: gsim.palace.results.TerminalMatrix
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.resolve.loaders.indexed_csv.load_indexed_csv
    options:
      show_source: false

::: gsim.palace.resolve.derived.participation.load_domain_energy_summary
    options:
      show_source: false

::: gsim.palace.resolve.derived.participation.load_surface_q_summary
    options:
      show_source: false

::: gsim.palace.resolve.derived.loss.summarize_domain_loss
    options:
      show_source: false

::: gsim.palace.resolve.derived.loss.summarize_surface_loss
    options:
      show_source: false

::: gsim.palace.resolve.derived.loss.summarize_loss_budget
    options:
      show_source: false

::: gsim.palace.resolve.loaders.eigenmodes.load_eigenmodes
    options:
      show_source: false

::: gsim.palace.resolve.loaders.eigenmodes.load_eigenmode_history
    options:
      show_source: false

::: gsim.palace.resolve.loaders.eigenmodes.summarize_eigenmode_history
    options:
      show_source: false

::: gsim.palace.resolve.derived.terminal_matrices.load_terminal_matrix_history
    options:
      show_source: false

::: gsim.palace.resolve.derived.terminal_matrices.summarize_terminal_matrix_history
    options:
      show_source: false

::: gsim.palace.resolve.derived.participation.load_port_epr_summary
    options:
      show_source: false

::: gsim.palace.resolve.derived.participation.summarize_surface_q_by_interface
    options:
      show_source: false

## Advanced Runtime, Sweep, And Handoff Records

These helpers operate on runtime sidecars, sweep metadata, dry-run handoff
artifacts, and resource records. They are reusable workflow APIs, but they live
in their owning modules rather than the root notebook-facing simulation API.

::: gsim.palace.resolve.sources.run_summary.load_palace_run_summary
    options:
      show_source: false

::: gsim.palace.resolve.sweeps.load_palace_sweep_summary
    options:
      show_source: false

::: gsim.palace.resolve.PalaceSweepPointSpec
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.resolve.sources.sidecars.write_palace_sweep_points
    options:
      show_source: false

::: gsim.palace.resolve.sources.resources.write_palace_resource_record
    options:
      show_source: false

::: gsim.palace.resolve.sources.resources.write_palace_resource_record_from_log
    options:
      show_source: false

::: gsim.palace.resolve.sweeps.write_palace_sweep_resource_index
    options:
      show_source: false

::: gsim.palace.handoff.PalaceSlurmSbatchSpec
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.handoff.PalaceSlurmSweepArraySpec
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.handoff.load_palace_slurm_profile_catalog
    options:
      show_source: false

::: gsim.palace.handoff.resolve_palace_slurm_profile
    options:
      show_source: false

::: gsim.palace.handoff.write_palace_slurm_sbatch_handoff
    options:
      show_source: false

::: gsim.palace.handoff.write_palace_slurm_sweep_array_handoff
    options:
      show_source: false

::: gsim.palace.handoff.write_palace_run_handoff_archive_manifest
    options:
      show_source: false

::: gsim.palace.handoff.write_palace_sweep_handoff_archive_manifest
    options:
      show_source: false
