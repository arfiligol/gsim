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
        - set_driven
        - set_material
        - set_numerical
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
        - set_eigenmode
        - set_material
        - set_numerical
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
        - set_electrostatic
        - set_material
        - set_numerical
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
        - set_magnetostatic
        - set_material
        - set_numerical
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

::: gsim.palace.ports.extract_ports
    options:
      show_source: false

::: gsim.palace.ports.configure_cpw_port
    options:
      show_source: false

::: gsim.palace.ports.configure_wave_port
    options:
      show_source: false

::: gsim.palace.ports.PalacePort
    options:
      show_source: false
      inherited_members: false

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
over the same folder to fill `logs/` and `results/palace/`. AEDT/HFSS export
and result packaging are public-PDK responsibilities, not `gsim.palace`
responsibilities.

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

::: gsim.palace.results.SimulationPerformance
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

::: gsim.palace.models.results.SimulationResult
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
