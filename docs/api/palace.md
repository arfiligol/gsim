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

## Results

::: gsim.palace.SParams
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.load_sparams
    options:
      show_source: false

::: gsim.palace.load_fields
    options:
      show_source: false

::: gsim.palace.load_driven_report
    options:
      show_source: false

::: gsim.palace.load_eigenmode_report
    options:
      show_source: false

::: gsim.palace.load_electrostatic_report
    options:
      show_source: false

::: gsim.palace.load_postprocessing_index_map
    options:
      show_source: false

::: gsim.palace.load_terminal_matrix
    options:
      show_source: false

::: gsim.palace.summarize_domain_loss
    options:
      show_source: false

::: gsim.palace.summarize_surface_loss
    options:
      show_source: false

::: gsim.palace.summarize_loss_budget
    options:
      show_source: false

::: gsim.palace.load_domain_energy_summary
    options:
      show_source: false

::: gsim.palace.load_surface_q_summary
    options:
      show_source: false

::: gsim.palace.load_domain_material_summary
    options:
      show_source: false

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

::: gsim.palace.load_dielectric_interface_summary
    options:
      show_source: false

## Advanced Result Details

These helpers expose lower-level Palace result tables, pass histories, and
indexed CSV provenance. They are reusable result APIs, but they live in
`gsim.palace.results` rather than the root notebook-facing simulation API.

::: gsim.palace.results.load_indexed_csv
    options:
      show_source: false

::: gsim.palace.results.load_eigenmodes
    options:
      show_source: false

::: gsim.palace.results.load_eigenmode_history
    options:
      show_source: false

::: gsim.palace.results.summarize_eigenmode_history
    options:
      show_source: false

::: gsim.palace.results.load_terminal_matrix_history
    options:
      show_source: false

::: gsim.palace.results.summarize_terminal_matrix_history
    options:
      show_source: false

::: gsim.palace.results.load_port_epr_summary
    options:
      show_source: false

::: gsim.palace.results.summarize_surface_q_by_interface
    options:
      show_source: false

## Advanced Runtime, Sweep, And Handoff Records

These helpers operate on runtime sidecars, sweep metadata, dry-run handoff
artifacts, and resource records. They are reusable workflow APIs, but they live
in their owning modules rather than the root notebook-facing simulation API.

::: gsim.palace.results.load_palace_run_summary
    options:
      show_source: false

::: gsim.palace.results.load_palace_sweep_summary
    options:
      show_source: false

::: gsim.palace.results.PalaceSweepPointSpec
    options:
      show_source: false
      inherited_members: false

::: gsim.palace.results.write_palace_sweep_points
    options:
      show_source: false

::: gsim.palace.results.write_palace_resource_record
    options:
      show_source: false

::: gsim.palace.results.write_palace_resource_record_from_log
    options:
      show_source: false

::: gsim.palace.results.write_palace_sweep_resource_index
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
