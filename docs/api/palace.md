# Palace API

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

::: gsim.palace.mesh.build_postprocessing_config_from_manifest
    options:
      show_source: false

::: gsim.palace.mesh.build_dielectric_interface_specs_from_assignments
    options:
      show_source: false

::: gsim.palace.mesh.build_dielectric_interface_specs_from_material_kinds
    options:
      show_source: false

## Stack

::: gsim.palace.LayerStack
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.palace.Layer
    options:
      show_source: false
      inherited_members: false
      members: false

## Materials

::: gsim.palace.resolve_palace_materials_with_report
    options:
      show_source: false

## Results

::: gsim.palace.load_palace_run_summary
    options:
      show_source: false

::: gsim.palace.load_palace_sweep_summary
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

::: gsim.palace.load_dielectric_interface_summary
    options:
      show_source: false
