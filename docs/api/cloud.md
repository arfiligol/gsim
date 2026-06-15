# Cloud API

Downloaded cloud results include a solver-specific runtime metadata sidecar when
the solver is known. Palace jobs write `palace_run_metadata.json`, so the same
`gsim.palace.results.load_palace_run_summary()` runtime surface can report
local and cloud execution metadata.

::: gsim.gcloud.run_simulation
    options:
      show_source: false

::: gsim.gcloud.get_status
    options:
      show_source: false

::: gsim.gcloud.wait_for_results
    options:
      show_source: false

::: gsim.gcloud.RunResult
    options:
      show_source: false
      inherited_members: false
      members: false
