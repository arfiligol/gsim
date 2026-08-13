# Cloud API

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

## Result caching

Passing `check_cache=True` to `sim.run()` looks for a completed cloud job with
byte-identical inputs and reuses its results instead of submitting a new job:

```python
sp = sim.run(check_cache=True)
```

The cache key is derived from the files written by `write_config()` — the same
bytes the solver consumes — rather than from the simulation object, so it also
covers changes to the generated solver script. A lookup failure is never fatal:
it degrades to a normal submit.

::: gsim.gcloud.check_cache
    options:
      show_source: false

::: gsim.gcloud.check_cache_for_dir
    options:
      show_source: false

::: gsim.hashing.compute_input_hash
    options:
      show_source: false
