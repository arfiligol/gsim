"""Resolve Palace ParaView field-output artifacts.

The field loader reads existing ParaView datasets after a solver run. It does
not configure field saving or provide domain-specific plotting policy.
"""

from __future__ import annotations

from pathlib import Path

from gsim.palace.resolve.sources.path_utils import find_file


def load_fields(
    source: str | Path | dict,
    *,
    excitation: int = 1,
    cycle: int | None = None,
    boundary: bool = False,
):
    """Load the ParaView volume or boundary dataset for a Palace simulation.

    Requires the simulation to have been run with ``save_step >= 1``
    so that field data was written to disk.

    Args:
        source: Results dict from ``sim.run_local()`` / ``sim.run()``,
            or a path to the simulation directory.
        excitation: Excitation index (1-based) to load.
        cycle: ParaView cycle number (``None`` -> last available).
        boundary: If ``True``, load boundary surface fields
            (``driven_boundary/``) instead of volume fields
            (``driven/``).  Boundary data includes ``J_s_real``,
            ``Q_s_real``, etc.

    Returns:
        ``pyvista.DataSet`` with point data such as
        ``E_real``, ``B_real``, ``S`` (volume) or
        ``J_s_real``, ``Q_s_real`` (boundary).

    Raises:
        FileNotFoundError: If paraview output is missing.

    Example::

        vol = load_fields(results)
        bnd = load_fields(results, boundary=True)
        plot_cross_section(vol, normal="x", origin=0)
    """
    import pyvista as pv

    _, base_dir = _resolve_source(source, require_csv=False)
    pvtu_path = _find_paraview_dir(base_dir, excitation, cycle, boundary=boundary)
    return pv.read(str(pvtu_path))


def _find_paraview_dir(
    base_dir: Path,
    excitation: int,
    cycle: int | None,
    *,
    boundary: bool = False,
) -> Path:
    """Locate the ``.pvtu`` file for the requested excitation and cycle.

    Palace uses two output layouts depending on the port type:
    - Multi-excitation (wave ports):  ``paraview/driven/excitation_N/CycleNNNNNN/``
    - Single-excitation (lumped ports): ``paraview/driven/CycleNNNNNN/`` (no subdir)
    Both are searched, with the explicit ``excitation_N`` folder taking priority.
    """
    subdir = "driven_boundary" if boundary else "driven"
    search_roots = [
        base_dir,
        base_dir / "results" / "palace",
    ]
    exc_dir: Path | None = None
    for root in search_roots:
        # Layout 1: explicit excitation subfolder (wave ports / multi-excitation)
        candidate = root / "paraview" / subdir / f"excitation_{excitation}"
        if candidate.is_dir():
            exc_dir = candidate
            break
        # Layout 2: flat — Cycle dirs sit directly under driven/ (lumped ports)
        flat = root / "paraview" / subdir
        if flat.is_dir() and any(flat.iterdir()):
            exc_dir = flat
            break

    if exc_dir is None:
        msg = (
            f"ParaView output not found for excitation {excitation}. "
            "Ensure the simulation was run with save_step >= 1."
        )
        raise FileNotFoundError(msg)

    if cycle is not None:
        pvtu_dir = exc_dir / f"Cycle{cycle:06d}"
        candidates = sorted(pvtu_dir.rglob("*.pvtu"))
        if not candidates:
            msg = f"No .pvtu files found in {pvtu_dir}"
            raise FileNotFoundError(msg)
        return candidates[-1]

    # Auto-select last available cycle that contains actual field data.
    # Palace writes a final cycle with only Indicator/Rank (mesh partition);
    # skip it and pick the latest cycle with real solution fields.
    candidates = sorted(exc_dir.rglob("*.pvtu"), reverse=True)
    if not candidates:
        msg = (
            f"No .pvtu files found under {exc_dir}. "
            "Ensure the simulation was run with save_step >= 1."
        )
        raise FileNotFoundError(msg)

    import pyvista as pv

    _partition_only = {"Indicator", "Rank"}
    for pvtu in candidates:
        ds = pv.read(str(pvtu))
        if set(ds.point_data.keys()) != _partition_only:
            return pvtu

    # All cycles are partition-only — return the last one and let the
    # caller surface the "field not found" error with context.
    return candidates[0]


def _resolve_source(
    source: str | Path | dict,
    *,
    require_csv: bool = True,
) -> tuple[Path | None, Path]:
    """Turn *source* into ``(csv_path, base_dir)``."""
    if isinstance(source, dict):
        csv_val = source.get("port-S.csv")
        if csv_val is not None:
            csv_path = Path(csv_val)
            return csv_path, csv_path.parent
        for val in source.values():
            p = Path(val)
            if p.exists():
                return None, p.parent
        if require_csv:
            msg = "Results dict has no 'port-S.csv' entry"
            raise FileNotFoundError(msg)
        return None, Path()

    output_dir = Path(source)
    csv_path = find_file(output_dir, "port-S.csv")
    if csv_path is None and require_csv:
        msg = f"port-S.csv not found in {output_dir} or its subdirectories"
        raise FileNotFoundError(msg)
    return csv_path, output_dir
