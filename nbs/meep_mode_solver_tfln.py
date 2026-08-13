# ---
# jupyter:
#   jupytext:
#     jupytext_version: 1.19.2
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.006539, "end_time": "2026-07-24T16:33:50.087547", "exception": false, "start_time": "2026-07-24T16:33:50.081008", "status": "completed"}
# # Mode Solver — TFLN Ridge Waveguide (meep)
#
# Fundamental TE mode of thin-film lithium niobate ridge waveguide at lambda=1.55 µm.
# Reference: Ying Li et al., ACS Omega 2023, 8(10), 9644–9651.
#
# **Design:** SiO2 cladding / LiNbO3 slab 220 nm / LiNbO3 ridge 180 nm (total 400 nm),
# ridge width 1.1 µm, sidewall angle 17°.
#
# **Expected:** n_eff ~ 1.85, n_group ~ 2.20.
# ``background_material="sio2"`` fills unpatterned space with SiO2.

# %% [markdown] papermill={"duration": 0.002776, "end_time": "2026-07-24T16:33:50.095297", "exception": false, "start_time": "2026-07-24T16:33:50.092521", "status": "completed"}
# ### Imports

# %% papermill={"duration": 7.319947, "end_time": "2026-07-24T16:33:57.417774", "exception": false, "start_time": "2026-07-24T16:33:50.097827", "status": "completed"}
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

import gsim.meep as gm
from gsim.common.stack.extractor import Layer, LayerStack

plt.close()

gf.gpdk.PDK.activate()

# %% [markdown] papermill={"duration": 0.004634, "end_time": "2026-07-24T16:33:57.427896", "exception": false, "start_time": "2026-07-24T16:33:57.423262", "status": "completed"}
# ### LiNbO3 material (Zelmon 1997)
#
# LiNbO3 is birefringent and is now registered as a uniaxial material
# in ``gsim.common.stack.materials.MATERIALS_DB`` (notebook no longer
# manually registers it).  Both ordinary and extraordinary Sellmeier
# models from Zelmon et al. (JOSA B 14(12), 3319--3322, 1997) are included:
#
# Ordinary:
#   n_o^2 = 1 + 2.6734 lam^2/(lam^2 - 0.01764) + 1.2290 lam^2/(lam^2 - 0.05914)
#             + 12.614 lam^2/(lam^2 - 474.60)
#
# Extraordinary:
#   n_e^2 = 1 + 2.9804 lam^2/(lam^2 - 0.02047) + 0.5981 lam^2/(lam^2 - 0.0666)
#             + 8.9543 lam^2/(lam^2 - 416.08)
#
# For x-cut TFLN the TE mode's dominant E-field aligns with the
# extraordinary axis (zz).


# %% [markdown] papermill={"duration": 0.0038, "end_time": "2026-07-24T16:33:57.436203", "exception": false, "start_time": "2026-07-24T16:33:57.432403", "status": "completed"}
# ### Build the GDS component

# %% papermill={"duration": 0.174397, "end_time": "2026-07-24T16:33:57.625614", "exception": false, "start_time": "2026-07-24T16:33:57.451217", "status": "completed"}
SLAB_WIDTH = 5.0  # um --- wide slab
CORE_WIDTH = 1.1  # um --- w0 from reference design
LENGTH = 10.0  # um --- waveguide length (arbitrary for mode solving)

c = gf.Component()

# LiNbO3 ridge (layer 2) --- narrow core on top
c.add_polygon(
    [
        (-LENGTH / 2, -CORE_WIDTH / 2),
        (LENGTH / 2, -CORE_WIDTH / 2),
        (LENGTH / 2, CORE_WIDTH / 2),
        (-LENGTH / 2, CORE_WIDTH / 2),
    ],
    layer=(2, 0),
)

# Ports at both ends
c.add_port(
    name="o1",
    center=(-LENGTH / 2, 0),
    width=CORE_WIDTH,
    orientation=180,
    layer=(1, 0),
)
c.add_port(
    name="o2",
    center=(LENGTH / 2, 0),
    width=CORE_WIDTH,
    orientation=0,
    layer=(1, 0),
)

print(f"Component: {c.name}")
print(f"  Ports:  {[p.name for p in c.ports]}")
print(f"  Layers: {list(c.layers)}")

# %% [markdown] papermill={"duration": 0.003233, "end_time": "2026-07-24T16:33:57.635212", "exception": false, "start_time": "2026-07-24T16:33:57.631979", "status": "completed"}
# ### Layer stack
#
# SiO2 fills background via ``background_material="sio2"``.

# %% papermill={"duration": 0.016785, "end_time": "2026-07-24T16:33:57.660454", "exception": false, "start_time": "2026-07-24T16:33:57.643669", "status": "completed"}
SLAB_THICKNESS = 0.22  # um  (h3)
CORE_THICKNESS = 0.40  # um --- total LiNbO3 thickness
RIDGE_THICKNESS = CORE_THICKNESS - SLAB_THICKNESS  # 0.18 um

layers = {
    "box": Layer(
        name="box",
        gds_layer=(0, 0),
        zmin=-1.0,
        zmax=0.0,
        thickness=1.0,
        material="sio2",
        layer_type="dielectric",
    ),
    "slab": Layer(
        name="slab",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=SLAB_THICKNESS,
        thickness=SLAB_THICKNESS,
        material="linbo3",
        layer_type="dielectric",
    ),
    "ridge": Layer(
        name="ridge",
        gds_layer=(2, 0),
        zmin=SLAB_THICKNESS,
        zmax=CORE_THICKNESS,
        thickness=RIDGE_THICKNESS,
        material="linbo3",
        layer_type="dielectric",
        sidewall_angle=17.0,
    ),
}
stack = LayerStack(layers=layers)

print("Layer stack (+ SiO2 background):")
for name, l in stack.layers.items():
    print(
        f"  {name:6s}  z=[{l.zmin:+.3f}, {l.zmax:+.3f}]  "
        f"t={l.thickness:.3f}  material={l.material}  gds={l.gds_layer}"
    )

# %% [markdown] papermill={"duration": 0.015262, "end_time": "2026-07-24T16:33:57.679862", "exception": false, "start_time": "2026-07-24T16:33:57.664600", "status": "completed"}
# ### Solve

# %% papermill={"duration": 1925.213727, "end_time": "2026-07-24T17:06:02.898833", "exception": false, "start_time": "2026-07-24T16:33:57.685106", "status": "completed"}
WAVELENGTH = 1.55  # um
RESOLUTION = 64  # grid points per um
PML_THICKNESS = WAVELENGTH  # um

sim = gm.Simulation(
    geometry=gm.Geometry(component=c, stack=stack),
    domain=gm.Domain(
        pml=PML_THICKNESS,
        margin_z=(0.0, 0.5),
    ),
)
sim.mode_solver.wavelengths = [WAVELENGTH]
sim.mode_solver.fundamental().at_port("o1")
sim.mode_solver.y_span = SLAB_WIDTH
sim.mode_solver.n_field_y = 1000
sim.mode_solver.n_field_z = 1000
sim.mode_solver.background_material = "sio2"

sweep = sim.solve_modes()
mode = sweep.at(WAVELENGTH).band(1)

print(f"n_eff     = {mode.n_eff}")
print(f"n_group   = {mode.n_group}")
print(f"kdom      = {[f'{k:.6f}' for k in mode.kdom]}")
print(f"band      = {mode.band_num}, parity = {mode.parity}")
print(f"fields    = {list(mode.fields.keys())}")
for comp, arr in mode.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

# %% [markdown] papermill={"duration": 0.016131, "end_time": "2026-07-24T17:06:02.925467", "exception": false, "start_time": "2026-07-24T17:06:02.909336", "status": "completed"}
# ### Index profile

# %% papermill={"duration": 1.695063, "end_time": "2026-07-24T17:06:04.627531", "exception": false, "start_time": "2026-07-24T17:06:02.932468", "status": "completed"}
mode.plot_index(show=True)

# %% [markdown] papermill={"duration": 0.004876, "end_time": "2026-07-24T17:06:04.637601", "exception": false, "start_time": "2026-07-24T17:06:04.632725", "status": "completed"}
# ### Mode profile (interactive)
#
# ``|field|`` maps for every component with zoom / pan / hover. The view is
# auto-cropped to the mode region and axes use equal aspect.

# %% papermill={"duration": 1.508412, "end_time": "2026-07-24T17:06:06.152802", "exception": false, "start_time": "2026-07-24T17:06:04.644390", "status": "completed"}
import plotly.graph_objects as go
from plotly.subplots import make_subplots

comps = [c for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if c in mode.fields]

# Crop to the (small, centered) mode region and downsample so the embedded
# heatmaps stay light-weight for the docs while remaining smooth on zoom.
energy = sum(np.abs(mode.fields[c]) ** 2 for c in comps)
thr = 0.02 * energy.max()
yi = np.where(energy.max(axis=0) > thr)[0]
zi = np.where(energy.max(axis=1) > thr)[0]
dy, dz = mode.y_grid[1] - mode.y_grid[0], mode.z_grid[1] - mode.z_grid[0]
pad = 0.4  # µm margin around the mode
y0, y1 = (
    max(yi.min() - int(pad / dy), 0),
    min(yi.max() + int(pad / dy) + 1, mode.y_grid.size),
)
z0, z1 = (
    max(zi.min() - int(pad / dz), 0),
    min(zi.max() + int(pad / dz) + 1, mode.z_grid.size),
)
max_pts = 140  # per axis
sy, sz = max(1, (y1 - y0) // max_pts), max(1, (z1 - z0) // max_pts)
y_sub, z_sub = mode.y_grid[y0:y1:sy], mode.z_grid[z0:z1:sz]

fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[f"|{c}|" for c in comps],
    horizontal_spacing=0.08,
    vertical_spacing=0.2,
)

for i, comp in enumerate(comps):
    row, col = i // 3 + 1, i % 3 + 1
    # Mode fields are in arbitrary units; normalize each panel to its own max
    # and quantize to 8-bit (0..255). Plotly embeds arrays as binary, so this
    # keeps the notebook small while staying visually identical.
    panel = np.abs(mode.fields[comp][z0:z1:sz, y0:y1:sy])
    panel = np.round(panel / panel.max() * 255).astype(np.uint8)
    fig.add_trace(
        go.Heatmap(
            x=y_sub,
            y=z_sub,
            z=panel,
            colorscale="Inferno",
            showscale=False,
            zmin=0,
            zmax=255,
            hovertemplate=(
                "y=%{x:.3f} µm<br>z=%{y:.3f} µm<br>"
                "|" + comp + "| (rel.)=%{z}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )
    x_anchor = "x" if i == 0 else f"x{i + 1}"
    fig.update_xaxes(title_text="y (µm)", row=row, col=col)
    fig.update_yaxes(
        title_text="z (µm)", scaleanchor=x_anchor, scaleratio=1, row=row, col=col
    )

fig.update_layout(
    height=650,
    autosize=True,
    margin=dict(t=70, b=60),
    title=dict(
        text=(
            f"lambda={WAVELENGTH:.2f} µm, w0={CORE_WIDTH:.1f} µm, "
            f"n_eff={mode.n_eff:.4f}"
        ),
        x=0.5,
        xanchor="center",
        font=dict(size=13),
    ),
)
fig.show(config={"responsive": True})

# %% [markdown] papermill={"duration": 0.007377, "end_time": "2026-07-24T17:06:06.169543", "exception": false, "start_time": "2026-07-24T17:06:06.162166", "status": "completed"}
# ### Width sweep — band diagram
#
# Sweep the ridge width ``w0`` and track ``n_eff`` of the first few bands.
# Narrow ridges guide fewer modes, so higher-band curves appear only once the
# width is wide enough to support them. Modes below the SiO2 cladding index are
# radiative and are dropped (dashed line = cladding cutoff).
#
# Each width is an independent cloud mode-solve. Jobs are submitted in batches
# (up to ``MAX_CONCURRENT`` in flight) and waited on concurrently, so the whole
# sweep finishes in a few minutes rather than running one width at a time.

# %% papermill={"duration": 2210.842074, "end_time": "2026-07-24T17:42:57.024025", "exception": false, "start_time": "2026-07-24T17:06:06.181951", "status": "completed"}
import shutil
import tempfile
from pathlib import Path

from gsim import gcloud
from gsim.common.stack.materials import resolve_material_at_wavelength

# Snap widths to the 0.002 µm grid so gdsfactory ports stay grid-aligned.
W0_VALUES = np.round(np.linspace(0.3, 3.0, 25) / 0.002) * 0.002  # um
N_SWEEP_BANDS = 8
MAX_CONCURRENT = 10  # cloud jobs in flight at once

# SiO2 cladding index at this wavelength — guided modes must lie above it.
_clad = resolve_material_at_wavelength("sio2", WAVELENGTH)
_eps = _clad.permittivity
n_clad = float(np.sqrt(_eps if np.isscalar(_eps) else np.mean(_eps)))


def tfln_component(core_width: float) -> gf.Component:
    """Build the TFLN ridge component for a given ridge width."""
    comp = gf.Component()
    comp.add_polygon(
        [
            (-LENGTH / 2, -core_width / 2),
            (LENGTH / 2, -core_width / 2),
            (LENGTH / 2, core_width / 2),
            (-LENGTH / 2, core_width / 2),
        ],
        layer=(2, 0),
    )
    comp.add_port(
        name="o1",
        center=(-LENGTH / 2, 0),
        width=core_width,
        orientation=180,
        layer=(1, 0),
    )
    comp.add_port(
        name="o2",
        center=(LENGTH / 2, 0),
        width=core_width,
        orientation=0,
        layer=(1, 0),
    )
    return comp


def submit_mode_solve(core_width: float) -> str:
    """Upload + start one cloud mode-solve job (non-blocking). Returns job_id."""
    sim = gm.Simulation(
        geometry=gm.Geometry(component=tfln_component(core_width), stack=stack),
        domain=gm.Domain(pml=PML_THICKNESS, margin_z=(0.0, 0.5)),
    )
    sim.mode_solver.wavelengths = [WAVELENGTH]
    sim.mode_solver.first(N_SWEEP_BANDS).at_port("o1")
    sim.mode_solver.y_span = SLAB_WIDTH
    sim.mode_solver.background_material = "sio2"  # n_field=0 -> n_eff only

    tmp = Path(tempfile.mkdtemp(prefix="meep_mode_sweep_"))
    try:
        sim.write_mode_solver_config(tmp)
        job_id = gcloud.upload(tmp, "meep", verbose=False)
        gcloud.start(job_id, verbose=False)
        return job_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Submit jobs in batches so at most MAX_CONCURRENT run on the cloud at once,
# then wait for the whole batch concurrently. n_eff only (no field download).
sweep_results = []  # ModeSweepResult per width, in W0_VALUES order
for i in range(0, len(W0_VALUES), MAX_CONCURRENT):
    batch = W0_VALUES[i : i + MAX_CONCURRENT]
    job_ids = [submit_mode_solve(w0) for w0 in batch]
    print(f"batch {i // MAX_CONCURRENT + 1}: started {len(job_ids)} jobs")
    batch_results = gcloud.wait_for_results(job_ids, verbose="quiet")
    if not isinstance(batch_results, list):  # single-job batch returns one result
        batch_results = [batch_results]
    sweep_results.extend(batch_results)

# n_eff per band across the width sweep (NaN where the band is not guided).
neff_by_band = {b: [] for b in range(1, N_SWEEP_BANDS + 1)}
for w0, res in zip(W0_VALUES, sweep_results, strict=True):
    found = {r.band_num: r.n_eff for r in res.at(WAVELENGTH).results}
    for b in range(1, N_SWEEP_BANDS + 1):
        neff = found.get(b, np.nan)
        # drop radiative modes below the cladding index
        neff_by_band[b].append(neff if neff > n_clad else np.nan)
    guided = [b for b in range(1, N_SWEEP_BANDS + 1) if found.get(b, 0) > n_clad]
    summary = ", ".join(f"b{b}={found[b]:.4f}" for b in guided)
    print(f"w0={w0:.2f} µm -> {len(guided)} guided mode(s): {summary}")

# %% [markdown] papermill={"duration": 0.00381, "end_time": "2026-07-24T17:42:57.033543", "exception": false, "start_time": "2026-07-24T17:42:57.029733", "status": "completed"}
# ``n_eff`` vs ``w0`` for each band (interactive).

# %% papermill={"duration": 0.034951, "end_time": "2026-07-24T17:42:57.071777", "exception": false, "start_time": "2026-07-24T17:42:57.036826", "status": "completed"}
band_fig = go.Figure()
for b in range(1, N_SWEEP_BANDS + 1):
    band_fig.add_trace(
        go.Scatter(
            x=W0_VALUES,
            y=neff_by_band[b],
            mode="lines+markers",
            name=f"band {b}",
            connectgaps=False,
            hovertemplate="w0=%{x:.2f} µm<br>n_eff=%{y:.4f}<extra>band "
            + str(b)
            + "</extra>",
        )
    )
band_fig.add_hline(
    y=n_clad,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"SiO2 cladding (n={n_clad:.3f})",
    annotation_position="bottom right",
)
band_fig.update_layout(
    height=550,
    width=550,
    autosize=False,
    margin=dict(t=60, b=60),
    xaxis_title="ridge width w0 (µm)",
    yaxis_title="n_eff",
    title=dict(
        text=f"TFLN band diagram (lambda={WAVELENGTH:.2f} µm)",
        x=0.5,
        xanchor="center",
        font=dict(size=13),
    ),
    legend=dict(title=""),
)
# Fixed square aspect (width == height); no responsive stretch.
band_fig.show()
