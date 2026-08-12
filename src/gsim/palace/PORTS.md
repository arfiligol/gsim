# Palace Port Types Explained

This document explains how ports work in Palace and how to define them for different transmission line configurations.

## Port Types Overview

### In-Plane Port

- **Orientation**: Horizontal surface in the XY plane
- **Location**: Sits on a **single** metal layer
- **Definition**: `target_layername='TopMetal2'`
- **GDS shape**: Rectangle with finite width and length

```text
      ════════════════  ← metal layer
         ▓▓▓▓▓▓▓▓       ← port surface (horizontal)
```

### Via Port

- **Orientation**: Vertical surface in the XZ or YZ plane
- **Location**: Spans **between two** metal layers
- **Definition**: `from_layername='Metal3', to_layername='TopMetal2'`
- **GDS shape**: Line (degenerate rectangle) - one dimension is minimal

```text
      ════════════════  ← TopMetal2
            ║
            ║  ← port surface (vertical, like a wall)
            ║
      ════════════════  ← Metal3
```

Note: "Via port" does NOT mean a port on via structures. It's named this way because it spans the vertical gap between
layers, similar to how a via connects layers.

## How Palace Lumped Ports Work

A lumped port is a **2D surface** with:

- `R` = impedance (typically 50Ω)
- `Direction` = positive solver field/polarization direction as a unit Cartesian vector

Palace integrates the E-field across this surface to compute voltage. The port acts as a lumped resistor connected
between whatever conductors touch the port surface.

`gsim` accepts legacy Manhattan labels such as `X`, `+X`, and `-Y` on `add_port(direction=...)`, but normalizes them to
vectors such as `[1.0, 0.0, 0.0]` before writing Palace config. Arbitrary finite nonzero 3-vectors are accepted and
normalized. Radial labels (`+R`, `-R`) are not valid for LumpedPort v1; they remain specific to current-source models.

`Direction` is solver intent only. Generated port-sheet geometry is determined from the GDSFactory port `center`,
`width`, `orientation`, and layer. Changing `direction` does not rotate or resize the generated sheet.

## Port Sheet Geometry Sources

`gsim` supports two horizontal sheet sources for in-plane LumpedPorts and CPW LumpedPort elements:

1. Generated sheets: `generate_sheet=True` (default). Mesh generation creates the solver sheet from the GDSFactory port
   anchor.
1. Layout-authored sheets: `generate_sheet=False`. Mesh generation selects an existing polygon from a PDK-declared
   simulation layer catalog.

The high-level port declaration stays the same in both cases. `add_port()` and `add_cpw_port()` declare solver port
intent; `set_simulation_layers()` supplies PDK-owned simulation-only layer meaning; mesh generation owns selecting or
creating the final solver boundary surface. Config generation only receives the resulting physical group.

### Layout-Authored Horizontal Sheets

Use layout-authored sheets when the port sheet is too geometry-specific for a generic rectangle. The PDK/project passes
a catalog:

```python
sim.set_simulation_layers(
    {
        "D0_TOP_SIM_BOUNDARY": {
            "gds_layer": (202, 1),
            "stack_layer": "D0_TOP_SIM_BOUNDARY",
        }
    }
)
```

Then the component port used by `add_port()` must live on that simulation layer. Passing `generate_sheet=False` tells
mesh generation to select the unique polygon on that layer that covers the port center:

```python
sim.add_port(
    "o1",
    layer="topmetal2",
    length=5.0,
    generate_sheet=False,
)
```

The `layer` argument remains the target stack/material layer. The authored sheet layer comes from
`component.ports["o1"].layer` and must be registered in the simulation layer catalog. If no catalog is set, the port
layer is not in the catalog, or the port center matches zero or multiple polygons, meshing fails.

Via ports remain generated vertical sheets because their geometry spans between `from_layer` and `to_layer` in z rather
than selecting a horizontal GDS polygon.

## Generated Port Geometry

### In-Plane Ports: Oriented Rectangle

In-plane lumped ports generate a rectangular sheet in the XY plane from the GDSFactory port metadata:

```python
center = port.center
width = port.width
orientation = port.orientation
length = palace_port.length
```

The sheet length follows `orientation`; the sheet width is transverse to `orientation`. This supports non-Manhattan
GDSFactory ports without coupling sheet rotation to the solver `Direction` vector.

### Via Ports: Line Expected

Via ports expect essentially a line in the GDS:

- The GDSFactory port orientation determines whether the vertical sheet spans XZ or YZ
- The port surface is created vertically between the two layers

## Transmission Line Configurations

### Case 1: Microstrip (Ground on Different Layer)

Use a **via port** spanning from signal layer to ground layer:

```text
TopMetal2 (signal) ════════════
                        ║ ← via port surface (vertical)
Metal1 (ground)    ════════════
```

Configuration:

```python
configure_port(
    port,
    type="via",
    from_layer="metal1",      # ground layer
    to_layer="topmetal2",     # signal layer
    impedance=50.0,
)
```

### Case 2: CPW (Signal/Ground on Same Layer)

Use an **in-plane port** spanning the gap between signal and ground:

```text
Ground ═══╡    ║    ╞═══ Signal ═══╡    ║    ╞═══ Ground
          └────┘                   └────┘
          port 1                   port 2
          (in gap)                 (in gap)
```

The port rectangle spans **from signal edge to ground edge** (across the gap).

Configuration:

```python
configure_port(
    port,
    type="lumped",
    layer="topmetal2",  # layer where CPW lives
    length=5.0,         # port extent along direction
    impedance=50.0,
)
```

## Converting gdsfactory Ports to Palace Format

gdsfactory ports have:

- `center`: (x, y) position
- `width`: port width (e.g., CPW signal + gaps)
- `orientation`: angle in degrees (0=east, 90=north, 180=west, 270=south)

Using the high-level `gsim` simulation API:

```python
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_geometry(c)
sim.set_stack()
sim.add_port("o1", layer="topmetal2", length=5.0, direction="+X")
sim.add_port("o2", layer="topmetal2", length=5.0, direction=[-1.0, 0.0, 0.0])
```

## Why Ports Must Touch Both Signal and Ground

A lumped port is essentially a **virtual VNA probe**. To compute S-parameters, Palace needs:

1. **Voltage** = potential difference between two conductors
1. **Current** = flow between those conductors

Palace computes these by:

- **Voltage**: Integrating E-field across the port surface (from one conductor to the other)
- **Current**: Integrating H-field around the port boundary

If the port only touches one conductor, voltage is undefined - there's no second reference point.

```text
                Port touching both (CORRECT):

Ground ═══════╡▓▓▓▓▓▓╞═══════ Signal
              ↑     ↑
              └──┬──┘
           E-field integrated -> Voltage


                Port touching only signal (WRONG):

Ground ═══════      ▓▓▓▓▓▓═══════ Signal
                    ↑
                    No reference -> Voltage undefined
```

### Real-World Analogy

When probing a CPW with a VNA:

- Signal pin touches the center conductor
- Ground pins touch the ground planes
- Measurement happens **across the gap**

The port rectangle represents exactly this: the cross-section where your virtual probe connects signal to ground.

## Multi-Element CPW Ports

For proper CPW mode excitation, use `add_cpw_port()` with one GDSFactory port at the signal center:

```python
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_geometry(c)
sim.add_cpw_port("o1", layer="topmetal2", s_width=10.0, gap_width=6.0, length=5.0)
```

This generates a multi-element lumped port in Palace:

```json
{
  "Index": 1,
  "R": 50.0,
  "Elements": [
    {"Attributes": [gap1_surface], "Direction": [0.0, -1.0, 0.0]},
    {"Attributes": [gap2_surface], "Direction": [0.0, 1.0, 0.0]}
  ]
}
```

## Key Points

1. **One port = one rectangle** defining the port surface
1. **Port must touch both signal and ground** - it's the bridge between them
1. **Direction** tells Palace the solver field/polarization direction and is emitted as a normalized Cartesian vector
1. **In-plane ports** need actual area (rectangle), not just a line
1. **Via ports** are for vertical connections between layers
1. **CPW ports** need two elements with opposite transverse directions derived from the GDSFactory port orientation
