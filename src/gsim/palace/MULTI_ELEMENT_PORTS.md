# Multi-Element Lumped Ports for CPW

**Status**: Implemented

## Problem

Coplanar waveguide (CPW) structures require multi-element lumped ports to properly excite the CPW mode. Single-element
ports treat each gap as independent and result in incorrect S-parameters (no transmission).

## Background

In a CPW (Ground-Signal-Ground), the E-field directions are opposite in the two gaps:

```text
Ground 2  ═══════════════════
            ↓ E-field (-Y)     Gap2
Signal    ═══════════════════
            ↑ E-field (+Y)     Gap1
Ground 1  ═══════════════════
```

Palace supports this via multi-element ports:

```json
{
  "Index": 1,
  "R": 56.02,
  "Elements": [
    {"Attributes": [gap1_surface], "Direction": [0.0, -1.0, 0.0]},
    {"Attributes": [gap2_surface], "Direction": [0.0, 1.0, 0.0]}
  ]
}
```

`gsim` emits normalized Cartesian vectors for `Direction`. The two CPW element
directions are opposite transverse vectors derived from the GDSFactory port
orientation.

## Reference

- [Palace CPW Example](https://awslabs.github.io/palace/stable/examples/cpw/)
- [Palace CPW config](https://github.com/awslabs/palace/blob/main/examples/cpw/cpw_lumped_uniform.json)

## Implementation

### API Usage

```python
from gsim.palace import DrivenSim

sim = DrivenSim()
sim.set_geometry(c)
sim.set_stack()
sim.add_cpw_port(
    "o1",
    layer="topmetal2",
    s_width=10.0,
    gap_width=6.0,
    length=5.0,
    impedance=50.0,
)
```

If the two gap sheets are already drawn on a PDK-declared simulation layer,
register that catalog and disable generated sheets:

```python
sim.set_simulation_layers(
    {"top_sim_boundary": {"gds_layer": (202, 1), "stack_layer": "top_sim_boundary"}}
)
sim.add_cpw_port(
    "o1",
    layer="topmetal2",
    s_width=10.0,
    gap_width=6.0,
    length=5.0,
    generate_sheet=False,
)
```

The GDSFactory port named `"o1"` remains the signal-center anchor. Its layer
must be the registered simulation layer, and mesh generation selects exactly one
authored polygon at each computed gap center.

### How It Works

1. `add_cpw_port()` selects one GDSFactory port at the signal center

   - Stores `palace_type='cpw'` in port info during mesh preparation
   - Computes the two gap centers from signal width, gap width, and port
     orientation

1. `extract_ports()` lowers the configured port into an internal
   `models.PalacePort`

   - Each `models.PalacePort` has two gap centers
   - Element `Direction` values are opposite normalized transverse vectors

1. Mesh generator creates separate surfaces for each element

   - Physical groups: `P1_E0`, `P1_E1` for port 1 elements
   - Generated rectangles are rotated from the GDSFactory port orientation, or
     layout-authored polygons are selected from registered simulation layers

1. Config generator outputs multi-element format:

```json
{
  "Index": 1,
  "R": 50.0,
  "Excitation": 1,
  "Elements": [
    {"Attributes": [phys_group_E0], "Direction": [0.0, -1.0, 0.0]},
    {"Attributes": [phys_group_E1], "Direction": [0.0, 1.0, 0.0]}
  ]
}
```

## Files

- `models/ports.py` - `PalacePort`, `PortType`, and `PortGeometry` contracts
- `ports/lowering.py` - `configure_cpw_port()` and `extract_ports()`
- `mesh/geometry.py` - `add_ports()` handles rotated CPW element surfaces
- `mesh/config_generator.py` - emits the Palace `Elements` array with vector
  directions
