# UIFO Design Dataset

This folder contains the precomputed pure-broadband UIFO design corpus:

- `dataset.h5`: 29,650 optimized UIFO setups.
- `dataset_dashboard.html`: interactive Plotly loss visualization.
- `examples/load_entry.py`: loads one entry lazily from the HDF5 pools.
- `examples/evaluate_entry.py`: evaluates a saved entry with the competition `UIFOProblem`.

The saved `loss` values were computed for broadband sensitivity using 50 frequency
points. To reproduce a saved loss with the competition code, construct the problem
with the entry's `size`, `topology_string`, and `n_frequencies=50`, then evaluate
the saved `bounded_params`.

## Interactive Visualization

Drag `dataset/dataset_dashboard.html` into a browser to get a quick feel for the
UIFO loss distribution and dataset contents.

The HTML is standalone and includes Plotly, so it does not require a server or
internet access. It contains:

- a loss swarmplot colored by run group (`original`, `reoptimized`, and size-4 setups),
- a second loss swarmplot colored by topology reuse group,
- a loss-threshold slider that updates the number of setups below the threshold,
- dashed reference lines at `0` and the random UIFO baseline,
- clickable legends for showing, hiding, or isolating groups.

## HDF5 Layout

`entries` is a structured table with one row per setup. Important fields:

| Item | Meaning |
| --- | --- |
| `unique_hash` | Stable identifier for this setup. |
| `initialized_from` | Parent `unique_hash` when present, else empty. |
| `topology_string` | Compact topology string accepted by `UIFOProblem(topology=...)`. |
| `param_offset`, `param_length` | Slice into the flat `bounded_params` pool. |
| `loss` | Saved broadband loss. |
| `sensitivities` | Separate row-aligned `(n_entries, 50)` sensitivity dataset. |
| `power_offset`, `power_length`, `power_rows`, `power_cols` | Slice and reshape metadata for `power_values`. |
| `power_map_id` | Selects named port indices for rows in the reshaped power array. |
| `size`, `complexity` | UIFO grid size and topology complexity. |

Large variable-length arrays are stored in flat pools. Slice only the row you need:

```python
params = h5["bounded_params"][entry["param_offset"]:entry["param_offset"] + entry["param_length"]]
power = h5["power_values"][entry["power_offset"]:entry["power_offset"] + entry["power_length"]]
power = power.reshape(entry["power_rows"], entry["power_cols"])
sensitivities = h5["sensitivities"][entry_index]
frequencies = h5["frequency_values"][:]
```

Power port names are stored separately because different topologies have different
named ports:

```python
map_id = entry["power_map_id"]
start = h5["power_port_map_offsets"][map_id]
stop = h5["power_port_map_offsets"][map_id + 1]
port_row_indices = h5["power_port_map_indices"][start:stop]
port_names = h5["power_port_map_names"][start:stop]
named_power = power[port_row_indices]
```

## Quick Start

Install the competition package and HDF5 dependency:

```bash
pip install -e .
pip install h5py
```

Inspect one saved setup:

```bash
python dataset/examples/load_entry.py --index 0
```

Evaluate one saved setup with the competition objective:

```bash
python dataset/examples/evaluate_entry.py --index 0
```

You can also select by hash:

```bash
python dataset/examples/evaluate_entry.py --hash <unique_hash>
```

The first evaluation may take a few minutes because JAX compiles the UIFO objective.
