#!/usr/bin/env python3
"""Visualize one UIFO dataset entry with Differometor."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

try:
    from dfbench.problems.uifo import topology_from_string
    from differometor import visualize_setup
    from differometor.setups import constrain_inter_grid_cell_spaces, uifo
except ModuleNotFoundError as exc:
    raise SystemExit("Install the competition package first: pip install -e .") from exc

from dataset_utils import DATASET_PATH, entry_to_dict, find_entry_index, load_bounded_params, load_power_data


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "visualizations"
DEFAULT_MODE = "space_modulation"
OPTIMIZED_PROPERTIES = [
    "reflectivity",
    "tuning",
    "db",
    "angle",
    "power",
    "mass",
    "length",
]


def resolve_output_path(output: Path | None, *, entry_index: int, unique_hash: str) -> Path:
    if output is None:
        output = DEFAULT_OUTPUT_DIR / f"entry_{entry_index:06d}_{unique_hash[:8]}.html"
    else:
        output = output.expanduser()

    if not str(output).lower().endswith(".html"):
        output = Path(f"{output}.html")
    return output


def port_map_to_index(port_names: list[str], port_indices) -> dict[str, int]:
    return {name: int(index) for name, index in zip(port_names, port_indices)}


def iter_parameter_pairs(optimization_pair):
    if optimization_pair and isinstance(optimization_pair[0], list):
        yield from optimization_pair
    else:
        yield optimization_pair


def apply_bounded_params(setup, component_property_pairs, bounded_params) -> None:
    optimization_pairs = constrain_inter_grid_cell_spaces(component_property_pairs, OPTIMIZED_PROPERTIES)
    if len(bounded_params) != len(optimization_pairs):
        raise ValueError(
            f"Saved params have length {len(bounded_params)}, but the reconstructed setup expects "
            f"{len(optimization_pairs)} optimization parameters."
        )

    for value, optimization_pair in zip(bounded_params, optimization_pairs):
        for component_name, property_name in iter_parameter_pairs(optimization_pair):
            if "_" in component_name:
                setup.edges[component_name]["properties"][property_name] = float(value)
            else:
                setup.nodes[component_name]["properties"][property_name] = float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--index", type=int, default=0, help="Entry index to visualize.")
    parser.add_argument("--hash", dest="unique_hash", help="Entry unique_hash. Overrides --index.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file. Defaults to dataset/examples/visualizations/.",
    )
    args = parser.parse_args()

    index_arg = None if args.unique_hash else args.index

    with h5py.File(args.dataset, "r") as handle:
        entry_index = find_entry_index(handle, index=index_arg, unique_hash=args.unique_hash)
        entry = handle["entries"][entry_index]
        metadata = entry_to_dict(entry)
        bounded_params = load_bounded_params(handle, entry)
        power, power_port_names, power_port_indices = load_power_data(handle, entry)

    centers, boundaries = topology_from_string(metadata["topology_string"], metadata["size"])
    # Match UIFOProblem construction. Dataset topologies are fully specified, so
    # random=True does not change the decoded centers or boundaries.
    setup, _ = uifo(
        size=metadata["size"],
        mode=DEFAULT_MODE,
        random=True,
        centers=centers,
        boundaries=boundaries,
    )
    apply_bounded_params(setup, setup.parameters, bounded_params)

    output_path = resolve_output_path(
        args.output,
        entry_index=entry_index,
        unique_hash=metadata["unique_hash"],
    )
    visualize_setup(
        setup,
        output_file=output_path,
        port_to_index=port_map_to_index(power_port_names, power_port_indices),
        powers=power,
    )

    print(f"entry_index: {entry_index}")
    print(f"unique_hash: {metadata['unique_hash']}")
    print(f"topology_string: {metadata['topology_string']}")
    print(f"size: {metadata['size']}")
    print(f"output: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
