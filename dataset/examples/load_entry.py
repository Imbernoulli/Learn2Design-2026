#!/usr/bin/env python3
"""Load one UIFO dataset entry and print its saved arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from dataset_utils import DATASET_PATH, entry_to_dict, find_entry_index, load_bounded_params, load_power_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--index", type=int, default=0, help="Entry index to load.")
    parser.add_argument("--hash", dest="unique_hash", help="Entry unique_hash. Overrides --index.")
    args = parser.parse_args()

    index_arg = None if args.unique_hash else args.index

    with h5py.File(args.dataset, "r") as handle:
        entry_index = find_entry_index(handle, index=index_arg, unique_hash=args.unique_hash)
        entry = handle["entries"][entry_index]
        metadata = entry_to_dict(entry)
        params = load_bounded_params(handle, entry)
        sensitivities = handle["sensitivities"][entry_index]
        frequencies = handle["frequency_values"][:]
        power, power_port_names, power_port_indices = load_power_data(handle, entry)
        named_power = power[power_port_indices]

    print(f"entry_index: {entry_index}")
    for key in (
        "unique_hash",
        "initialized_from",
        "topology_string",
        "size",
        "loss",
        "complexity",
        "source_file",
        "run_id",
    ):
        print(f"{key}: {metadata[key]}")

    print(f"bounded_params shape: {params.shape}")
    print(f"bounded_params first 5: {params[:5]}")
    print(f"frequency_values shape: {frequencies.shape}")
    print(f"frequency_values first 5: {frequencies[:5]}")
    print(f"sensitivities shape: {sensitivities.shape}")
    print(f"sensitivities first 5: {sensitivities[:5]}")
    print(f"power shape: {power.shape}")
    print(f"power first row first 5: {power[0, :5]}")
    print(f"power port count: {len(power_port_names)}")
    print(f"power port labels first 5: {power_port_names[:5]}")
    print(f"power port indices first 5: {power_port_indices[:5]}")
    print(f"named power first 5 rows: {named_power[:5].ravel()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
