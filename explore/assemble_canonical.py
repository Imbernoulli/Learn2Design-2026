#!/usr/bin/env python3
"""Assemble explore/canonical_index.npz from per-topology ckeys/<topo>.json files.

Works with ANY subset (partial index). Only dataset entries whose topology has
a ckeys file are included. Output fields:
 keys (superset), topo_list, X (canonical params), M (mask), L (loss), T (topo id)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CKEYS = REPO / "explore" / "ckeys"
OUT = REPO / "explore" / "canonical_index.npz"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = sorted(CKEYS.glob("*.json"))
    if limit:
        files = files[:limit]
    print(f"ckeys files: {len(files)}")
    topo2keys = {}
    for fp in files:
        t = fp.stem
        if "-" not in t or len(t.split("-")[0]) != 9:
            continue
        try:
            topo2keys[t] = json.loads(fp.read_text())
        except Exception:
            pass
    print(f"loaded {len(topo2keys)} topology key-lists")

    superset = sorted({k for ks in topo2keys.values() for k in ks})
    kid = {k: i for i, k in enumerate(superset)}
    D = len(superset)
    print(f"superset D={D}")

    topo_list = sorted(topo2keys.keys())
    t2i = {t: i for i, t in enumerate(topo_list)}

    h5 = h5py.File(REPO / "dataset" / "dataset.h5", "r")
    entries = h5["entries"]
    topos = entries["topology_string"][:]
    topos = np.array([t.decode() if isinstance(t, bytes) else t for t in topos])
    sizes = entries["size"][:]
    losses = entries["loss"][:]
    pool = h5["bounded_params"]

    keep_rows = []
    for ei in np.where(sizes == 3)[0]:
        if topos[ei] in t2i:
            keep_rows.append(int(ei))
    print(f"entries covered: {len(keep_rows)}")

    X = np.zeros((len(keep_rows), D), dtype=np.float32)
    M = np.zeros((len(keep_rows), D), dtype=bool)
    L = np.zeros(len(keep_rows), dtype=np.float32)
    T = np.zeros(len(keep_rows), dtype=np.int32)
    t0 = time.time()
    for row, ei in enumerate(keep_rows):
        t = topos[ei]
        ks = topo2keys[t]
        e = entries[ei]
        off, ln = int(e["param_offset"]), int(e["param_length"])
        if ln != len(ks):
            continue
        cols = np.array([kid[k] for k in ks])
        X[row, cols] = pool[off : off + ln]
        M[row, cols] = True
        L[row] = losses[ei]
        T[row] = t2i[t]
        if row % 5000 == 0:
            print(f" {row}/{len(keep_rows)} ({time.time()-t0:.0f}s)", flush=True)

    maps = np.full((len(topo_list), 400), -1, dtype=np.int32)
    map_len = np.zeros(len(topo_list), dtype=np.int32)
    for t, ks in topo2keys.items():
        i = t2i[t]
        idx = [kid[k] for k in ks]
        maps[i, : len(idx)] = idx
        map_len[i] = len(idx)
    np.savez_compressed(
        OUT,
        keys=np.array(superset),
        topo_list=np.array(topo_list),
        maps=maps,
        map_len=map_len,
        X=X,
        M=M,
        L=L,
        T=T,
    )
    print(f"saved {OUT}: {X.shape} entries, D={D}")


if __name__ == "__main__":
    main()
