#!/usr/bin/env python3
"""Canonicalize the Differometor-30k dataset into a fixed key space.

For every unique size-3 topology in the dataset, construct its UIFOProblem to
obtain the canonical (component, property) pair keys; build the global key
superset; then export every entry as (topology_id, canonical_param_vector,
present_mask, loss). Output: explore/canonical_index.npz

Run: python explore/build_canonical.py [--workers 48] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "explore" / "canonical_index.npz"


def pair_key(pair) -> str:
    return f"{repr(pair[0])}|{repr(pair[1])}"


def topo_keys(topology: str) -> list[str]:
    from dfbench.problems import UIFOProblem

    # n_frequencies=1: pair keys don't depend on freq count; ~2x faster tracing
    p = UIFOProblem(size=3, topology=topology, n_frequencies=1)
    return [pair_key(pr) for pr in p._optimization_pairs]


import json as _json


def _ck_path(t):
    return REPO / "explore" / "ckeys" / f"{t}.json"


def worker(topos_chunk):
    """Return {topology: keys}; write per-topology files for resume."""
    out = {}
    for t in topos_chunk:
        fp = _ck_path(t)
        if fp.exists():
            try:
                out[t] = _json.loads(fp.read_text())
                continue
            except Exception:
                pass
        try:
            ks = topo_keys(t)
            out[t] = ks
            fp.write_text(_json.dumps(ks))
        except Exception as e:  # noqa
            out[t] = None
            print(f"FAIL {t}: {e!r}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="shard index k of N (for multi-machine)")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    h5 = h5py.File(REPO / "dataset" / "dataset.h5", "r")
    entries = h5["entries"]
    losses = entries["loss"][:]
    topos = entries["topology_string"][:]
    topos = np.array([t.decode() if isinstance(t, bytes) else t for t in topos])
    sizes = entries["size"][:]
    keep = np.where(sizes == 3)[0]
    uniq = sorted(set(topos[keep].tolist()))
    # prioritize deeply-optimized topologies (many entries) — best surrogate data
    from collections import Counter

    _cnt = Counter(topos[keep].tolist())
    uniq.sort(key=lambda t: -_cnt[t])
    if args.limit:
        uniq = uniq[: args.limit]
    print(f"entries={len(keep)} unique_topologies={len(uniq)}")

    t0 = time.time()
    (REPO / "explore" / "ckeys").mkdir(parents=True, exist_ok=True)
    # skip already-done topologies (resume support)
    done = [t for t in uniq if _ck_path(t).exists()]
    todo = [t for t in uniq if not _ck_path(t).exists()]
    if args.nshards > 1:
        todo = todo[args.shard :: args.nshards]
    print(f"already done: {len(done)}, todo: {len(todo)} (shard {args.shard}/{args.nshards})")
    chunks = [todo[i :: args.workers] for i in range(args.workers) if todo[i :: args.workers]]
    with Pool(args.workers) as pool:
        results = pool.map(worker, chunks)
    topo2keys = {}
    for r in results:
        topo2keys.update(r)
    # reload everything from per-topology files (covers pre-done entries)
    for t in uniq:
        if t not in topo2keys or topo2keys.get(t) is None:
            fp = _ck_path(t)
            if fp.exists():
                try:
                    topo2keys[t] = _json.loads(fp.read_text())
                except Exception:
                    pass
    n_fail = sum(1 for v in topo2keys.values() if v is None)
    print(f"constructed {len(topo2keys) - n_fail}, failed {n_fail}, took {time.time()-t0:.0f}s")

    # global key superset (sorted for determinism)
    superset = sorted({k for ks in topo2keys.values() if ks for k in ks})
    kid = {k: i for i, k in enumerate(superset)}
    D = len(superset)
    print(f"superset dim D={D}")

    topo_list = sorted(topo2keys.keys())
    topo_index = {t: i for i, t in enumerate(topo_list)}
    # per-topology map into superset
    maps = np.full((len(topo_list), 400), -1, dtype=np.int32)  # 400 > max n_params
    map_len = np.zeros(len(topo_list), dtype=np.int32)
    for t in topo_list:
        ks = topo2keys[t]
        if ks is None:
            continue
        i = topo_index[t]
        idx = [kid[k] for k in ks]
        maps[i, : len(idx)] = idx
        map_len[i] = len(idx)

    # per-entry canonical vectors
    nE = len(keep)
    X = np.zeros((nE, D), dtype=np.float32)
    M = np.zeros((nE, D), dtype=bool)
    L = np.zeros(nE, dtype=np.float32)
    T = np.zeros(nE, dtype=np.int32)
    pool = h5["bounded_params"]
    for row, ei in enumerate(keep):
        e = entries[ei]
        t = topos[ei]
        ti = topo_index[t]
        lm = map_len[ti]
        if lm == 0:
            continue
        off, ln = int(e["param_offset"]), int(e["param_length"])
        if ln != lm:
            continue
        cols = maps[ti, :lm]
        X[row, cols] = pool[off : off + ln]
        M[row, cols] = True
        L[row] = losses[ei]
        T[row] = ti
        if row % 5000 == 0:
            print(f" entry {row}/{nE}", flush=True)

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
    print(f"saved {OUT} entries={nE} D={D}")


if __name__ == "__main__":
    main()
