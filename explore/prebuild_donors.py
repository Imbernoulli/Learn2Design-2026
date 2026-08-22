#!/usr/bin/env python3
"""Pre-build ckeys for the warm-start donors of specific eval seeds.

For each seed: construct the target UIFOProblem (topology_seed), find its
nearest donor topologies via the submission's WarmStartIndex, and construct
ckeys/<topo>.json for any donor not yet covered. This makes the warm-start
key cache nearly complete for those targets even before the full canonical
build finishes.

Run: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python explore/prebuild_donors.py 107 108 ...
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submission"))

CKEYS = REPO / "explore" / "ckeys"


def _ck_path(t):
    return CKEYS / f"{t}.json"


def topo_keys(topology: str) -> list[str]:
    from dfbench.problems import UIFOProblem

    p = UIFOProblem(size=3, topology=topology, n_frequencies=1)
    return [f"{repr(pr[0])}|{repr(pr[1])}" for pr in p._optimization_pairs]


def worker(topos):
    out = 0
    for t in topos:
        fp = _ck_path(t)
        if fp.exists():
            continue
        try:
            fp.write_text(json.dumps(topo_keys(t)))
            out += 1
        except Exception as e:  # noqa
            print(f"FAIL {t}: {e!r}", flush=True)
    return out


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    workers = 48
    t0 = time.time()

    # 1) target topology strings for the seeds (CPU construction, one each)
    from dfbench.problems import UIFOProblem

    targets = {}
    for s in seeds:
        targets[s] = UIFOProblem(size=3, topology_seed=s, n_frequencies=1).topology_string
        print(f"seed {s}: {targets[s]}", flush=True)

    # 2) nearest donors per target (dataset index, no objective needed)
    from parallel_adam_submission import WarmStartIndex

    ws = WarmStartIndex(REPO / "dataset" / "dataset.h5", size=None)
    donor_topos: set[str] = set()
    for s, tstr in targets.items():
        # generous k: covers n_transplant=96 init + kick_donor_pool=64 cycles
        donors = ws.nearest_donors(tstr, k=256, loss_weight=0.5)
        donor_topos.update(d[0] for d in donors)
    print(f"unique donor topologies: {len(donor_topos)}", flush=True)

    # 3) build missing ckeys
    todo = [t for t in sorted(donor_topos) if not _ck_path(t).exists()]
    print(f"missing ckeys: {len(todo)} (of {len(donor_topos)})", flush=True)
    if todo:
        chunks = [todo[i::workers] for i in range(workers) if todo[i::workers]]
        with Pool(workers) as pool:
            n = sum(pool.map(worker, chunks))
        print(f"built {n} donor ckeys in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
