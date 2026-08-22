#!/usr/bin/env python3
"""Analyze a run checkpoint (results/<tag>_seed<N>.ckpt.npz): internal metrics.

Prints: trajectory (best merit vs time), per-chain best_merit stats, phase
events (kicks/selects/screens), and per-chain ranking — the internal state
needed to diagnose plateaus, kick ROI, and chain-level dynamics.

Usage: python submission/analyze_ckpt.py results/w4_seed201.ckpt.npz [--chains]
"""
from __future__ import annotations

import sys

import numpy as np


def main():
    path = sys.argv[1]
    show_chains = "--chains" in sys.argv
    d = np.load(path, allow_pickle=True)

    step = int(d["step"][0])
    cum = float(d["cum_time"][0])
    bm = np.asarray(d["best_merit"])
    print(f"step={step} cum_time={cum:.0f}s chains={len(bm)}")
    fin = bm[np.isfinite(bm)]
    print(f"best_merit: min={fin.min():.4f} p25={np.percentile(fin,25):.4f} "
          f"med={np.median(fin):.4f} p75={np.percentile(fin,75):.4f} "
          f"n_inf={int((~np.isfinite(bm)).sum())}")

    traj = np.asarray(d["traj"]) if "traj" in d else np.zeros((0, 2))
    if len(traj):
        print("\ntrajectory (t_s, best):")
        # downsample to ~24 points
        idx = np.linspace(0, len(traj) - 1, min(24, len(traj))).astype(int)
        for i in idx:
            print(f"  {traj[i][0]:8.0f}s  {traj[i][1]:.5f}")

    phase = [str(x) for x in np.asarray(d["phase_log"])]
    ev = [l for l in phase if any(k in l for k in ("kick@", "select@", "screen", "demes", "es_stream"))]
    if ev:
        print("\nevents:")
        for l in ev[-30:]:
            print(" ", l[:120])

    ckp = d["chain_kick_prog"] if "chain_kick_prog" in d and len(d["chain_kick_prog"]) else None
    if show_chains:
        print("\nper-chain (rank, best_merit, last_kick_prog):")
        order = np.argsort(bm)
        for r, c in enumerate(order):
            kp = f"{ckp[c]:.2f}" if ckp is not None and ckp[c] > -1e8 else "-"
            tag = ""
            print(f"  #{r:2d} chain={c:3d} merit={bm[c]:10.5f} last_kick={kp} {tag}")


if __name__ == "__main__":
    main()
