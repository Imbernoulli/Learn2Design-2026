#!/usr/bin/env python3
"""Run one ParallelAdam-submission evaluation on one UIFO topology.

Usage:
  CUDA_VISIBLE_DEVICES=0 python submission/eval_one.py --seed 101 --max-time 14400 \
      --out results/seed101_pa.json [--kwargs '{}']

Score = obj.best_loss (best feasible loss after start_logging), matching
docs/scoring.md. Writes a JSON with best loss, phase log, and wall time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="UIFOProblem topology_seed")
    ap.add_argument("--max-time", type=float, default=4 * 3600)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--kwargs", type=str, default="{}", help="JSON overrides for the algorithm")
    ap.add_argument("--baseline", type=str, default="", help="run an example baseline instead (e.g. adam_gd)")
    ap.add_argument("--ckpt-interval", type=float, default=120.0, help="chain-state checkpoint interval (s); 0 disables")
    args = ap.parse_args()

    from dfbench.problems import UIFOProblem
    from dfbench import Objective

    problem = UIFOProblem(topology_seed=args.seed)
    obj = Objective(
        problem,
        verbose=0,
        max_time=args.max_time,
        save_params_history=False,
        display_mode="log",
    )

    t0 = time.perf_counter()
    phase_log = []
    algo_best = None
    err = None
    try:
        if args.baseline:
            from learn2design.example_algorithms import AdamGD

            opt = AdamGD()
            opt.optimize(obj, learning_rate=0.1, random_seed=args.seed)
            best = obj.best_loss
        else:
            from submission.parallel_adam_submission import ParallelAdamSubmission

            kw = json.loads(args.kwargs)
            algo = ParallelAdamSubmission(**kw)
            # harness-level instrumentation (no algorithm change): periodic
            # full chain-state checkpoints -> internal metrics for analysis
            if args.ckpt_interval > 0:
                algo.set_checkpointing(args.out.replace(".json", ".ckpt.npz"),
                                       interval_s=args.ckpt_interval)
            algo.optimize(obj, random_seed=args.seed)
            phase_log = list(algo.phase_log)
            algo_best = algo.best_feasible_loss_algo
            best = obj.best_loss
            # algo's own feasible tracker should agree with obj.best_loss
            phase_log.append(f"algo.best_feasible={algo.best_feasible_loss_algo:.6f}")
    except Exception:
        import traceback

        err = traceback.format_exc()
        best = None
    if best is None:
        # no feasible eval logged (or crash): fall back to the algo tracker
        best = algo_best if algo_best is not None else float("inf")
    best = float(best)

    wall = time.perf_counter() - t0
    out = {
        "seed": args.seed,
        "topology": problem.topology_string,
        "best_loss": best,
        "wall_s": wall,
        "max_time": args.max_time,
        "kwargs": json.loads(args.kwargs),
        "baseline": args.baseline or "parallel_adam_submission",
        "phase_log": phase_log,
        "error": err,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for line in phase_log:
        print("PHASE:", line, flush=True)
    if err:
        print(err, flush=True)
    print(f"DONE seed={args.seed} best_loss={best:.5f} wall={wall:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
