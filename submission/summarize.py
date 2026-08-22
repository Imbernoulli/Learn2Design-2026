#!/usr/bin/env python3
"""Summarize result JSONs: mean best_loss per tag.

Usage: python submission/summarize.py [results_dir] [pattern]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "results"
pat = sys.argv[2] if len(sys.argv) > 2 else "*.json"

by_tag = defaultdict(list)
for fp in sorted(glob.glob(f"{d}/{pat}")):
    try:
        r = json.load(open(fp))
    except Exception:
        continue
    tag = fp.split("/")[-1].split("_seed")[0]
    by_tag[tag].append((r["seed"], r["best_loss"], r.get("wall_s", 0)))

for tag, rows in sorted(by_tag.items()):
    losses = [x[1] for x in rows]
    seeds = [x[0] for x in rows]
    m, s = np.mean(losses), np.std(losses) / max(1, len(losses)) ** 0.5
    print(f"{tag:24s} n={len(rows)} mean={m:8.4f} sem={s:7.4f}  seeds={seeds}")
    for seed, loss, wall in rows:
        print(f"    seed={seed:5d} loss={loss:9.5f} wall={wall:7.0f}s")
