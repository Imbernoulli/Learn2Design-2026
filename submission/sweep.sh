#!/usr/bin/env bash
# Hyperparameter sweep driver: run one config on N seeds, one GPU each.
# Usage: bash submission/sweep.sh <tag> <max_time_s> '<kwargs_json>' seed1 [seed2 ...]
# Example: bash submission/sweep.sh polyak 3600 '{"polyak_ema":0.05}' 201 202 203
set -u
TAG=$1; MAXT=$2; KW=$3; shift 3
SEEDS=("$@")
GPUS=(0 1 2 3 4 5)
cd "$(dirname "$0")/.."
mkdir -p results
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
PIDS=()
for i in "${!SEEDS[@]}"; do
  GPU=${GPUS[$i]}
  SEED=${SEEDS[$i]}
  OUT="results/${TAG}_seed${SEED}.json"
  LOG="results/${TAG}_seed${SEED}.log"
  if [[ -f "$OUT" ]]; then echo "skip seed $SEED (exists)"; continue; fi
  echo "launch tag=$TAG seed=$SEED gpu=$GPU kw=$KW"
  CUDA_VISIBLE_DEVICES=$GPU JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache \
    nohup python submission/eval_one.py --seed "$SEED" --max-time "$MAXT" \
    --out "$OUT" --kwargs "$KW" > "$LOG" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && wait "$p"; done
echo "sweep $TAG done"
