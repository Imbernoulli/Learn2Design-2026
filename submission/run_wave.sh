#!/usr/bin/env bash
# Launch a wave of submission evals, one per free GPU.
# Usage: bash submission/run_wave.sh <tag> <max_time_s> <seed1> [seed2 ...]
# GPU i gets seed_i. Results -> results/<tag>_seed<seed>.json, logs -> results/<tag>_seed<seed>.log
set -u
TAG=$1; MAXT=$2; shift 2
SEEDS=("$@")
GPUS=(0 1 2 3 4 5)
cd "$(dirname "$0")/.."
mkdir -p results
# compile-speed flags: parallel LLVM codegen (single-threaded LLVM was the
# bottleneck at ~20% CPU on this loaded box) + skip remat via higher mem cap
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
for i in "${!SEEDS[@]}"; do
  GPU=${GPUS[$i]}
  SEED=${SEEDS[$i]}
  OUT="results/${TAG}_seed${SEED}.json"
  LOG="results/${TAG}_seed${SEED}.log"
  if [[ -f "$OUT" ]]; then echo "skip seed $SEED (exists)"; continue; fi
  echo "launch seed=$SEED gpu=$GPU max_time=$MAXT -> $LOG"
  CUDA_VISIBLE_DEVICES=$GPU JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache \
    nohup python submission/eval_one.py --seed "$SEED" --max-time "$MAXT" \
    --out "$OUT" ${EXTRA_ARGS:-} > "$LOG" 2>&1 &
done
wait
echo "wave $TAG done"
