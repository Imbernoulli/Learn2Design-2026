#!/usr/bin/env bash
# Variant sweep: one GPU per variant, same seed+budget, different kwargs.
# Usage: bash submission/variants.sh <seed> <max_time> '<json array of [tag, kwargs]>'
# Example: bash submission/variants.sh 104 3600 '[["ctrl",{}],["cf8",{"composite_fill":8}]]'
set -u
SEED=$1; MAXT=$2; SPECS=$3
cd "$(dirname "$0")/.."
mkdir -p results
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache
GPUS=(${GPU_LIST:-0 1 2 3 4 5 6 7})
echo "$SPECS" | python -c "
import json,sys
for i,(tag,kw) in enumerate(json.load(sys.stdin)):
    print(i, tag, json.dumps(kw))
" | while read -r i tag kw; do
  GPU=${GPUS[$i]}
  OUT="results/v_${tag}_s${SEED}.json"
  [[ -f "$OUT" ]] && { echo "skip $tag"; continue; }
  echo "launch $tag gpu=$GPU seed=$SEED kw=$kw"
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python submission/eval_one.py \
    --seed "$SEED" --max-time "$MAXT" --out "$OUT" --kwargs "$kw" \
    > "results/v_${tag}_s${SEED}.log" 2>&1 < /dev/null &
  sleep 2
done
