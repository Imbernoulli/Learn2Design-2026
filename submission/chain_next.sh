#!/usr/bin/env bash
# Chain launcher: as each wave-1 run finishes (JSON appears), start the next
# seed on the freed GPU. Runs detached via setsid.
# Usage: setsid nohup bash submission/chain_next.sh > results/chain_next.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache

declare -A NEXT=( [101,0]=109 [102,1]=110 [103,2]=111 [104,3]=112 [105,4]=113 [106,5]=114 )
# GPU 6,7 already run w2 seeds 107,108; when they finish, queue more
declare -A NEXT2=( [107,6]=115 [108,7]=116 )

launch() { # gpu seed tag
  local GPU=$1 SEED=$2 TAG=$3
  echo "$(date +%H:%M) launch seed=$SEED gpu=$GPU tag=$TAG"
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python submission/eval_one.py \
    --seed "$SEED" --max-time 14400 --out "results/${TAG}_seed${SEED}.json" \
    > "results/${TAG}_seed${SEED}.log" 2>&1 < /dev/null &
}

while :; do
  alldone=1
  for key in "${!NEXT[@]}"; do
    SEED=${key%,*}; GPU=${key#*,}
    if [[ -f "results/w1_seed${SEED}.json" ]]; then
      NSEED=${NEXT[$key]}
      [[ -n "$NSEED" ]] && launch "$GPU" "$NSEED" w3
      unset 'NEXT[$key]'
    else
      alldone=0
    fi
  done
  for key in "${!NEXT2[@]}"; do
    SEED=${key%,*}; GPU=${key#*,}
    if [[ -f "results/w2_seed${SEED}.json" ]]; then
      NSEED=${NEXT2[$key]}
      [[ -n "$NSEED" ]] && launch "$GPU" "$NSEED" w3
      unset 'NEXT2[$key]'
    fi
  done
  ((${#NEXT[@]} == 0 && ${#NEXT2[@]} == 0)) && { echo "all chained"; break; }
  sleep 120
done
