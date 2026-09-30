#!/usr/bin/env bash

set -euo pipefail

remote='cd ~/Projects/MachineLearning/skyrl-rwkv && '
for name in MODEL_DIR DATA_DIR NUM_GPUS MICRO_BATCH_SIZE LOGGER RUN_NAME OUTPUT_ROOT; do
  if [[ ${!name+x} ]]; then
    printf -v value '%q' "${!name}"
    remote+="$name=$value "
  fi
done
remote+='exec bash examples/train/rwkv/run_rwkv_grpo.sh'
for arg in "$@"; do
  printf -v quoted '%q' "$arg"
  remote+=" $quoted"
done
exec ssh rwkv-sha-pro6000x8 "$remote"
