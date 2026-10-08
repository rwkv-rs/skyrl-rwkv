#!/usr/bin/env bash

set -euo pipefail

remote='cd ~/Projects/MachineLearning/skyrl-rwkv && '
for name in MODEL_DIR DATA_DIR NUM_GPUS NUM_POLICY_GPUS NUM_INFERENCE_GPUS MINI_BATCH_SIZE MICRO_BATCH_SIZE LOGGER RUN_NAME OUTPUT_ROOT TRAINING_ENTRYPOINT; do
  if [[ ${!name+x} ]]; then
    printf -v value '%q' "${!name}"
    remote+="$name=$value "
  fi
done
if [[ ${1:-} == show ]]; then
  shift
  remote+='exec uv run --isolated --no-sync --extra rwkv python temp/show_details.py'
else
  remote+='exec bash examples/train/rwkv/run_rwkv_flashreinforce.sh'
fi
for arg in "$@"; do
  printf -v quoted '%q' "$arg"
  remote+=" $quoted"
done
exec ssh rwkv-sha-pro6000x8 "$remote"
