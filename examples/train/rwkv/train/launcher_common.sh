#!/usr/bin/env bash

# Shared shell-only setup.  Keeping data slicing in train/data.py means every
# run.sh remains a launcher and contains no embedded Python program.
set -euo pipefail

RWKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$RWKV_ROOT/../../.." && pwd)"
# ROOT is consumed by the individual launchers after this file is sourced.
# shellcheck disable=SC2034
ROOT="$RWKV_ROOT"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python3}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export FLASHRWKV2_CACHE_DIR="${FLASHRWKV2_CACHE_DIR:-$HOME/.cache/flashrwkv2}"
export VLLM_RWKV_TUNING_MODE="${VLLM_RWKV_TUNING_MODE:-require}"
# The repository contains the separately packaged skyrl-gym project.  Export
# both roots so every launcher works when invoked directly, without requiring
# callers to repair imports in their shell first.
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/skyrl-gym${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

run_rwkv_python() {
  uv run --isolated --no-config --no-project "$PYTHON" "$@"
}

run_rwkv_training() {
  if [[ -z "${RWKV_TRAIN_EVAL:-}" ]]; then
    run_rwkv_python "$@"
    return
  fi
  timeout --signal=TERM --kill-after=120s "${TRAIN_MAX_SECONDS:?Set TRAIN_MAX_SECONDS}s" \
    uv run --isolated --no-config --no-project "$PYTHON" "$@" || return "$?"
  run_rwkv_python "$RWKV_ROOT/eval/prepare.py" --after-train "$RWKV_TRAIN_EVAL"
}

configure_wandb() {
  if [[ "${1:-false}" == true ]]; then
    export WANDB_MODE=disabled
    export WANDB_DISABLED=true
  else
    unset WANDB_DISABLED
    export WANDB_MODE=online
  fi
}

prepare_test_parquet() {
  local source="$1" destination="$2" rows="${3:-128}"
  run_rwkv_python "$RWKV_ROOT/train/common/test_data.py" --source "$source" --destination "$destination" \
    --rows "$rows" --format parquet > "/tmp/results_rwkv_test_data.log" 2>&1
}

prepare_test_jsonl() {
  local source="$1" destination="$2" rows="${3:-128}"
  run_rwkv_python "$RWKV_ROOT/train/common/test_data.py" --source "$source" --destination "$destination" \
    --rows "$rows" --format jsonl > "/tmp/results_rwkv_test_data.log" 2>&1
}
