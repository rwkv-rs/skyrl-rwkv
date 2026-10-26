#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")/../train" && pwd)/launcher_common.sh"

SOURCE_ROOT="${SOURCE_ROOT:-$HOME/data/ultradata-rwkv-source/files}"
MODEL="${MODEL:-$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
ROLLOUT_OUTPUT="${ROLLOUT_OUTPUT:-$HOME/data/ultradata-rwkv-rollouts}"
FILTERED_OUTPUT="${FILTERED_OUTPUT:-$HOME/data/ultradata-rwkv/files}"
PREPARE_SOURCE="${PREPARE_SOURCE:-true}"
PREPARE_LIMIT_PER_SOURCE="${PREPARE_LIMIT_PER_SOURCE:-0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-64}"
LIMIT="${LIMIT:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
CONCURRENCY="${CONCURRENCY:-}"
BASE_URL="${BASE_URL:-}"

ROLLOUT_ARGS=(
  "$ROOT/train/common/rollout.py"
  --data-root "$SOURCE_ROOT"
  --output "$ROLLOUT_OUTPUT"
  --model "$MODEL"
  --samples-per-prompt "$SAMPLES_PER_PROMPT"
  --sampling eval
  --limit "$LIMIT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-tokens "$MAX_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
)
if [[ -n "$CONCURRENCY" ]]; then ROLLOUT_ARGS+=(--concurrency "$CONCURRENCY"); fi
if [[ -n "$BASE_URL" ]]; then ROLLOUT_ARGS+=(--base-url "$BASE_URL"); fi
if [[ "${SAVE_ROLLOUTS:-false}" == true ]]; then ROLLOUT_ARGS+=(--save-rollouts); fi
if [[ "${RESUME:-false}" == true ]]; then ROLLOUT_ARGS+=(--resume); fi

mkdir -p "$ROLLOUT_OUTPUT"
if [[ "$PREPARE_SOURCE" == true ]]; then
  run_rwkv_python "$ROOT/get_ultradata_rwkv/prepare_source.py" \
    --output "$SOURCE_ROOT" \
    --limit-per-source "$PREPARE_LIMIT_PER_SOURCE" \
    > "$ROLLOUT_OUTPUT/source.log" 2>&1
fi
run_rwkv_python "${ROLLOUT_ARGS[@]}" > "$ROLLOUT_OUTPUT/rollout.log" 2>&1
run_rwkv_python "$ROOT/get_ultradata_rwkv/filter_dataset.py" \
  --input "$SOURCE_ROOT" \
  --rollout-output "$ROLLOUT_OUTPUT" \
  --output "$FILTERED_OUTPUT" \
  --expected-samples "$SAMPLES_PER_PROMPT" \
  > "$ROLLOUT_OUTPUT/filter.log" 2>&1

echo "Filtered RWKV corpus written to $FILTERED_OUTPUT"
