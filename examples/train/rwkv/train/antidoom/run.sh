#!/usr/bin/env bash

source "$(cd "$(dirname "$0")/.." && pwd)/launcher_common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
DATA_DIR="${DATA_DIR:-$HOME/data/rwkv-antidoom-ftpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/rwkv7-g1k-antidoom}"
NUM_GPUS="${NUM_GPUS:-8}"
TEST_MODE=false
if [[ "${1:-}" == "--test_mode" || "${1:-}" == "--test-mode" ]]; then TEST_MODE=true; shift; fi
configure_wandb "$TEST_MODE"
if [[ "$TEST_MODE" == true ]]; then REPORT_TO=none; else REPORT_TO=wandb; fi

TRAIN_DATA="$DATA_DIR"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  mkdir -p "$TEST_DATA_DIR"
  prepare_test_jsonl "$DATA_DIR/train.jsonl" "$TEST_DATA_DIR/train.jsonl" "${TEST_ROWS:-256}"
  prepare_test_jsonl "$DATA_DIR/validation.jsonl" "$TEST_DATA_DIR/validation.jsonl" "${TEST_VAL_ROWS:-32}"
  cp "$DATA_DIR/manifest.json" "$TEST_DATA_DIR/manifest.json" 2>/dev/null || true
  TRAIN_DATA="$TEST_DATA_DIR"
  MAX_STEPS="${MAX_STEPS:-5}"
  EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-256}"
else
  MAX_STEPS="${MAX_STEPS:-0}"
  EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1024}"
fi

# The NoRA reference uses eight FSDP ranks.  Two examples per microbatch fit
# under the 32k-token cap on the 96 GB Pro6000 GPUs; halving accumulation keeps
# the same 128-example effective batch while removing a synchronization bubble.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-32768}"

mkdir -p "$OUTPUT_ROOT"
run_rwkv_python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  "$ROOT/train/antidoom/train.py" train --model "$MODEL_DIR" --data "$TRAIN_DATA" --output "$OUTPUT_ROOT" \
  --rank "${LORA_RANK:-256}" --lora-alpha "${LORA_ALPHA:-256}" \
  --learning-rate "${LEARNING_RATE:-5e-7}" --epochs "${EPOCHS:-1}" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" --max-batch-tokens "$MAX_BATCH_TOKENS" \
  --fsdp-size "${FSDP_SIZE:-$NUM_GPUS}" --eval-max-samples "$EVAL_MAX_SAMPLES" \
  --max-steps "$MAX_STEPS" --save-steps "${SAVE_STEPS:-50}" \
  --report-to "$REPORT_TO" "$@"
