#!/usr/bin/env bash

source "$(cd "$(dirname "$0")/.." && pwd)/launcher_common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
DATA_DIR="${DATA_DIR:-$HOME/data/rwkv-antidoom-ftpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/rwkv7-g1k-antidoom}"
TEST_MODE=false
if [[ "${1:-}" == "--test_mode" || "${1:-}" == "--test-mode" ]]; then TEST_MODE=true; shift; fi
configure_wandb "$TEST_MODE"
if [[ "$TEST_MODE" == true ]]; then REPORT_TO=none; else REPORT_TO=wandb; fi

TRAIN_DATA="$DATA_DIR"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  mkdir -p "$TEST_DATA_DIR"
  prepare_test_jsonl "$DATA_DIR/train.jsonl" "$TEST_DATA_DIR/train.jsonl" 128
  prepare_test_jsonl "$DATA_DIR/validation.jsonl" "$TEST_DATA_DIR/validation.jsonl" 32
  cp "$DATA_DIR/manifest.json" "$TEST_DATA_DIR/manifest.json" 2>/dev/null || true
  TRAIN_DATA="$TEST_DATA_DIR"
  MAX_STEPS=5
else
  MAX_STEPS="${MAX_STEPS:-0}"
fi

mkdir -p "$OUTPUT_ROOT"
run_rwkv_python -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/train/antidoom/train.py" train --model "$MODEL_DIR" --data "$TRAIN_DATA" --output "$OUTPUT_ROOT" \
  --rank "${LORA_RANK:-256}" --lora-alpha "${LORA_ALPHA:-256}" \
  --learning-rate "${LEARNING_RATE:-5e-7}" --epochs "${EPOCHS:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" --max-batch-tokens "${MAX_BATCH_TOKENS:-32768}" \
  --max-steps "$MAX_STEPS" --save-steps "${SAVE_STEPS:-50}" \
  --report-to "$REPORT_TO" "$@"
