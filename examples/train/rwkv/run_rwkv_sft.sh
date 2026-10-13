#!/usr/bin/env bash
# Full-parameter RWKV SFT on complete, pretokenized Nemotron CoT.
# Prepare data with examples.train.rwkv.prepare_sft before running this script.
# Usage from the repository root:
#   DATA_DIR=<prepared-data> OUTPUT_ROOT=<run-directory> NUM_GPUS=8 \
#     bash examples/train/rwkv/run_rwkv_sft.sh

set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_DIR:=$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
: "${DATA_DIR:=$HOME/data/rwkv-direct-sft}"
: "${NUM_GPUS:=8}"
: "${BATCH_SIZE:=$((NUM_GPUS * 8))}"
: "${MICRO_BATCH_SIZE:=1}"
: "${CHUNK_SIZE:=2048}"
: "${LEARNING_RATE:=1e-6}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=rwkv7-g1k-1.5b-20260930-ctx25600-nemotron-cot-sft}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
export RAY_ADDRESS=local
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export SKYRL_PYTHONPATH_EXPORT=1
export XDG_CACHE_HOME="$HOME/.cache/skyrl-rwkv"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/train.log") 2>&1

TOTAL_STEPS=$(uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" - "$DATA_DIR" "$BATCH_SIZE" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
print(math.ceil(manifest["counts"]["train_rows"] / int(sys.argv[2])))
PY
)
WARMUP_STEPS=$((TOTAL_STEPS / 20))

uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" -m skyrl.train.main_sft \
  strategy=fsdp model.path="$MODEL_DIR" \
  "pretokenized_dataset_paths=['$DATA_DIR/train']" \
  "eval_pretokenized_dataset_paths=['$DATA_DIR/eval']" \
  'eval_dataset_names=[nemotron_cot]' \
  max_length=null remove_microbatch_padding=false use_sequence_packing=false \
  rwkv_chunk_size="$CHUNK_SIZE" 'model_config_kwargs={wkv_mode: fp32io16}' \
  num_epochs=1 max_training_steps="$TOTAL_STEPS" \
  batch_size="$BATCH_SIZE" micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  placement.num_nodes=1 placement.num_gpus_per_node="$NUM_GPUS" \
  fsdp_config.fsdp_size="$NUM_GPUS" fsdp_config.reshard_after_forward=false \
  fsdp_config.sync_gradients_each_microbatch=false \
  optimizer_config.lr="$LEARNING_RATE" optimizer_config.max_grad_norm=1.0 \
  optimizer_config.scheduler=cosine optimizer_config.num_warmup_steps="$WARMUP_STEPS" \
  optimizer_config.offload_after_step=false \
  sampler=custom sampler_class_path=examples.train.rwkv.prepare_sft.LengthGroupedSampler \
  "sampler_kwargs={batch_size: $BATCH_SIZE, seed: 0}" seed=0 \
  dataloader_num_workers=2 dataloader_persistent_workers=true num_workers=0 \
  ckpt_path="$OUTPUT_ROOT/checkpoints" export_path="$OUTPUT_ROOT/hf_exports" \
  ckpt_interval=50 eval_interval=50 hf_save_interval=50 \
  eval_before_train=true max_ckpts_to_keep=2 \
  logger="$LOGGER" project_name=rwkv_direct_sft run_name="$RUN_NAME" \
  "$@"
