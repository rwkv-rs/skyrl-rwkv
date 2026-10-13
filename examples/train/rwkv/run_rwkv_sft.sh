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
: "${MIN_LR_RATE:=0.1}"
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
  optimizer_config.scheduler=cosine_with_min_lr \
  "optimizer_config.scheduler_specific_kwargs={min_lr_rate: $MIN_LR_RATE}" \
  optimizer_config.num_warmup_steps="$WARMUP_STEPS" \
  optimizer_config.offload_after_step=false \
  sampler=custom sampler_class_path=examples.train.rwkv.prepare_sft.LengthGroupedSampler \
  "sampler_kwargs={batch_size: $BATCH_SIZE, seed: 0}" seed=0 \
  dataloader_num_workers=2 dataloader_persistent_workers=true num_workers=0 \
  ckpt_path="$OUTPUT_ROOT/checkpoints" export_path="$OUTPUT_ROOT/hf_exports" \
  ckpt_interval=50 eval_interval=50 hf_save_interval=50 \
  eval_before_train=true max_ckpts_to_keep=2 \
  logger="$LOGGER" project_name=rwkv_direct_sft run_name="$RUN_NAME" \
  "$@"

# An optional prepared benchmark directory queues full eval after the epoch.
if [[ -n "${EVAL_DATA_DIR:-}" ]]; then
  unset WANDB_RUN_ID WANDB_RESUME
  for phase in base sft; do
    EVAL_MODEL="$MODEL_DIR"
    if [[ "$phase" == sft ]]; then
      EVAL_MODEL="$OUTPUT_ROOT/hf_exports/global_step_$TOTAL_STEPS/policy"
    fi
    uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
      -m skyrl.train.entrypoints.main_generate \
      "data.val_data=['$EVAL_DATA_DIR/math500.parquet','$EVAL_DATA_DIR/gsm8k.parquet']" \
      trainer.policy.model.path="$EVAL_MODEL" \
      'trainer.policy.model_config_kwargs={wkv_mode: fp32io16}' \
      trainer.strategy=fsdp trainer.flash_attn=false trainer.remove_microbatch_padding=false \
      trainer.placement.colocate_all=false trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
      trainer.eval_interval=1 trainer.eval_batch_size=256 trainer.max_prompt_length=4096 \
      trainer.dump_eval_results=true trainer.logger="$LOGGER" \
      trainer.project_name=rwkv_direct_sft_eval trainer.run_name="$RUN_NAME-$phase-full-eval" \
      trainer.export_path="$OUTPUT_ROOT/evaluation/$phase" \
      generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=float16 \
      generator.inference_engine.run_engines_locally=true \
      generator.inference_engine.num_engines="$NUM_GPUS" generator.inference_engine.tensor_parallel_size=1 \
      generator.inference_engine.gpu_memory_utilization=0.8 \
      generator.inference_engine.max_num_seqs=256 generator.inference_engine.max_num_batched_tokens=32768 \
      generator.batched=false generator.eval_n_samples_per_prompt=1 \
      'generator.chat_template_kwargs={rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}' \
      generator.eval_sampling_params.max_generate_length=8192 \
      generator.eval_sampling_params.temperature=0.96 generator.eval_sampling_params.top_p=0.76 \
      generator.eval_sampling_params.top_k=32 \
      'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
      environment.env_class=gsm8k
  done
fi
