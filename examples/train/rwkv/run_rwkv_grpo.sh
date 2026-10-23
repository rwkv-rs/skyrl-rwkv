#!/usr/bin/env bash

set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_DIR:=$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
: "${DATA_DIR:=$HOME/data/gsm8k}"
: "${NUM_GPUS:=8}"

# Optional native async rollout/learner placement. Fixed modes keep whole logical batches.
: "${ASYNC_MODE:=disabled}"
: "${ASYNC_STREAMING_BATCH_SIZE:=0}"
: "${MINI_BATCH_SIZE:=64}"
ASYNC_ENABLED=false
COLOCATE_ALL=true
ASYNC_STALENESS=1
if [[ "$ASYNC_MODE" != disabled ]]; then
  ASYNC_ENABLED=true
  COLOCATE_ALL=false
  : "${NUM_POLICY_GPUS:=$((NUM_GPUS - 1))}"
  : "${NUM_INFERENCE_GPUS:=1}"
  if [[ "$ASYNC_MODE" == zero_stale ]]; then
    ASYNC_STALENESS=0
  fi
fi
: "${NUM_POLICY_GPUS:=$NUM_GPUS}"
: "${NUM_INFERENCE_GPUS:=$NUM_GPUS}"
: "${ASYNC_GENERATION_WORKERS:=$MINI_BATCH_SIZE}"
POLICY_LOSS_TYPE=regular
if [[ "$ASYNC_ENABLED" == true && "$ASYNC_MODE" != zero_stale ]]; then
  POLICY_LOSS_TYPE=rollout_is
fi
: "${MICRO_BATCH_SIZE:=2}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=rwkv7-g1k-1.5b-20260930-ctx25600-gsm8k-grpo-50step}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
export UV_PROJECT_ENVIRONMENT="$(realpath .venv)"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# FlashRWKV2 uses get_default_build_root() (XDG_CACHE_HOME), not TORCH_EXTENSIONS_DIR.
export XDG_CACHE_HOME="$HOME/.cache/skyrl-rwkv"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"

if [[ "${1:-}" == --prepare-flashrwkv2 ]]; then
  shift
  if (( $# )); then
    printf 'Unexpected arguments to --prepare-flashrwkv2\n' >&2
    exit 2
  fi
  export MAX_JOBS="${MAX_JOBS:-2}"
  uv sync --extra rwkv
  uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" examples/train/rwkv/prepare_flashrwkv2.py
  exit 0
fi

# Check the exact native cache key before loading.
uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" examples/train/rwkv/prepare_flashrwkv2.py --check

uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type="$POLICY_LOSS_TYPE" \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="$MODEL_DIR" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.strategy=fsdp \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.policy.sequence_parallel_size=1 \
  trainer.policy.use_torch_compile=false \
  trainer.gradient_checkpointing=true \
  trainer.fully_async.enabled="$ASYNC_ENABLED" \
  trainer.fully_async.mode="${ASYNC_MODE/disabled/fully_async}" \
  trainer.fully_async.max_staleness_steps="$ASYNC_STALENESS" \
  trainer.fully_async.num_parallel_generation_workers="$ASYNC_GENERATION_WORKERS" \
  trainer.fully_async.streaming_batch_size="$ASYNC_STREAMING_BATCH_SIZE" \
  trainer.fully_async.clear_kv_cache_on_weight_sync=true \
  trainer.placement.colocate_all="$COLOCATE_ALL" \
  trainer.placement.policy_num_gpus_per_node="$NUM_POLICY_GPUS" \
  trainer.placement.critic_num_gpus_per_node="$NUM_POLICY_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_POLICY_GPUS" \
  trainer.epochs=50 \
  trainer.max_training_steps=50 \
  trainer.train_batch_size="$MINI_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.dump_train_results=true \
  trainer.num_logger_train_samples=20 \
  trainer.eval_interval=50 \
  trainer.ckpt_interval=50 \
  trainer.hf_save_interval=50 \
  trainer.max_prompt_length=512 \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name=skyrl-rwkv \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="$OUTPUT_ROOT/logs" \
  trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" \
  trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.batched=false \
  generator.n_samples_per_prompt=4 \
  generator.eval_n_samples_per_prompt=1 \
  generator.chat_template_kwargs="{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}" \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.logprobs=1 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.additional_kwargs="{presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}" \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 \
  generator.eval_sampling_params.top_k=32 \
  generator.eval_sampling_params.additional_kwargs="{presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}" \
  environment.env_class=gsm8k \
  "$@"
