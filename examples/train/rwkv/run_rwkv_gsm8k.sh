#!/usr/bin/env bash

set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_DIR:=$HOME/models/rwkv7-g1-st-e1a670a5523742b5cfe8cb6759c1eb8f1d88b637/rwkv7-g1j-1.5b-20260831-ctx16384}"
: "${DATA_DIR:=$HOME/data/gsm8k}"
: "${NUM_GPUS:=8}"
: "${MICRO_BATCH_SIZE:=2}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=rwkv7-g1j-1.5b-gsm8k-grpo-50step}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
: "${UV_PROJECT_ENVIRONMENT:=$PWD/.venv}"
export UV_PROJECT_ENVIRONMENT

uv run --no-sync --extra rwkv -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.model.path="$MODEL_DIR" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.strategy=fsdp \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.policy.sequence_parallel_size=1 \
  trainer.policy.use_torch_compile=false \
  trainer.gradient_checkpointing=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.critic_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.epochs=50 \
  trainer.max_training_steps=50 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
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
  generator.inference_engine.model_dtype=float16 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.batched=false \
  generator.n_samples_per_prompt=4 \
  generator.eval_n_samples_per_prompt=1 \
  generator.chat_template_kwargs="{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}" \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.additional_kwargs="{presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}" \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 \
  generator.eval_sampling_params.top_k=32 \
  generator.eval_sampling_params.additional_kwargs="{presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}" \
  environment.env_class=gsm8k \
  "$@"
