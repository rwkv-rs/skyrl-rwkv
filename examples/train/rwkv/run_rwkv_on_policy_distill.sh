#!/usr/bin/env bash

set -euo pipefail

# Native OPD on local UltraData, with AIME 2024 evaluation.
# Teacher: rwkv7-g1k-13.3b-20260930-ctx25600-antidoom-nora.
# Student: rwkv7-g1k-1.5b-20260930-ctx25600.

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${STUDENT_MODEL:=$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
: "${TEACHER_MODEL:=$HOME/Weights/RWKV/hf/rwkv7-g1k-13.3b-20260930-ctx25600-antidoom-nora}"
: "${DATA_DIR:=$HOME/data/ultradata-opd}"
: "${TRAIN_FILE:=$DATA_DIR/train.parquet}"
: "${TEST_FILE:=$HOME/data/dapo/aime-2024-cleaned.parquet}"
: "${NUM_GPUS:=8}"
: "${USE_NORA:=true}"
: "${LORA_RANK:=256}"
: "${LORA_ALPHA:=$LORA_RANK}"
: "${MINI_BATCH_SIZE:=512}"
: "${MICRO_BATCH_SIZE:=4}"
: "${MICRO_FORWARD_BATCH_SIZE:=4}"
: "${LR:=2e-6}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=rwkv7-g1k-1.5b-20260930-ctx25600-ultradata-opd-nora-from-rwkv7-g1k-13.3b-20260930-ctx25600-antidoom-nora}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
# Share the exact interpreter/cache identity with torchrun across worktrees.
export UV_PROJECT_ENVIRONMENT="$(realpath .venv)"
# Workers inherit this prepared environment, including AntiDoom's PEFT fork.
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
  uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" examples/train/rwkv/prepare_flashrwkv2.py
  exit 0
fi

# Check the exact native cache key before loading.
uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" examples/train/rwkv/prepare_flashrwkv2.py --check

uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" -m examples.train.on_policy_distillation.main_on_policy_distill \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator=no_op \
  trainer.algorithm.policy_loss_type=importance_sampling \
  trainer.algorithm.use_kl_in_reward=true \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$STUDENT_MODEL" \
  trainer.ref.model.path="$TEACHER_MODEL" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.ref.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.policy.model.lora.use_nora="$USE_NORA" \
  trainer.policy.model.lora.rank="$LORA_RANK" \
  trainer.policy.model.lora.alpha="$LORA_ALPHA" \
  trainer.policy.model.lora.dropout=0.0 \
  trainer.policy.model.lora.target_modules="[r_proj,k_proj,v_proj,o_proj,lm_head]" \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.strategy=fsdp \
  trainer.bf16=true \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.policy.sequence_parallel_size=1 \
  trainer.ref.sequence_parallel_size=1 \
  trainer.policy.use_torch_compile=false \
  trainer.gradient_checkpointing=true \
  trainer.policy.fsdp_config.fsdp_size=1 \
  trainer.policy.fsdp_config.reshard_after_forward=false \
  trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.ref.fsdp_config.fsdp_size="$NUM_GPUS" \
  trainer.ref.fsdp_config.reshard_after_forward=false \
  trainer.update_ref_every_epoch=false \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.epochs=1 \
  trainer.max_training_steps=null \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$MINI_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_FORWARD_BATCH_SIZE" \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=true \
  trainer.dump_train_results=true \
  trainer.num_logger_train_samples=20 \
  trainer.eval_interval=50 \
  trainer.ckpt_interval=50 \
  trainer.hf_save_interval=50 \
  trainer.max_prompt_length=17408 \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name=skyrl-rwkv \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="$OUTPUT_ROOT/logs" \
  trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" \
  trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.model_dtype=float16 \
  generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.engine_init_kwargs.kv_cache_memory_bytes=2147483648 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.max_num_seqs=128 \
  generator.batched=false \
  generator.n_samples_per_prompt=1 \
  generator.eval_n_samples_per_prompt=32 \
  generator.chat_template_kwargs="{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}" \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.additional_kwargs="{presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}" \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 \
  generator.eval_sampling_params.top_k=32 \
  generator.eval_sampling_params.additional_kwargs="{presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}" \
  environment.env_class=aime \
  "$@"
