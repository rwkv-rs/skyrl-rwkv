#!/usr/bin/env bash

set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_DIR:=$HOME/Weights/RWKV/hf/rwkv7-g1j-1.5b-20260831-ctx16384}"
: "${DATA_DIR:=$HOME/data/gsm8k}"
: "${NUM_GPUS:=8}"
: "${NUM_POLICY_GPUS:=$NUM_GPUS}"
: "${NUM_INFERENCE_GPUS:=$NUM_GPUS}"
: "${MINI_BATCH_SIZE:=128}"
: "${MICRO_BATCH_SIZE:=2}"
: "${LOGGER:=wandb}"
: "${TRAINING_ENTRYPOINT:=skyrl.train.entrypoints.main_base}"
: "${RUN_NAME:=rwkv7-g1j-1.5b-20260831-ctx16384-gsm8k-flashreinforce-50step}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
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
  uv run --no-sync --extra rwkv python examples/train/rwkv/prepare_flashrwkv2.py
  exit 0
fi

# Check the exact native cache key before loading.
uv run --no-sync --extra rwkv python examples/train/rwkv/prepare_flashrwkv2.py --check

uv run --no-sync --extra rwkv -m "$TRAINING_ENTRYPOINT" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.policy_loss_type=flashreinforce \
  trainer.algorithm.advantage_estimator=flashreinforce \
  trainer.algorithm.loss_reduction=sequence_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.policy.model.path="$MODEL_DIR" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.scheduler=cosine \
  trainer.strategy=fsdp \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.policy.sequence_parallel_size=1 \
  trainer.policy.use_torch_compile=false \
  trainer.gradient_checkpointing=true \
  trainer.placement.colocate_all=true \
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
  generator.inference_engine.model_dtype=float16 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.batched=false \
  generator.preserve_truncated_action_mask=true \
  generator.n_samples_per_prompt=1 \
  generator.eval_n_samples_per_prompt=1 \
  generator.sampling_params.logprobs=1 \
  generator.chat_template_kwargs="{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}" \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.additional_kwargs=null \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 \
  generator.eval_sampling_params.top_k=32 \
  generator.eval_sampling_params.additional_kwargs="{presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}" \
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32 \
  generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true \
  environment.env_class=gsm8k \
  environment.skyrl_gym.gsm8k.strict_reward=true \
  "$@"
