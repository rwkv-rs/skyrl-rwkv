#!/usr/bin/env bash

source "$(cd "$(dirname "$0")/.." && pwd)/launcher_common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/rwkv7-g1k-1.5b-flashreinforce}"
RUN_NAME="${RUN_NAME:-rwkv7-g1k-1.5b-flashreinforce}"
TEST_MODE=false
if [[ "${1:-}" == "--test_mode" || "${1:-}" == "--test-mode" ]]; then TEST_MODE=true; shift; fi
configure_wandb "$TEST_MODE"
if [[ "$TEST_MODE" == true ]]; then LOGGER=console; else LOGGER=wandb; fi

# Fully async is the throughput-oriented default: generation has its own GPU
# group and can continue while the learner consumes completed trajectories.
# Set ASYNC_MODE=disabled to benchmark the colocated synchronous path.
ASYNC_MODE="${ASYNC_MODE:-fully_async}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-4}"
if [[ "$ASYNC_MODE" == disabled ]]; then
  ASYNC_ENABLED=false
  ASYNC_MODE_CONFIG=fully_async
  COLOCATE_ALL=true
  POLICY_GPUS="${POLICY_GPUS:-8}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-8}"
else
  ASYNC_ENABLED=true
  ASYNC_MODE_CONFIG="$ASYNC_MODE"
  COLOCATE_ALL=false
  POLICY_GPUS="${POLICY_GPUS:-4}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-4}"
fi

if [[ "${1:-}" == "--prepare-flashrwkv2" ]]; then
  shift
  [[ $# -eq 0 ]] || { echo 'unexpected arguments to --prepare-flashrwkv2' >&2; exit 2; }
  run_rwkv_python "$ROOT/train/common/prepare_flashrwkv2.py"
  exit 0
fi

TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="$DATA_DIR/validation.parquet"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  prepare_test_parquet "$TRAIN_DATA" "$TEST_DATA_DIR/train.parquet" "${TEST_ROWS:-320}"
  prepare_test_parquet "$VAL_DATA" "$TEST_DATA_DIR/validation.parquet" "${TEST_VAL_ROWS:-32}"
  TRAIN_DATA="$TEST_DATA_DIR/train.parquet"
  VAL_DATA="$TEST_DATA_DIR/validation.parquet"
  MAX_STEPS="${MAX_STEPS:-5}"
  # The synchronous trainer trims a batch to a multiple of its effective DP
  # size.  Keep the smoke batch divisible by the configured learner GPU count,
  # otherwise a six-GPU diagnostic turns 16 prompts into 12 and violates the
  # prompt-boundary invariant before the first optimizer step.
  BATCH_SIZE="${BATCH_SIZE:-32}"
  if (( BATCH_SIZE % POLICY_GPUS != 0 )); then
    BATCH_SIZE=$(( ((BATCH_SIZE + POLICY_GPUS - 1) / POLICY_GPUS) * POLICY_GPUS ))
  fi
  if [[ "$ASYNC_MODE" == disabled ]]; then
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
  else
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
  fi
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
  EVAL_MAX_GENERATE_LENGTH="${EVAL_MAX_GENERATE_LENGTH:-1024}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
else
  MAX_STEPS="${MAX_STEPS:-50}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
  EVAL_MAX_GENERATE_LENGTH="${EVAL_MAX_GENERATE_LENGTH:-1024}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
fi
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-32768}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-$BATCH_SIZE}"
if [[ "$ASYNC_ENABLED" == true ]]; then
  DEFAULT_GENERATION_WORKERS=$((INFERENCE_GPUS * 64))
  if [[ "$TEST_MODE" == true && "$DEFAULT_GENERATION_WORKERS" -gt 128 ]]; then
    DEFAULT_GENERATION_WORKERS=128
  fi
  MAX_GENERATION_WORKERS=$((POLICY_MINI_BATCH_SIZE * (MAX_STALENESS_STEPS + 1)))
  if [[ "$DEFAULT_GENERATION_WORKERS" -gt "$MAX_GENERATION_WORKERS" ]]; then
    DEFAULT_GENERATION_WORKERS="$MAX_GENERATION_WORKERS"
  fi
  NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-$DEFAULT_GENERATION_WORKERS}"
else
  NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-$BATCH_SIZE}"
fi

USE_NORA="${USE_NORA:-false}"
LORA_RANK="${LORA_RANK:-256}"
LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
LORA_OVERRIDES=(trainer.policy.model.lora.use_nora="$USE_NORA")
if [[ "$USE_NORA" == true ]]; then
  LORA_OVERRIDES+=(trainer.policy.model.lora.rank="$LORA_RANK" trainer.policy.model.lora.alpha="$LORA_ALPHA" \
    'trainer.policy.model.lora.target_modules=[r_proj,k_proj,v_proj,o_proj,mlp.key,mlp.value,lm_head]')
else
  LORA_OVERRIDES+=(trainer.policy.model.lora.rank=0)
fi

run_rwkv_python -m skyrl.train.entrypoints.main_base \
  "data.train_data=['$TRAIN_DATA']" "data.val_data=['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator=flashreinforce trainer.algorithm.policy_loss_type=flashreinforce \
  trainer.algorithm.loss_reduction=sequence_mean trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL_DIR" trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  "${LORA_OVERRIDES[@]}" trainer.strategy=fsdp trainer.bf16=true trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false trainer.gradient_checkpointing=true \
  trainer.policy.optimizer_config.lr="${LEARNING_RATE:-1e-6}" trainer.policy.optimizer_config.scheduler=cosine \
  trainer.policy.optimizer_config.offload_after_step=false \
  trainer.policy.fsdp_config.reshard_after_forward=false trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.fully_async.enabled="$ASYNC_ENABLED" trainer.fully_async.mode="$ASYNC_MODE_CONFIG" \
  trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.fully_async.clear_kv_cache_on_weight_sync=true \
  trainer.placement.colocate_all="$COLOCATE_ALL" trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_gpus_per_node="$POLICY_GPUS" trainer.placement.ref_num_gpus_per_node="$POLICY_GPUS" \
  trainer.epochs=50 trainer.max_training_steps="$MAX_STEPS" \
  trainer.train_batch_size="$BATCH_SIZE" trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.max_tokens_per_microbatch="$MAX_TOKENS_PER_MICROBATCH" \
  trainer.eval_interval=50 trainer.ckpt_interval=50 trainer.hf_save_interval=50 trainer.eval_before_train=false \
  trainer.max_prompt_length=512 \
  trainer.logger="$LOGGER" trainer.project_name=skyrl-rwkv trainer.run_name="$RUN_NAME" \
  trainer.log_path="$OUTPUT_ROOT/logs" trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.run_engines_locally=true generator.inference_engine.num_engines="$INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.engine_init_kwargs.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32 \
  generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true \
  generator.batched=false generator.preserve_truncated_action_mask=true \
  generator.n_samples_per_prompt=1 generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES_PER_PROMPT" \
  generator.chat_template_kwargs='{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}' \
  generator.sampling_params.logprobs=1 generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH:-8192}" \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 generator.sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length="$EVAL_MAX_GENERATE_LENGTH" generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 generator.eval_sampling_params.top_k=32 \
  'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
  environment.env_class=gsm8k environment.skyrl_gym.gsm8k.strict_reward=true "$@"
