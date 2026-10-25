#!/usr/bin/env bash

source "$(cd "$(dirname "$0")/.." && pwd)/launcher_common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/Weights/RWKV/hf/rwkv7-g1l-1.5b-temp-6331}"
CRITIC_MODEL_DIR="${CRITIC_MODEL_DIR:-$MODEL_DIR}"
DATA_DIR="${DATA_DIR:-$HOME/data/ultradata-sao}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/rwkv7-g1l-sao}"
RUN_NAME="${RUN_NAME:-rwkv7-g1l-sao-8gpu}"
TEST_MODE=false
if [[ "${1:-}" == "--test_mode" || "${1:-}" == "--test-mode" ]]; then TEST_MODE=true; shift; fi
configure_wandb "$TEST_MODE"
if [[ "$TEST_MODE" == true ]]; then LOGGER=console; else LOGGER=wandb; fi

# Keep rollout on dedicated GPUs by default so the critic and policy can train
# while the next single-rollout batch is generated.  ASYNC_MODE=disabled
# selects the fully colocated synchronous baseline.
ASYNC_MODE="${ASYNC_MODE:-fully_async}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-4}"
if [[ "$ASYNC_MODE" == disabled ]]; then
  ASYNC_ENABLED=false
  ASYNC_MODE_CONFIG=fully_async
  COLOCATE_ALL=true
  TRAIN_CRITIC_AND_POLICY_PARALLEL=false
  POLICY_GPUS="${POLICY_GPUS:-8}"
  CRITIC_GPUS="${CRITIC_GPUS:-8}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-8}"
else
  ASYNC_ENABLED=true
  ASYNC_MODE_CONFIG="$ASYNC_MODE"
  COLOCATE_ALL=false
  TRAIN_CRITIC_AND_POLICY_PARALLEL=true
  POLICY_GPUS="${POLICY_GPUS:-2}"
  CRITIC_GPUS="${CRITIC_GPUS:-4}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-2}"
fi

TRAIN_DATA="$DATA_DIR/train.parquet"; VAL_DATA="$DATA_DIR/validation.parquet"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  prepare_test_parquet "$TRAIN_DATA" "$TEST_DATA_DIR/train.parquet" "${TEST_ROWS:-320}"
  prepare_test_parquet "$VAL_DATA" "$TEST_DATA_DIR/validation.parquet" "${TEST_VAL_ROWS:-32}"
  TRAIN_DATA="$TEST_DATA_DIR/train.parquet"; VAL_DATA="$TEST_DATA_DIR/validation.parquet"
  MAX_STEPS="${MAX_STEPS:-5}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
  POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-32}"; CRITIC_MINI_BATCH_SIZE="${CRITIC_MINI_BATCH_SIZE:-16}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
  # Smoke runs should exercise the trainer, not expand evaluation into a full
  # pass@128 job.  The validation slice has 32 rows, so the old value of 128
  # created 4096 concurrent trajectories before step 1.
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
  EVAL_MAX_GENERATE_LENGTH="${EVAL_MAX_GENERATE_LENGTH:-1024}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
else
  MAX_STEPS="${MAX_STEPS:-50}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
  POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-128}"; CRITIC_MINI_BATCH_SIZE="${CRITIC_MINI_BATCH_SIZE:-64}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-128}"
  EVAL_MAX_GENERATE_LENGTH="${EVAL_MAX_GENERATE_LENGTH:-25600}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
fi
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-32768}"
if [[ "$ASYNC_ENABLED" == true ]]; then
  POLICY_MINI_BATCH_SIZE="$TRAIN_BATCH_SIZE"
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
  NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-$TRAIN_BATCH_SIZE}"
fi

POLICY_USE_NORA="${POLICY_USE_NORA:-false}"
CRITIC_USE_NORA="${CRITIC_USE_NORA:-true}"
POLICY_LORA_RANK="${POLICY_LORA_RANK:-256}"
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-64}"
POLICY_LORA=(trainer.policy.model.lora.use_nora="$POLICY_USE_NORA")
CRITIC_LORA=(trainer.critic.model.lora.use_nora="$CRITIC_USE_NORA")
if [[ "$POLICY_USE_NORA" == true ]]; then
  POLICY_LORA+=(trainer.policy.model.lora.rank="$POLICY_LORA_RANK" trainer.policy.model.lora.alpha="$POLICY_LORA_RANK" \
    'trainer.policy.model.lora.target_modules=[r_proj,k_proj,v_proj,o_proj,mlp.key,mlp.value,lm_head]')
else POLICY_LORA+=(trainer.policy.model.lora.rank=0); fi
if [[ "$CRITIC_USE_NORA" == true ]]; then
  CRITIC_LORA+=(trainer.critic.model.lora.rank="$CRITIC_LORA_RANK" trainer.critic.model.lora.alpha="$CRITIC_LORA_RANK" \
    'trainer.critic.model.lora.target_modules=[mlp.key,mlp.value]')
else CRITIC_LORA+=(trainer.critic.model.lora.rank=0); fi

run_rwkv_python -m skyrl.train.entrypoints.main_base \
  "data.train_data=['$TRAIN_DATA']" "data.val_data=['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator=gae_adaptive trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.loss_reduction=sequence_mean trainer.algorithm.use_kl_loss=false trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.gae_adaptive.alpha=1.5 trainer.algorithm.gae_adaptive.critic_lambd=1.0 \
  trainer.algorithm.eps_clip_low=0.3 trainer.algorithm.eps_clip_high=5.0 \
  trainer.policy.model.path="$MODEL_DIR" trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.critic.model.path="$CRITIC_MODEL_DIR" trainer.critic.resume_path=null trainer.critic.model_config_kwargs.wkv_mode=fp32io16 \
  "${POLICY_LORA[@]}" "${CRITIC_LORA[@]}" trainer.strategy=fsdp trainer.bf16=true \
  trainer.critic_update_epochs_per_batch=2 trainer.policy.optimizer_config.lr=1e-6 trainer.critic.optimizer_config.lr=5e-6 \
  trainer.policy.optimizer_config.offload_after_step=false trainer.critic.optimizer_config.offload_after_step=false \
  trainer.policy.fsdp_config.fsdp_size=1 trainer.critic.fsdp_config.fsdp_size=1 \
  trainer.policy.fsdp_config.reshard_after_forward=false trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.critic.fsdp_config.reshard_after_forward=false trainer.flash_attn=false trainer.remove_microbatch_padding=false \
  trainer.gradient_checkpointing=true trainer.fully_async.enabled="$ASYNC_ENABLED" \
  trainer.fully_async.mode="$ASYNC_MODE_CONFIG" trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.clear_kv_cache_on_weight_sync=true \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.placement.colocate_all="$COLOCATE_ALL" trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_gpus_per_node="$POLICY_GPUS" trainer.placement.critic_num_gpus_per_node="$CRITIC_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$POLICY_GPUS" \
  trainer.train_critic_and_policy_parallel="$TRAIN_CRITIC_AND_POLICY_PARALLEL" \
  trainer.epochs=2 trainer.max_training_steps="$MAX_STEPS" \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" \
  trainer.critic_mini_batch_size="$CRITIC_MINI_BATCH_SIZE" trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" trainer.max_tokens_per_microbatch="$MAX_TOKENS_PER_MICROBATCH" \
  trainer.max_prompt_length=24576 \
  trainer.eval_interval=50 trainer.ckpt_interval=50 trainer.hf_save_interval=50 \
  trainer.logger="$LOGGER" trainer.project_name=skyrl-rwkv trainer.run_name="$RUN_NAME" trainer.log_path="$OUTPUT_ROOT/logs" \
  trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.run_engines_locally=true generator.inference_engine.num_engines="$INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.enforce_eager=false generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
  generator.inference_engine.engine_init_kwargs.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32 \
  generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true \
  generator.batched=false generator.use_conversation_multi_turn=true generator.n_samples_per_prompt=1 \
  generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES_PER_PROMPT" generator.preserve_truncated_action_mask=true \
  generator.chat_template_kwargs='{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}' \
  generator.sampling_params.logprobs=1 generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH:-8192}" \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 generator.sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length="$EVAL_MAX_GENERATE_LENGTH" generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 generator.eval_sampling_params.top_k=32 \
  'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
  environment.env_class=ultradata "$@"
