#!/usr/bin/env bash

source "$(cd "$(dirname "$0")/.." && pwd)/launcher_common.sh"

STUDENT_MODEL="${STUDENT_MODEL:-$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
TEACHER_MODEL="${TEACHER_MODEL:-$HOME/Weights/RWKV/hf/rwkv7-g1k-13.3b-20260930-ctx25600-antidoom-nora}"
DATA_DIR="${DATA_DIR:-$HOME/data/ultradata-opd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/rwkv7-g1k-opd}"
RUN_NAME="${RUN_NAME:-rwkv7-g1k-opd}"
TEST_MODE=false
if [[ "${1:-}" == "--test_mode" || "${1:-}" == "--test-mode" ]]; then TEST_MODE=true; shift; fi
configure_wandb "$TEST_MODE"
if [[ "$TEST_MODE" == true ]]; then LOGGER=console; else LOGGER=wandb; fi

# OPD benefits from the same producer/learner overlap as the reference
# implementation.  Use ASYNC_MODE=disabled to run the colocated synchronous
# baseline, or ASYNC_MODE=zero_stale for a one-step-off-policy comparison.
ASYNC_MODE="${ASYNC_MODE:-fully_async}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-4}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
if [[ "$ASYNC_MODE" == disabled ]]; then
  ASYNC_ENABLED=false
  ASYNC_OPD=false
  ASYNC_MODE_CONFIG=fully_async
  COLOCATE_ALL=true
  POLICY_GPUS="${POLICY_GPUS:-8}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-8}"
elif [[ "$ASYNC_MODE" == zero_stale ]]; then
  ASYNC_ENABLED=true
  ASYNC_OPD=false
  ASYNC_MODE_CONFIG=zero_stale
  COLOCATE_ALL=false
  POLICY_GPUS="${POLICY_GPUS:-4}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-4}"
else
  ASYNC_ENABLED=true
  ASYNC_OPD=true
  ASYNC_MODE_CONFIG="$ASYNC_MODE"
  COLOCATE_ALL=false
  # OPD carries a 13.3B teacher on the policy/ref side.  Six learner GPUs
  # leave two dedicated inference GPUs, which keeps the initial weight-sync
  # fan-out small enough to start reliably on an eight-GPU node.
  POLICY_GPUS="${POLICY_GPUS:-6}"
  INFERENCE_GPUS="${INFERENCE_GPUS:-2}"
fi
if [[ "$ASYNC_OPD" == true ]]; then
  POLICY_LOSS_TYPE=async_opd
else
  POLICY_LOSS_TYPE=importance_sampling
fi

TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="${VAL_DATA:-$DATA_DIR/validation.parquet}"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  prepare_test_parquet "$TRAIN_DATA" "$TEST_DATA_DIR/train.parquet" 320
  prepare_test_parquet "$VAL_DATA" "$TEST_DATA_DIR/validation.parquet" 32
  TRAIN_DATA="$TEST_DATA_DIR/train.parquet"; VAL_DATA="$TEST_DATA_DIR/validation.parquet"
  MAX_STEPS="${MAX_STEPS:-5}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
  POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-32}"
  if [[ "$ASYNC_MODE" == disabled ]]; then
    # Colocated policy/ref/inference shares the device allocator; one
    # microbatch is the stable setting on the eight-way Pro6000 node.
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
  else
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
  fi
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
  EVAL_MAX_GENERATE_LENGTH="${EVAL_MAX_GENERATE_LENGTH:-1024}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
else
  MAX_STEPS="${MAX_STEPS:-null}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
  POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-512}"; MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
  EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-32}"
  EVAL_MAX_GENERATE_LENGTH="${EVAL_MAX_GENERATE_LENGTH:-25600}"
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
fi
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-32768}"
if [[ "$ASYNC_ENABLED" == true ]]; then
  # Bound request fan-out by engine count.  Using the whole stale window here
  # creates a large queue and increases tail latency without improving token
  # throughput on long RWKV generations.
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
  NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-$POLICY_MINI_BATCH_SIZE}"
fi
REF_FSDP_SIZE="${REF_FSDP_SIZE:-$POLICY_GPUS}"
WEIGHT_SYNC_BACKEND="${WEIGHT_SYNC_BACKEND:-nccl}"
WEIGHT_SYNC_ARGS=(generator.inference_engine.weight_sync_backend="$WEIGHT_SYNC_BACKEND")
if [[ "$WEIGHT_SYNC_BACKEND" == delta ]]; then
  DELTA_SYNC_DIR="${DELTA_SYNC_DIR:-$OUTPUT_ROOT/delta_sync}"
  DELTA_LOCAL_CHECKPOINT_DIR="${DELTA_LOCAL_CHECKPOINT_DIR:-$OUTPUT_ROOT/delta_checkpoints}"
  DELTA_PUBLISH_STAGING_DIR="${DELTA_PUBLISH_STAGING_DIR:-$OUTPUT_ROOT/delta_staging}"
  WEIGHT_SYNC_ARGS+=(
    generator.inference_engine.delta_weight_sync.sync_dir="$DELTA_SYNC_DIR"
    generator.inference_engine.delta_weight_sync.local_checkpoint_dir="$DELTA_LOCAL_CHECKPOINT_DIR"
    generator.inference_engine.delta_weight_sync.publish_staging_dir="$DELTA_PUBLISH_STAGING_DIR"
  )
fi

USE_NORA="${USE_NORA:-true}"
LORA_RANK="${LORA_RANK:-256}"
LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
LORA_OVERRIDES=(trainer.policy.model.lora.use_nora="$USE_NORA")
if [[ "$USE_NORA" == true ]]; then
  LORA_OVERRIDES+=(trainer.policy.model.lora.rank="$LORA_RANK" trainer.policy.model.lora.alpha="$LORA_ALPHA" \
    'trainer.policy.model.lora.target_modules=[r_proj,k_proj,v_proj,o_proj,mlp.key,mlp.value,lm_head]')
else
  LORA_OVERRIDES+=(trainer.policy.model.lora.rank=0)
fi

run_rwkv_training -m examples.train.on_policy_distillation.main_on_policy_distill \
  "data.train_data=['$TRAIN_DATA']" "data.val_data=['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator=no_op trainer.algorithm.async_opd="$ASYNC_OPD" \
  trainer.algorithm.policy_loss_type="${POLICY_LOSS_TYPE:-importance_sampling}" \
  trainer.algorithm.loss_reduction=token_mean trainer.algorithm.use_kl_in_reward=true trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$STUDENT_MODEL" trainer.ref.model.path="$TEACHER_MODEL" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 trainer.ref.model_config_kwargs.wkv_mode=fp32io16 \
  "${LORA_OVERRIDES[@]}" trainer.strategy=fsdp trainer.bf16=true trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false trainer.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  trainer.policy.optimizer_config.lr="${LEARNING_RATE:-2e-6}" trainer.policy.optimizer_config.offload_after_step=false \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.policy.fsdp_config.reshard_after_forward=false trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.ref.fsdp_config.fsdp_size="$REF_FSDP_SIZE" \
  trainer.fully_async.enabled="$ASYNC_ENABLED" trainer.fully_async.mode="$ASYNC_MODE_CONFIG" \
  trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.fully_async.clear_kv_cache_on_weight_sync=true \
  trainer.placement.colocate_all="$COLOCATE_ALL" trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_gpus_per_node="$POLICY_GPUS" trainer.placement.ref_num_gpus_per_node="$POLICY_GPUS" \
  trainer.epochs=1 trainer.max_training_steps="$MAX_STEPS" trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" trainer.max_tokens_per_microbatch="$MAX_TOKENS_PER_MICROBATCH" \
  trainer.max_prompt_length=17408 \
  trainer.eval_interval=50 trainer.ckpt_interval=50 \
  trainer.hf_save_interval=50 trainer.eval_before_train=true trainer.logger="$LOGGER" \
  trainer.project_name=skyrl-rwkv trainer.run_name="$RUN_NAME" trainer.log_path="$OUTPUT_ROOT/logs" \
  trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.run_engines_locally=true generator.inference_engine.num_engines="$INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 "${WEIGHT_SYNC_ARGS[@]}" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" generator.inference_engine.max_num_seqs="$MAX_NUM_SEQS" \
  generator.inference_engine.enforce_eager=false generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.engine_init_kwargs.kv_cache_memory_bytes="$KV_CACHE_MEMORY_BYTES" \
  generator.inference_engine.engine_init_kwargs.max_model_len=25600 \
  generator.inference_engine.engine_init_kwargs.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  generator.inference_engine.engine_init_kwargs.compilation_config='{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32 \
  generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true \
  generator.batched=false generator.n_samples_per_prompt=1 generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES_PER_PROMPT" \
  generator.chat_template_kwargs='{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}' \
  generator.sampling_params.logprobs=1 generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH:-8192}" \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 generator.sampling_params.top_k=-1 \
  'generator.sampling_params.additional_kwargs={presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}' \
  generator.eval_sampling_params.max_generate_length="$EVAL_MAX_GENERATE_LENGTH" generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 generator.eval_sampling_params.top_k=32 \
  'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
  environment.env_class=aime "$@"
