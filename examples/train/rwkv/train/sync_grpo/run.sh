#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../launcher_common.sh"
: "${MODEL_DIR:?Set MODEL_DIR to an exact RWKV7 Hugging Face checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the prepared filtered Parquet directory}"
RUN_NAME="${RUN_NAME:-rwkv7-sync-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/$RUN_NAME}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/skyrl-rwkv-artifacts/$RUN_NAME}"
mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/exports" "$OUTPUT_ROOT/checkpoints"
exec > >(tee -a "$ARTIFACT_ROOT/console.log") 2>&1
exec 9>"$OUTPUT_ROOT/.training.lock"
flock -n 9 || { echo "Another launcher owns $OUTPUT_ROOT"; exit 1; }

export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE="${SKYRL_GENERATE_CONCURRENCY_PER_ENGINE:-2048}"
export SKYRL_ROLLOUT_ARCHIVE="${SKYRL_ROLLOUT_ARCHIVE:-$ARTIFACT_ROOT/rollouts}"
export SKYRL_MATH_ARCHIVE="${SKYRL_MATH_ARCHIVE:-$ARTIFACT_ROOT/scoring}"
export SKYRL_ISOLATED_MATH_SCORING="${SKYRL_ISOLATED_MATH_SCORING:-1}"
export SKYRL_MATH_WORKERS="${SKYRL_MATH_WORKERS:-8}"
export SKYRL_MATH_TIMEOUT_SECONDS="${SKYRL_MATH_TIMEOUT_SECONDS:-3}"
export SKYRL_COMPACT_ROLLOUT_LOGPROBS="${SKYRL_COMPACT_ROLLOUT_LOGPROBS:-1}"
export SKYRL_RWKV_LENGTH_BUCKET_SIZE="${SKYRL_RWKV_LENGTH_BUCKET_SIZE:-512}"
export SKYRL_RWKV_TRANSPOSED_GEMM_BACKEND="${SKYRL_RWKV_TRANSPOSED_GEMM_BACKEND:-cublas}"
export SKYRL_RWKV_PREFETCH="${SKYRL_RWKV_PREFETCH:-1}"
export SKYRL_BALANCED_DP_PACKING="${SKYRL_BALANCED_DP_PACKING:-1}"
export SKYRL_DEFER_LOSS_METRICS="${SKYRL_DEFER_LOSS_METRICS:-1}"
export SKYRL_ROLLOUT_LOGPROB_DIFF_METRICS="${SKYRL_ROLLOUT_LOGPROB_DIFF_METRICS:-0}"
export SKYRL_POLICY_ENTROPY_METRICS="${SKYRL_POLICY_ENTROPY_METRICS:-0}"

RESUME_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  RESUME_ARGS+=(trainer.resume_mode=from_path "trainer.resume_path=$RESUME_PATH")
fi

run_rwkv_python -m skyrl.train.entrypoints.main_base \
  "data.train_data=['$DATA_DIR/train.parquet']" "data.val_data=['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.policy_loss_type=regular \
  trainer.algorithm.loss_reduction=token_mean trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.eps_clip_low=0.2 trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.grpo_norm_by_std=true trainer.algorithm.github_strict_loss=true \
  trainer.algorithm.old_logprobs_source=rollout \
  trainer.algorithm.dynamic_sampling.type=filter trainer.algorithm.dynamic_sampling.max_sample_batches=10 \
  trainer.algorithm.dynamic_sampling.pool_mode=true \
  "trainer.algorithm.dynamic_sampling.pool_prompts=${POOL_PROMPTS:-2048}" \
  trainer.algorithm.dynamic_sampling.pool_spill=true \
  trainer.algorithm.dynamic_sampling.keep_all_informative_groups=false \
  trainer.algorithm.overlong_buffer_len=2048 trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
  trainer.algorithm.zero_variance_filter=true \
  "trainer.policy.model.path=$MODEL_DIR" trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.policy.model.lora.rank=0 trainer.policy.model.lora.use_nora=false \
  trainer.strategy=ddp trainer.bf16=true trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false trainer.gradient_checkpointing=true \
  "trainer.policy.optimizer_config.lr=${LEARNING_RATE:-2e-7}" \
  trainer.policy.optimizer_config.scheduler=constant trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.offload_after_step=false trainer.policy.keep_model_on_gpu=false \
  trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.fully_async.enabled=false trainer.placement.colocate_all=true trainer.placement.colocate_policy_ref=true \
  "trainer.placement.policy_num_gpus_per_node=$NUM_GPUS" "trainer.placement.ref_num_gpus_per_node=$NUM_GPUS" \
  trainer.epochs=50 "trainer.max_training_steps=${MAX_STEPS:-200}" \
  trainer.train_batch_size=512 trainer.policy_mini_batch_size=32 \
  trainer.micro_train_batch_size_per_gpu=4 trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.max_tokens_per_microbatch=-1 trainer.max_prompt_length=1024 \
  trainer.eval_interval=20 trainer.ckpt_interval=1 trainer.max_ckpts_to_keep=3 trainer.hf_save_interval=20 \
  trainer.eval_before_train=true "trainer.logger=${LOGGER:-console}" \
  trainer.project_name=skyrl-rwkv "trainer.run_name=$RUN_NAME" \
  "trainer.log_path=$ARTIFACT_ROOT/logs" "trainer.export_path=$ARTIFACT_ROOT/exports" \
  "trainer.ckpt_path=$OUTPUT_ROOT/checkpoints" \
  generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.run_engines_locally=true "generator.inference_engine.num_engines=$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  "generator.inference_engine.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.30}" \
  "generator.inference_engine.max_num_seqs=${MAX_NUM_SEQS:-1024}" generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.engine_init_kwargs.max_num_batched_tokens=65536 \
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32 \
  generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true \
  'generator.inference_engine.engine_init_kwargs.compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,384,512,640,768,896,1024],"max_cudagraph_capture_size":1024,"cudagraph_num_of_warmups":1}' \
  generator.batched=false generator.preserve_truncated_action_mask=true \
  generator.n_samples_per_prompt=16 generator.eval_n_samples_per_prompt=1 \
  generator.sampling_params.logprobs=0 generator.sampling_params.max_generate_length=16383 \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 generator.sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length=16383 generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 generator.eval_sampling_params.top_k=32 \
  'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
  environment.env_class=gsm8k environment.skyrl_gym.gsm8k.strict_reward=true "${RESUME_ARGS[@]}" "$@"
