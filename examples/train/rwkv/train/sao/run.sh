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

TRAIN_DATA="$DATA_DIR/train.parquet"; VAL_DATA="$DATA_DIR/validation.parquet"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  prepare_test_parquet "$TRAIN_DATA" "$TEST_DATA_DIR/train.parquet" 128
  prepare_test_parquet "$VAL_DATA" "$TEST_DATA_DIR/validation.parquet" 32
  TRAIN_DATA="$TEST_DATA_DIR/train.parquet"; VAL_DATA="$TEST_DATA_DIR/validation.parquet"
  MAX_STEPS=5; TRAIN_BATCH_SIZE=16; POLICY_MINI_BATCH_SIZE=16; CRITIC_MINI_BATCH_SIZE=8; MICRO_BATCH_SIZE=1
else
  MAX_STEPS="${MAX_STEPS:-50}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
  POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-128}"; CRITIC_MINI_BATCH_SIZE="${CRITIC_MINI_BATCH_SIZE:-64}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
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
  trainer.gradient_checkpointing=true trainer.fully_async.enabled=true trainer.fully_async.max_staleness_steps=4 \
  trainer.fully_async.clear_kv_cache_on_weight_sync=true trainer.fully_async.num_parallel_generation_workers=512 \
  trainer.placement.colocate_all=true trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_gpus_per_node=8 trainer.placement.critic_num_gpus_per_node=8 trainer.placement.ref_num_gpus_per_node=8 \
  trainer.train_critic_and_policy_parallel=true trainer.epochs=2 trainer.max_training_steps="$MAX_STEPS" \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" \
  trainer.critic_mini_batch_size="$CRITIC_MINI_BATCH_SIZE" trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" trainer.max_tokens_per_microbatch=32768 \
  trainer.eval_interval=50 trainer.ckpt_interval=50 trainer.hf_save_interval=50 \
  trainer.logger="$LOGGER" trainer.project_name=skyrl-rwkv trainer.run_name="$RUN_NAME" trainer.log_path="$OUTPUT_ROOT/logs" \
  trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.run_engines_locally=true generator.inference_engine.num_engines=8 \
  generator.inference_engine.tensor_parallel_size=1 generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
  generator.batched=false generator.use_conversation_multi_turn=true generator.n_samples_per_prompt=1 \
  generator.eval_n_samples_per_prompt=128 generator.preserve_truncated_action_mask=true \
  generator.chat_template_kwargs='{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}' \
  generator.sampling_params.logprobs=1 generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH:-8192}" \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 generator.sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length=25600 generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 generator.eval_sampling_params.top_k=32 \
  'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
  environment.env_class=ultradata "$@"
