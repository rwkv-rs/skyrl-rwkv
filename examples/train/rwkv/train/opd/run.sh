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

TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="${VAL_DATA:-$DATA_DIR/validation.parquet}"
if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  prepare_test_parquet "$TRAIN_DATA" "$TEST_DATA_DIR/train.parquet" 128
  prepare_test_parquet "$VAL_DATA" "$TEST_DATA_DIR/validation.parquet" 32
  TRAIN_DATA="$TEST_DATA_DIR/train.parquet"; VAL_DATA="$TEST_DATA_DIR/validation.parquet"
  MAX_STEPS=5; TRAIN_BATCH_SIZE=16; POLICY_MINI_BATCH_SIZE=16; MICRO_BATCH_SIZE=1
else
  MAX_STEPS="${MAX_STEPS:-null}"; TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
  POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-512}"; MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
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

run_rwkv_python -m examples.train.on_policy_distillation.main_on_policy_distill \
  "data.train_data=['$TRAIN_DATA']" "data.val_data=['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator=no_op trainer.algorithm.policy_loss_type=importance_sampling \
  trainer.algorithm.loss_reduction=token_mean trainer.algorithm.use_kl_in_reward=true trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$STUDENT_MODEL" trainer.ref.model.path="$TEACHER_MODEL" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 trainer.ref.model_config_kwargs.wkv_mode=fp32io16 \
  "${LORA_OVERRIDES[@]}" trainer.strategy=fsdp trainer.bf16=true trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false trainer.gradient_checkpointing=true \
  trainer.policy.fsdp_config.reshard_after_forward=false trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.ref.fsdp_config.fsdp_size=8 trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=8 trainer.placement.ref_num_gpus_per_node=8 \
  trainer.epochs=1 trainer.max_training_steps="$MAX_STEPS" trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" trainer.eval_interval=50 trainer.ckpt_interval=50 \
  trainer.hf_save_interval=50 trainer.eval_before_train=true trainer.logger="$LOGGER" \
  trainer.project_name=skyrl-rwkv trainer.run_name="$RUN_NAME" trainer.log_path="$OUTPUT_ROOT/logs" \
  trainer.ckpt_path="$OUTPUT_ROOT/checkpoints" trainer.export_path="$OUTPUT_ROOT/exports" \
  generator.inference_engine.backend=vllm generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.run_engines_locally=true generator.inference_engine.num_engines=8 \
  generator.inference_engine.tensor_parallel_size=1 generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.batched=false generator.n_samples_per_prompt=1 generator.eval_n_samples_per_prompt=128 \
  generator.chat_template_kwargs='{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}' \
  generator.sampling_params.logprobs=1 generator.sampling_params.max_generate_length="${MAX_GENERATE_LENGTH:-8192}" \
  generator.sampling_params.temperature=1.0 generator.sampling_params.top_p=1.0 generator.sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length=25600 generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 generator.eval_sampling_params.top_k=32 \
  'generator.eval_sampling_params.additional_kwargs={presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}' \
  environment.env_class=aime "$@"
