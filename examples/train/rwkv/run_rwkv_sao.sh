#!/usr/bin/env bash

set -euo pipefail

# Fixed eight-card synchronous SAO launcher.  `test` prepares a small parquet
# prefix before starting the same policy/ref/critic placement and update path.
MODE="${1:-train}"
if [[ "$MODE" == "test" || "$MODE" == "train" ]]; then
  shift
else
  printf 'Usage: %s [train|test] [Hydra overrides...]\n' "$0" >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_DIR:=$HOME/Weights/RWKV/hf/rwkv-g1l-1b-temp-6331}"
: "${CRITIC_MODEL_DIR:=$MODEL_DIR}"
: "${DATA_DIR:=$HOME/data/ultradata-sao}"
: "${RUN_NAME:=rwkv-g1l-1b-sao-50step-8gpu}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
: "${LOGGER:=console}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8 NUM_POLICY_GPUS=8 NUM_INFERENCE_GPUS=8
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache/skyrl-rwkv}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}"
export UV_PROJECT_ENVIRONMENT="$(realpath .venv)"
export RAY_ADDRESS=local
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

if [[ "$MODE" == "test" ]]; then
  : "${TEST_TRAIN_SAMPLES:=256}"
  : "${TEST_VAL_SAMPLES:=64}"
  : "${TEST_STEPS:=2}"
  : "${TEST_MAX_PROMPT_LENGTH:=17408}"
  : "${TEST_MAX_GENERATE_LENGTH:=512}"
  : "${TEST_DATA_DIR:=$HOME/.cache/skyrl-rwkv/sao-test-${TEST_TRAIN_SAMPLES}-${TEST_VAL_SAMPLES}}"
  mkdir -p "$TEST_DATA_DIR"
  uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" - \
    "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet" "$TEST_DATA_DIR" \
    "$TEST_TRAIN_SAMPLES" "$TEST_VAL_SAMPLES" > /tmp/results_rwkv_sao_test_data.log 2>&1 <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def prefix(source: str, destination: Path, rows: int) -> None:
    parquet = pq.ParquetFile(source)
    batch = next(parquet.iter_batches(batch_size=rows))
    pq.write_table(pa.Table.from_batches([batch]), destination)


train_source, val_source, output_dir, train_rows, val_rows = sys.argv[1:]
output = Path(output_dir)
prefix(train_source, output / "train.parquet", int(train_rows))
prefix(val_source, output / "validation.parquet", int(val_rows))
PY
  TRAIN_DATA="$TEST_DATA_DIR/train.parquet"
  VAL_DATA="$TEST_DATA_DIR/validation.parquet"
  TRAIN_BATCH_SIZE=32
  POLICY_MINI_BATCH_SIZE=32
  CRITIC_MINI_BATCH_SIZE=16
  MICRO_BATCH_SIZE=2
  NUM_PARALLEL_GENERATION_WORKERS=128
  MAX_STEPS="$TEST_STEPS"
  MAX_PROMPT_LENGTH="$TEST_MAX_PROMPT_LENGTH"
  MAX_GENERATE_LENGTH="$TEST_MAX_GENERATE_LENGTH"
  RUN_NAME="${RUN_NAME}-test"
else
  TRAIN_DATA="$DATA_DIR/train.parquet"
  VAL_DATA="$DATA_DIR/validation.parquet"
  TRAIN_BATCH_SIZE=128
  POLICY_MINI_BATCH_SIZE=128
  CRITIC_MINI_BATCH_SIZE=64
  MICRO_BATCH_SIZE=2
  NUM_PARALLEL_GENERATION_WORKERS=512
  MAX_STEPS=50
  MAX_PROMPT_LENGTH=24576
  MAX_GENERATE_LENGTH=8192
fi

RUN_ROOT="${OUTPUT_ROOT%/*}/$RUN_NAME"
mkdir -p "$RUN_ROOT"/{logs,checkpoints,exports}
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD:$PWD/skyrl-gym${PYTHONPATH:+:$PYTHONPATH}"

uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" examples/train/rwkv/prepare_flashrwkv2.py --check \
  > /tmp/results_rwkv_sao_flashrwkv_check.log 2>&1

exec uv run --isolated --no-config --no-project "$UV_PROJECT_ENVIRONMENT/bin/python3" \
  -m skyrl.train.entrypoints.main_base \
  data.train_data="['$TRAIN_DATA']" \
  data.val_data="['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator=gae_adaptive \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.loss_reduction=sequence_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.gae_adaptive.alpha=1.5 \
  trainer.algorithm.gae_adaptive.critic_lambd=1.0 \
  trainer.algorithm.eps_clip_low=0.3 \
  trainer.algorithm.eps_clip_high=5.0 \
  trainer.policy.model.path="$MODEL_DIR" \
  trainer.policy.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.critic.model.path="$CRITIC_MODEL_DIR" \
  trainer.critic.resume_path=null \
  trainer.critic.model.lora.use_nora=true \
  trainer.critic.model.lora.rank=64 \
  trainer.critic.model.lora.alpha=64 \
  trainer.critic.model.lora.target_modules="['mlp.key','mlp.value']" \
  trainer.critic.model_config_kwargs.wkv_mode=fp32io16 \
  trainer.critic_update_epochs_per_batch=2 \
  trainer.strategy=fsdp \
  trainer.bf16=true \
  trainer.fully_async.enabled=true \
  trainer.fully_async.clear_kv_cache_on_weight_sync=true \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.fully_async.max_staleness_steps=4 \
  trainer.policy.fsdp_config.reshard_after_forward=false \
  trainer.policy.fsdp_config.fsdp_size=1 \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.offload_after_step=false \
  trainer.critic.fsdp_config.fsdp_size=1 \
  trainer.critic.optimizer_config.lr=5e-6 \
  trainer.critic.optimizer_config.offload_after_step=false \
  trainer.policy.fsdp_config.sync_gradients_each_microbatch=false \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.policy.sequence_parallel_size=1 \
  trainer.policy.use_torch_compile=false \
  trainer.gradient_checkpointing=true \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.placement.critic_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.train_critic_and_policy_parallel=true \
  trainer.epochs=2 \
  trainer.max_training_steps="$MAX_STEPS" \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE" \
  trainer.critic_mini_batch_size="$CRITIC_MINI_BATCH_SIZE" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.dump_train_results=true \
  trainer.num_logger_train_samples=20 \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=10 \
  trainer.max_prompt_length="$MAX_PROMPT_LENGTH" \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name=skyrl-rwkv \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="$RUN_ROOT/logs" \
  trainer.ckpt_path="$RUN_ROOT/checkpoints" \
  trainer.export_path="$RUN_ROOT/exports" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.engine_init_kwargs.logprobs_mode=raw_logprobs \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.engine_init_kwargs.kv_cache_memory_bytes=51539607552 \
  generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
  generator.inference_engine.engine_init_kwargs.max_num_batched_tokens=8192 \
  generator.batched=false \
  generator.preserve_truncated_action_mask=true \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=1 \
  generator.eval_n_samples_per_prompt=1 \
  generator.sampling_params.logprobs=1 \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.additional_kwargs="{presence_penalty: 0.0, frequency_penalty: 0.0, penalty_decay: 1.0}" \
  generator.chat_template_kwargs="{rwkv_prompt_template: bot, rwkv_generation_prompt: open_think}" \
  generator.eval_sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.eval_sampling_params.temperature=0.96 \
  generator.eval_sampling_params.top_p=0.76 \
  generator.eval_sampling_params.top_k=32 \
  generator.eval_sampling_params.additional_kwargs="{presence_penalty: 1.0, frequency_penalty: 0.1, penalty_decay: 0.988}" \
  generator.inference_engine.engine_init_kwargs.mamba_ssm_cache_dtype=float32 \
  generator.inference_engine.engine_init_kwargs.skip_tokenizer_init=true \
  trainer.max_tokens_per_microbatch=32768 \
  environment.env_class=ultradata \
  "$@"
