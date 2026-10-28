#!/usr/bin/env bash

source "$(cd "$(dirname "$0")/.." && pwd)/launcher_common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
DATA_DIR="${DATA_DIR:-$HOME/data/rwkv-direct-sft}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/skyrl-rwkv-runs/rwkv7-g1k-sft}"
RUN_NAME="${RUN_NAME:-rwkv7-g1k-sft}"
NUM_GPUS="${NUM_GPUS:-8}"
TEST_MODE=false
SST=false
FORWARD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --test_mode|--test-mode) TEST_MODE=true ;;
    --sst) SST=true ;;
    *) FORWARD_ARGS+=("$arg") ;;
  esac
done
configure_wandb "$TEST_MODE"
if [[ "$TEST_MODE" == true ]]; then LOGGER=console; else LOGGER=wandb; fi

if [[ "$TEST_MODE" == true ]]; then
  TEST_DATA_DIR="${TEST_DATA_DIR:-$OUTPUT_ROOT/test-data}"
  run_rwkv_python "$ROOT/train/common/test_data.py" --directory "$DATA_DIR" --destination "$TEST_DATA_DIR" \
    --rows "${TEST_ROWS:-256}" --format parquet > /tmp/results_rwkv_sft_test_data.log 2>&1
  DATA_ROOT="$TEST_DATA_DIR"; MAX_STEPS="${MAX_STEPS:-5}"
  BATCH_SIZE="${BATCH_SIZE:-32}"; MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
else
  DATA_ROOT="$DATA_DIR"; MAX_STEPS="${MAX_STEPS:-null}"
  BATCH_SIZE="${BATCH_SIZE:-128}"; MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
fi

USE_NORA="${USE_NORA:-false}"
if [[ "$SST" == true && "$USE_NORA" == true ]]; then
  echo "--sst is mutually exclusive with USE_NORA=true; SST requires full-parameter SFT." >&2
  exit 2
fi
LORA_RANK="${LORA_RANK:-256}"
LORA_OVERRIDES=(model.lora.use_nora="$USE_NORA")
if [[ "$USE_NORA" == true ]]; then
  LORA_OVERRIDES+=(model.lora.rank="$LORA_RANK" model.lora.alpha="$LORA_RANK" \
    'model.lora.target_modules=[r_proj,k_proj,v_proj,o_proj,mlp.key,mlp.value,lm_head]')
else LORA_OVERRIDES+=(model.lora.rank=0); fi

# SFT's native eval hook starts eval/run.sh at every checkpoint interval. The
# hook supplies {global_step}; the evaluator starts one vLLM worker per GPU.
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-8}"
EVAL_ROOT="$OUTPUT_ROOT/evaluation"
EVAL_COMMAND="['env','MODEL=$OUTPUT_ROOT/hf_exports/global_step_{global_step}/policy','OUTPUT=$EVAL_ROOT/global_step_{global_step}','STEP={global_step}','NUM_GPUS=$EVAL_NUM_GPUS','bash','$ROOT/eval/run.sh']"

run_rwkv_training -m skyrl.train.main_sft \
  strategy=fsdp model.path="$MODEL_DIR" "${LORA_OVERRIDES[@]}" \
  "pretokenized_dataset_paths=['$DATA_ROOT/train']" "eval_pretokenized_dataset_paths=['$DATA_ROOT/eval']" \
  'eval_dataset_names=[rwkv]' max_length=null remove_microbatch_padding=false use_sequence_packing=false \
  'model_config_kwargs={wkv_mode: fp32io16}' rwkv_chunk_size="${RWKV_CHUNK_SIZE:-0}" \
  max_training_steps="$MAX_STEPS" \
  batch_size="$BATCH_SIZE" micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  placement.num_nodes=1 placement.num_gpus_per_node="$NUM_GPUS" fsdp_config.fsdp_size="$NUM_GPUS" fsdp_config.reshard_after_forward=false \
  fsdp_config.sync_gradients_each_microbatch=false optimizer_config.lr="${LEARNING_RATE:-1e-6}" \
  optimizer_config.max_grad_norm=1.0 optimizer_config.scheduler=cosine \
  optimizer_config.num_warmup_steps="${WARMUP_STEPS:-0}" optimizer_config.offload_after_step=false \
  sst="$SST" sst_rank="${SST_RANK:-0.99}" sst_refresh_interval="${SST_REFRESH_INTERVAL:-0}" \
  sampler=custom sampler_class_path=examples.train.rwkv.train.sft.prepare.LengthGroupedSampler \
  "sampler_kwargs={batch_size: $BATCH_SIZE, seed: 0}" seed=0 \
  dataloader_num_workers="$DATALOADER_NUM_WORKERS" dataloader_persistent_workers=true async_batch_collation=true \
  num_workers="${NUM_WORKERS:-0}" ckpt_path="$OUTPUT_ROOT/checkpoints" \
  export_path="$OUTPUT_ROOT/hf_exports" ckpt_interval=50 eval_interval=50 hf_save_interval=50 \
  eval_before_train=true max_ckpts_to_keep=2 logger="$LOGGER" project_name=skyrl-rwkv run_name="$RUN_NAME" \
  "eval_command=$EVAL_COMMAND" "eval_command_metrics_path=$EVAL_ROOT/global_step_{global_step}/eval_metrics.json" "${FORWARD_ARGS[@]}"
