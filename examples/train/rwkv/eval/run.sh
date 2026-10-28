#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/../../.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python3}"
MODEL="${MODEL:?Set MODEL to the HF RWKV checkpoint to evaluate}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
OUTPUT="${OUTPUT:?Set OUTPUT to the checkpoint evaluation directory}"
STEP="${STEP:-0}"
NUM_GPUS="${NUM_GPUS:-8}"
[[ "$NUM_GPUS" -eq 8 ]] || { echo 'RWKV evaluation requires NUM_GPUS=8' >&2; exit 2; }
PREPARE_FLASHRWKV2=false
TEST_MODE=false
case "${1:-}" in
  --prepare-flashrwkv2) PREPARE_FLASHRWKV2=true; shift ;;
  --test_mode|--test-mode) TEST_MODE=true; shift ;;
esac
[[ $# -eq 0 ]] || { echo "unexpected arguments: $*" >&2; exit 2; }
if [[ "$TEST_MODE" == true || "$PREPARE_FLASHRWKV2" == true ]]; then
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
else
  unset WANDB_DISABLED
  export WANDB_MODE=online
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export FLASHRWKV2_CACHE_DIR="${FLASHRWKV2_CACHE_DIR:-$HOME/.cache/flashrwkv2}"
export VLLM_RWKV_TUNING_MODE=require
STARTUP_SECONDS=900
if [[ "$PREPARE_FLASHRWKV2" == true ]]; then
  export VLLM_RWKV_TUNING_MODE=tune
  STARTUP_SECONDS=7200
fi

BASE_PORT="${BASE_PORT:-19100}"
ROUTER_PORT="${ROUTER_PORT:-19200}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PIDS=()
URLS=()
mkdir -p "$OUTPUT"
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  local deadline=$((SECONDS + 30)) running
  while (( SECONDS < deadline )); do
    running=false
    for pid in "${PIDS[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then running=true; fi
    done
    [[ "$running" == true ]] || break
    sleep 1
  done
  for pid in "${PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_servers() {
  local gpu
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    kill -0 "${PIDS[$gpu]}" 2>/dev/null || {
      echo "inference worker exited on GPU $gpu; see $OUTPUT/inference-gpu$gpu.log" >&2
      return 1
    }
  done
  kill -0 "${PIDS[$NUM_GPUS]}" 2>/dev/null || {
    echo "router exited; see $OUTPUT/router.log" >&2
    return 1
  }
}

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  port=$((BASE_PORT + gpu))
  URLS+=("http://127.0.0.1:$port")
  setsid env CUDA_VISIBLE_DEVICES="$gpu" uv run --isolated --no-config --no-project "$PYTHON" \
    -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
    --mamba-ssm-cache-dtype float32 \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 32768 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enable-prefix-caching \
    --enable-chunked-prefill --no-enable-log-requests > "$OUTPUT/inference-gpu$gpu.log" 2>&1 &
  PIDS+=("$!")
done

ROUTER_ARGS=(--host 127.0.0.1 --port "$ROUTER_PORT" --policy round_robin \
  --worker-startup-timeout-secs "$STARTUP_SECONDS" --worker-urls "${URLS[@]}")
setsid uv run --isolated --no-config --no-project "$PYTHON" -m vllm_router.launch_router \
  "${ROUTER_ARGS[@]}" \
  > "$OUTPUT/router.log" 2>&1 &
PIDS+=("$!")

deadline=$((SECONDS + STARTUP_SECONDS))
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  port=$((BASE_PORT + gpu))
  while true; do
    check_servers
    if curl --fail --silent --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then break; fi
    (( SECONDS >= deadline )) && { echo "inference startup timed out on port $port" >&2; exit 1; }
    sleep 2
  done
done

while true; do
  check_servers
  if curl --fail --silent --max-time 2 "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null; then break; fi
  (( SECONDS >= deadline )) && { echo "router startup timed out on port $ROUTER_PORT" >&2; exit 1; }
  sleep 2
done
check_servers
if [[ "$PREPARE_FLASHRWKV2" == true ]]; then
  echo "FlashRWKV2 profiles prepared for all $NUM_GPUS inference workers"
  exit 0
fi
BASE_URL="http://127.0.0.1:$ROUTER_PORT"
BASE_URL="$BASE_URL,$BASE_URL,$BASE_URL,$BASE_URL,$BASE_URL,$BASE_URL,$BASE_URL,$BASE_URL"
ARGS=(--model "$MODEL" --base-url "$BASE_URL" --output "$OUTPUT" --step "$STEP" \
  --served-model-name "$SERVED_MODEL_NAME" --seed "${SEED:-42}" \
  --context-length "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --prompt-template "${PROMPT_TEMPLATE:-bot}" \
  --lighteval-project "${LIGHTEVAL_PROJECT:-$HOME/Projects/MachineLearning/lighteval-rwkv}")
[[ -z "${BENCHMARKS:-}" ]] || ARGS+=(--benchmarks "$BENCHMARKS")
[[ -z "${MAX_SAMPLES:-}" ]] || ARGS+=(--max-samples "$MAX_SAMPLES")
if [[ "$TEST_MODE" == true ]]; then
  ARGS+=(--test_mode)
fi
uv run --isolated --no-config --no-project "$PYTHON" "$ROOT/eval/prepare.py" "${ARGS[@]}"
