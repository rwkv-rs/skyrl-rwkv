#!/usr/bin/env bash
# Full-parameter RWKV SFT on complete, pretokenized Nemotron CoT.
# Prepare data with examples.train.rwkv.prepare_sft before running this script.
# Usage from the repository root:
#   DATA_DIR=<prepared-data> OUTPUT_ROOT=<run-directory> NUM_GPUS=8 \
#     bash examples/train/rwkv/run_rwkv_sft.sh

set -euo pipefail

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_DIR:=$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600}"
: "${DATA_DIR:=$HOME/data/rwkv-direct-sft}"
: "${NUM_GPUS:=8}"
: "${BATCH_SIZE:=$((NUM_GPUS * 8))}"
: "${MICRO_BATCH_SIZE:=1}"
: "${CHUNK_SIZE:=2048}"
: "${LEARNING_RATE:=1e-6}"
: "${MIN_LR_RATE:=0.1}"
: "${LOGGER:=wandb}"
: "${RUN_NAME:=rwkv7-g1k-1.5b-20260930-ctx25600-nemotron-cot-sft}"
: "${OUTPUT_ROOT:=$HOME/skyrl-rwkv-runs/$RUN_NAME}"
: "${EVAL_ROOT:=$OUTPUT_ROOT/evaluation/lighteval_avg16}"
export MODEL_DIR
export RAY_ADDRESS=local
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export SKYRL_PYTHONPATH_EXPORT=1
export XDG_CACHE_HOME="$HOME/.cache/skyrl-rwkv"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
if [[ "${UPLOAD_ONLY:-0}" != 1 ]]; then
  exec > >(tee -a "$OUTPUT_ROOT/train.log") 2>&1
fi

if [[ "${UPLOAD_ONLY:-0}" == 1 ]]; then
  unset WANDB_RUN_ID WANDB_RESUME WANDB_SERVICE
  uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
    - "$EVAL_ROOT" "$EVAL_STEP" "$UPLOAD_PHASE" "$WANDB_RUN_PATH" <<'PY'
import fcntl
import json
import os
import sys
from pathlib import Path

import wandb

evaluation_root, step, phase, run_path = sys.argv[1:]
evaluation_root = Path(evaluation_root)
root = evaluation_root / (f"global_step_{step}/sft" if phase == "sft" else "base")
status = root / "upload_status.json"
with (evaluation_root.parent / ".wandb.upload.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    status.write_text(json.dumps({"state": "uploading", "global_step": int(step), "pid": os.getpid()}) + "\n")
    try:
        scores = {path.parent.name: json.loads(path.read_text()) for path in root.glob("*/scores.json")}
        parent = wandb.Api().run(run_path)
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".lighteval_cache" not in path.parts and path.name not in (
                "upload.log", "upload.pid", "upload_status.json"
            ):
                parent.upload_file(str(path), root=str(evaluation_root.parent))
        # During training, the native eval_command hook owns metric logging.
        # Standalone evaluations append to that same run after training stops.
        if parent.state in ("finished", "failed", "crashed", "killed"):
            entity, project, run_id = run_path.split("/")
            run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
            try:
                values = {f"eval/{phase}/{selector}/{key}": value for selector, rows in scores.items()
                          for row in rows.values() for key, value in row.items()}
                run.define_metric("eval/checkpoint_step", hidden=True)
                run.define_metric(f"eval/{phase}/*", step_metric="eval/checkpoint_step")
                run.log({**values, "eval/checkpoint_step": 0 if phase == "base" else int(step)})
                run.summary["eval/source_run_state"] = parent.state
            finally:
                run.finish(exit_code=0 if parent.state == "finished" else 1)
    except Exception as error:
        status.write_text(json.dumps({"state": "failed", "global_step": int(step),
                                      "pid": os.getpid(), "error": str(error)}) + "\n")
        raise
    status.write_text(json.dumps({"state": "completed", "global_step": int(step), "pid": os.getpid()}) + "\n")
PY
  exit 0
fi

TOTAL_STEPS=$(uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" - "$DATA_DIR" "$BATCH_SIZE" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
print(math.ceil(manifest["counts"]["train_rows"] / int(sys.argv[2])))
PY
)
WARMUP_STEPS=$((TOTAL_STEPS / 20))
EVAL_OVERRIDES=()
if [[ -n "${LIGHTEVAL_PROJECT:-}" && "${EVAL_ONLY:-0}" != 1 ]]; then
  : "${WANDB_RUN_PATH:?Set the training run path: entity/project/run_id}"
  EVAL_COMMAND=$(uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
    - "$OUTPUT_ROOT" "$DATA_DIR" "$LIGHTEVAL_PROJECT" "$WANDB_RUN_PATH" <<'PY'
import json
import sys

output, data, producer, run = sys.argv[1:]
print(json.dumps(["env", "EVAL_ONLY=1", "EVAL_STEP={global_step}",
                  f"OUTPUT_ROOT={output}", f"DATA_DIR={data}",
                  f"LIGHTEVAL_PROJECT={producer}", f"WANDB_RUN_PATH={run}",
                  "NUM_GPUS=8", "BATCH_SIZE=56", "EVAL_MAX_NUM_SEQS=1024",
                  "EVAL_GPU_MEMORY_UTILIZATION=0.65",
                  "bash", "examples/train/rwkv/run_rwkv_sft.sh"]))
PY
)
  EVAL_OVERRIDES+=("eval_command=$EVAL_COMMAND"
    "eval_command_metrics_path=$OUTPUT_ROOT/evaluation/lighteval_avg16/global_step_{global_step}/sft/eval_metrics.json")
fi

if [[ "${EVAL_ONLY:-0}" != 1 ]]; then
  uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" -m skyrl.train.main_sft \
    strategy=fsdp model.path="$MODEL_DIR" \
    "pretokenized_dataset_paths=['$DATA_DIR/train']" \
    "eval_pretokenized_dataset_paths=['$DATA_DIR/eval']" \
    'eval_dataset_names=[nemotron_cot]' \
    max_length=null remove_microbatch_padding=false use_sequence_packing=false \
    rwkv_chunk_size="$CHUNK_SIZE" 'model_config_kwargs={wkv_mode: fp32io16}' \
    num_epochs=1 max_training_steps="$TOTAL_STEPS" \
    batch_size="$BATCH_SIZE" micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    placement.num_nodes=1 placement.num_gpus_per_node="$NUM_GPUS" \
    fsdp_config.fsdp_size="$NUM_GPUS" fsdp_config.reshard_after_forward=false \
    fsdp_config.sync_gradients_each_microbatch=false \
    optimizer_config.lr="$LEARNING_RATE" optimizer_config.max_grad_norm=1.0 \
    optimizer_config.scheduler=cosine_with_min_lr \
    "optimizer_config.scheduler_specific_kwargs={min_lr_rate: $MIN_LR_RATE}" \
    optimizer_config.num_warmup_steps="$WARMUP_STEPS" \
    optimizer_config.offload_after_step=false \
    sampler=custom sampler_class_path=examples.train.rwkv.prepare_sft.LengthGroupedSampler \
    "sampler_kwargs={batch_size: $BATCH_SIZE, seed: 0}" seed=0 \
    dataloader_num_workers=2 dataloader_persistent_workers=true num_workers=0 \
    ckpt_path="$OUTPUT_ROOT/checkpoints" export_path="$OUTPUT_ROOT/hf_exports" \
    ckpt_interval=50 eval_interval=50 hf_save_interval=50 \
    eval_before_train=true max_ckpts_to_keep=2 \
    logger="$LOGGER" project_name=skyrl-rwkv run_name="$RUN_NAME" \
    "${EVAL_OVERRIDES[@]}" \
    "$@"
fi

# Invoke the existing LightEval producer without changing its checkout.
if [[ -n "${LIGHTEVAL_PROJECT:-}" && "${EVAL_ONLY:-0}" == 1 ]]; then
  : "${WANDB_RUN_PATH:?Set the training run path: entity/project/run_id}"
  : "${EVAL_MAX_NUM_SEQS:=1024}"
  : "${EVAL_GPU_MEMORY_UTILIZATION:=0.65}"
  : "${EVAL_STEP:=$TOTAL_STEPS}"
  : "${BENCHMARK_INTERVAL:=500}"
  if (( EVAL_STEP % BENCHMARK_INTERVAL != 0 && EVAL_STEP != TOTAL_STEPS )); then
    mkdir -p "$EVAL_ROOT/global_step_$EVAL_STEP/sft"
    printf '{}\n' > "$EVAL_ROOT/global_step_$EVAL_STEP/sft/eval_metrics.json"
    echo "Skipping benchmark at step $EVAL_STEP (interval=$BENCHMARK_INTERVAL)."
    exit 0
  fi
  SKYRL_PROJECT="$PWD"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
  read -r -a EVAL_PHASES <<< "${EVAL_PHASES:-base sft}"
  if [[ -f "$EVAL_ROOT/base/completed.json" ]]; then
    EVAL_PHASES=(sft)
  fi
  GPUS_PER_PHASE=$((NUM_GPUS / ${#EVAL_PHASES[@]}))
  EVAL_PIDS=()
  cleanup_eval() {
    for pid in "${EVAL_PIDS[@]}"; do
      kill -- "-$pid" 2>/dev/null || true
    done
    for pid in "${EVAL_PIDS[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  }
  trap cleanup_eval EXIT
  trap 'exit 130' INT TERM

  phase_index=0
  for phase in "${EVAL_PHASES[@]}"; do
    PHASE_ROOT="$EVAL_ROOT/$phase"
    EVAL_MODEL="$MODEL_DIR"
    EVAL_NAME="$(basename "$MODEL_DIR")"
    FIRST_GPU=$((phase_index * GPUS_PER_PHASE))
    ROUTER_PORT=$((19200 + phase_index))
    if [[ "$phase" == sft ]]; then
      PHASE_ROOT="$EVAL_ROOT/global_step_$EVAL_STEP/sft"
      EVAL_MODEL="$OUTPUT_ROOT/hf_exports/global_step_$EVAL_STEP/policy"
      EVAL_NAME="$EVAL_NAME-nemotron-cot-sft-step$EVAL_STEP"
      for config_name in fake_think_generation_config.json tools_generation_config.json; do
        cp "$MODEL_DIR/$config_name" "$EVAL_MODEL/$config_name"
      done
    fi
    mkdir -p "$PHASE_ROOT"
    WORKER_URLS=()
    for ((gpu=FIRST_GPU; gpu<FIRST_GPU+GPUS_PER_PHASE; gpu++)); do
      port=$((19100 + gpu))
      WORKER_URLS+=("http://127.0.0.1:$port")
      setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
        -m vllm.entrypoints.openai.api_server \
        --model "$EVAL_MODEL" --served-model-name "$EVAL_NAME" \
        --host 127.0.0.1 --port "$port" --dtype float16 \
        --max-model-len 32768 --max-num-seqs "$EVAL_MAX_NUM_SEQS" \
        --max-num-batched-tokens 32768 --gpu-memory-utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
        --enable-prefix-caching --enable-chunked-prefill \
        --no-enable-log-requests \
        > "$PHASE_ROOT/inference-gpu$gpu.log" 2>&1 &
      EVAL_PIDS+=("$!")
    done
    setsid uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
      -m vllm_router.launch_router --host 127.0.0.1 --port "$ROUTER_PORT" \
      --prometheus-port "$((19300 + phase_index))" \
      --worker-urls "${WORKER_URLS[@]}" \
      > "$PHASE_ROOT/router.log" 2>&1 &
    EVAL_PIDS+=("$!")
    phase_index=$((phase_index + 1))
  done

  uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
    - "${#EVAL_PHASES[@]}" "${EVAL_PIDS[@]}" <<'PY'
import time
import sys
from pathlib import Path
import httpx

pending = [19100 + gpu for gpu in range(8)] + [19200 + i for i in range(int(sys.argv[1]))]
deadline = time.monotonic() + 900
with httpx.Client(trust_env=False) as client:
    while pending:
        for pid in sys.argv[2:]:
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                raise RuntimeError(f"Inference process {pid} exited during startup; see inference/router logs")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Inference startup timed out: {pending}")
        for port in pending[:]:
            try:
                response = client.get(f"http://127.0.0.1:{port}/health", timeout=1)
            except httpx.RequestError:
                continue
            if response.status_code == 200:
                pending.remove(port)
        time.sleep(2)
PY

  PRODUCER_PIDS=()
  phase_index=0
  for phase in "${EVAL_PHASES[@]}"; do
    PHASE_ROOT="$EVAL_ROOT/$phase"
    EVAL_NAME="$(basename "$MODEL_DIR")"
    ROUTER_PORT=$((19200 + phase_index))
    if [[ "$phase" == sft ]]; then
      PHASE_ROOT="$EVAL_ROOT/global_step_$EVAL_STEP/sft"
      EVAL_NAME="$EVAL_NAME-nemotron-cot-sft-step$EVAL_STEP"
    fi
    (
      cd "$LIGHTEVAL_PROJECT"
      unset WANDB_RUN_ID WANDB_RESUME WANDB_SERVICE
      export PYTHONPATH="$SKYRL_PROJECT/.venv/rwkv-eval:$SKYRL_PROJECT"
      exec setsid uv run --isolated --no-config --no-project "$SKYRL_PROJECT/.venv/bin/python" \
        - "$phase" "$PHASE_ROOT" "$EVAL_NAME" "$ROUTER_PORT" \
        "$EVAL_MAX_NUM_SEQS" "$WANDB_RUN_PATH" "$SKYRL_PROJECT" \
        "$EVAL_STEP" "$GPUS_PER_PHASE" <<'PY'
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from argparse import Namespace

logging.basicConfig(level=logging.INFO)
phase, root, model_name, port, capacity, run_path, skyrl_project, step, gpu_count = sys.argv[1:]
sys.path.insert(0, skyrl_project)
from examples.train.rwkv.evaluate import evaluate

root = Path(root)
eval_metrics = {}
selectors = os.environ.get("EVAL_TASKS", "aime24 aime25 aimo_progress_prize_1 math_500").split()
for selector in selectors:
    avg_k = int(os.environ.get("EVAL_AVG_K", 4 if selector == "math_500" else 16))
    evaluate(Namespace(
        project=Path.cwd(), url=f"http://127.0.0.1:{port}", model=model_name,
        tokenizer=os.environ["MODEL_DIR"], output=root / selector, step=int(step),
        task=selector, num_samples=avg_k, context_length=25600,
        concurrency=int(gpu_count) * int(capacity) + 32, max_samples=None,
    ))
    rows = json.loads((root / selector / "scores.json").read_text())
    for row in rows.values():
        eval_metrics.update({f"{phase}/{selector}/{key}": value for key, value in row.items()})
    (root / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2) + "\n")
(root / "completed.json").write_text(json.dumps({"global_step": int(step), "model_name": model_name,
                                               "benchmarks": selectors}) + "\n")
(root / "upload_status.json").write_text(json.dumps({"state": "queued", "global_step": int(step)}) + "\n")
evaluation_root = root.parent.parent if phase == "sft" else root.parent
try:
    with (root / "upload.log").open("ab") as log:
        uploader = subprocess.Popen([
            "bash", str(Path(skyrl_project) / "examples/train/rwkv/run_rwkv_sft.sh"),
        ], cwd=skyrl_project, env={**os.environ, "UPLOAD_ONLY": "1", "UPLOAD_PHASE": phase,
                                  "EVAL_STEP": step, "OUTPUT_ROOT": str(evaluation_root.parent.parent),
                                  "EVAL_ROOT": str(evaluation_root),
                                  "WANDB_RUN_PATH": run_path, "CUDA_VISIBLE_DEVICES": ""},
           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
except OSError as error:
    (root / "upload_status.json").write_text(json.dumps({"state": "failed", "global_step": int(step),
                                                       "error": str(error)}) + "\n")
    logging.exception("Could not start W&B uploader; evaluation results remain saved")
else:
    (root / "upload.pid").write_text(str(uploader.pid) + "\n")
    logging.info("Evaluation saved; background W&B upload pid=%s, log=%s", uploader.pid, root / "upload.log")
PY
    ) > "$PHASE_ROOT/producer.log" 2>&1 &
    PRODUCER_PIDS+=("$!")
    EVAL_PIDS+=("$!")
    phase_index=$((phase_index + 1))
  done
  for pid in "${PRODUCER_PIDS[@]}"; do
    wait "$pid"
  done
  uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
    - "$EVAL_ROOT" "$EVAL_STEP" "${EVAL_PHASES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root, step, *phases = sys.argv[1:]
root = Path(root)
path = root / f"global_step_{step}/sft/eval_metrics.json"
if "sft" in phases:
    metrics = json.loads(path.read_text())
    if "base" in phases:
        metrics.update(json.loads((root / "base/eval_metrics.json").read_text()))
    path.write_text(json.dumps(metrics, indent=2) + "\n")
PY
fi
