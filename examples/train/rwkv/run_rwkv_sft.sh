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
export RAY_ADDRESS=local
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export SKYRL_PYTHONPATH_EXPORT=1
export XDG_CACHE_HOME="$HOME/.cache/skyrl-rwkv"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/train.log") 2>&1

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
    logger="$LOGGER" project_name=rwkv_direct_sft run_name="$RUN_NAME" \
    "${EVAL_OVERRIDES[@]}" \
    "$@"
fi

# Invoke the existing LightEval producer without changing its checkout.
if [[ -n "${LIGHTEVAL_PROJECT:-}" && "${EVAL_ONLY:-0}" == 1 ]]; then
  : "${WANDB_RUN_PATH:?Set the training run path: entity/project/run_id}"
  : "${EVAL_MAX_NUM_SEQS:=1024}"
  : "${EVAL_GPU_MEMORY_UTILIZATION:=0.65}"
  : "${EVAL_STEP:=$TOTAL_STEPS}"
  SKYRL_PROJECT="$PWD"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
  EVAL_ROOT="$OUTPUT_ROOT/evaluation/lighteval_avg16"
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
    wait || true
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
      unset PYTHONPATH WANDB_RUN_ID WANDB_RESUME
      exec setsid uv run --isolated --no-config --no-project "$PWD/.venv/bin/python" \
        - "$phase" "$PHASE_ROOT" "$EVAL_NAME" "$ROUTER_PORT" \
        "$EVAL_MAX_NUM_SEQS" "$WANDB_RUN_PATH" "$SKYRL_PROJECT" \
        "$EVAL_STEP" "$GPUS_PER_PHASE" <<'PY'
import json
import logging
import subprocess
import sys
from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.pipeline import ParallelismManager, PipelineParameters
from lighteval.tasks.requests import SamplingMethod
from src.rwkv_eval.configs import ModelEndpoint
from src.rwkv_eval.main import RWKVExactMatches, RWKVPipeline, _litelm_model

logging.basicConfig(level=logging.INFO)
phase, root, model_name, port, capacity, run_path, skyrl_project, step, gpu_count = sys.argv[1:]
root = Path(root)
eval_metrics = {}
endpoint = ModelEndpoint(f"http://127.0.0.1:{port}", "unused", model_name,
                         int(gpu_count) * int(capacity) + 32, 25600)
for selector in ["aime24", "aime25", "aimo_progress_prize_1", "math_500"]:
    config = _litelm_model(endpoint, [endpoint], "CoT", "bot", 24576, 42,
                          cache_dir=str(root / ".lighteval_cache"))
    tracker = EvaluationTracker(output_dir=str(root / selector),
                                save_details=False, save_streaming_completions=True)
    pipeline = RWKVPipeline(selector, PipelineParameters(
        launcher_type=ParallelismManager.NONE, streaming_evaluation=True),
        tracker, model_config=config)
    for task in pipeline.tasks_dict.values():
        metric = Metrics.avg_at_n_math(sample_params={"n": 16})
        metric.sample_level_fn.compute_score = RWKVExactMatches().compute
        task.metrics = task.config.metrics = [metric]
        task.num_samples = [16]
        task.sampling_methods = [SamplingMethod.GENERATIVE]
    tracker.task_config_logger.log(pipeline.tasks_dict)
    pipeline.evaluate()
    pipeline.save_and_push_results()
    results = pipeline.get_results()
    rows = {}
    for task_name in pipeline.tasks_dict:
        metrics = results["results"][task_name.replace("|", ":")]
        score = next(value for name, value in metrics.items() if not name.endswith("_stderr"))
        count = pipeline.task_sample_counts[task_name]
        completions = pipeline.task_completion_counts[task_name]
        if completions != 16 * count or pipeline.task_avg_k[task_name] != 16:
            raise ValueError(f"Incomplete avg@16 evaluation: {task_name}, {count=}, {completions=}")
        row = {"avg@16": float(score), "num_questions": count,
               "num_completions": completions,
               "truncation_rate": pipeline.task_truncation_counts[task_name] / completions}
        rows[task_name] = row
        eval_metrics.update({f"{phase}/{selector}/{key}": value for key, value in row.items()})
    (root / selector / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2) + "\n")
    subprocess.run([
        "uv", "run", "--isolated", "--no-config", "--no-project",
        str(Path(skyrl_project) / ".venv/bin/python"), "-", run_path, phase,
        selector, str(root / selector), model_name, step,
    ], check=True, text=True, cwd=skyrl_project, input='''
import json, sys
from pathlib import Path
import wandb
run_path, phase, selector, directory, model_name, step = sys.argv[1:]
entity, project, run_id = run_path.split("/")
run = wandb.init(entity=entity, project=project, id=f"{run_id}-eval-{phase}",
                 resume="allow", name=f"{run_id}-{phase}-avg16", group=run_id,
                 job_type="evaluation", allow_val_change=True, config={"model_name": model_name,
                     "phase": phase, "avg_k": 16, "max_generated_tokens": 24576,
                     "wkv_mode": "fp32io16"})
rows = json.loads((Path(directory) / "scores.json").read_text())
values = {f"eval/{phase}/{selector}/{key}": value
          for row in rows.values() for key, value in row.items()}
run.define_metric("eval/global_step")
run.define_metric(f"eval/{phase}/*", step_metric="eval/global_step")
run.log({**values, "eval/global_step": int(step)})
training_run = wandb.Api().run(run_path)
training_run.summary.update(values)
artifact = wandb.Artifact(f"{run_id}-{phase}-{selector}-avg16-step{step}", type="evaluation")
artifact.add_dir(directory)
run.log_artifact(artifact).wait()
run.finish()
''')
(root / "completed.json").write_text(json.dumps({"global_step": int(step), "model_name": model_name}) + "\n")
PY
    ) > "$PHASE_ROOT/producer.log" 2>&1 &
    PRODUCER_PIDS+=("$!")
    EVAL_PIDS+=("$!")
    phase_index=$((phase_index + 1))
  done
  for pid in "${PRODUCER_PIDS[@]}"; do
    wait "$pid"
  done
fi
