"""Run the pinned ``lighteval-rwkv`` producer and publish step metrics.

The producer owns task loading, RWKV prompt templates, sampling, answer
extraction, and scoring. This file only supplies the eight endpoint replicas
and converts the producer's score records into the training run's W&B schema.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

BENCHMARKS = (
    ("maths", "gsm8k", 1),
    ("maths", "math_500", 8),
    ("maths", "aime24", 128),
    ("maths", "aime25", 128),
    ("maths", "aimo_progress_prize_1", 384),
)


def _write_manifests(
    root: Path, model: str, endpoints: list[str], *, max_num_seqs: int = 128, context_length: int = 32768
) -> tuple[Path, Path]:
    """Write private model and benchmark manifests for the stable producer."""
    root.mkdir(parents=True, exist_ok=True)
    models = root / "models.toml"
    models.write_text(
        "\n".join(
            "[[model]]\n"
            f'url = "{url}"\napi_key = "EMPTY"\nmodel_name = "{model}"\n'
            f"max_num_seqs = {max_num_seqs}\nctx_len = {context_length}\n"
            for url in endpoints
        )
    )
    benchmarks = root / "benchmarks.toml"
    benchmarks.write_text("maths = [\n" + "\n".join(f'  "{name}",' for _, name, _ in BENCHMARKS) + "\n]\n")
    return models, benchmarks


def _log_wandb(step: int, rows: list[dict[str, object]]) -> None:
    """Log scalar curves and a reusable avg/pass table when W&B is enabled."""
    # The SFT eval hook reads eval_metrics.json and logs it on the active
    # training run. Never reopen that run here: wandb.finish() would mark an
    # in-progress training run as finished. Standalone callers can opt into a
    # separate evaluation run explicitly.
    if os.environ.get("WANDB_EVAL_LOG_NEW_RUN") != "1":
        return
    import wandb

    kwargs = {
        "project": os.environ.get("WANDB_PROJECT", "skyrl-rwkv"),
        "job_type": "evaluation",
    }
    run = wandb.init(**kwargs)
    try:
        table = wandb.Table(columns=["step", "benchmark", "k", "avg", "pass"])
        payload: dict[str, float | int | object] = {"eval/checkpoint_step": step}
        for row in rows:
            name, k = str(row["benchmark"]), int(row["k"])
            avg, passed = float(row["avg"]), float(row["pass"])
            payload[f"eval/{name}/avg@{k}"] = avg
            payload[f"eval/{name}/pass@{k}"] = passed
            table.add_data(step, name, k, avg, passed)
        payload["eval/avg_pass_curve"] = table
        run.log(payload, step=step)
    finally:
        run.finish()


def evaluate(args: argparse.Namespace) -> Path:
    """Run the producer in its own environment and publish checkpoint metrics."""
    args.output = args.output.expanduser().resolve()
    args.lighteval_project = args.lighteval_project.expanduser().resolve()
    project = args.lighteval_project
    python = project / ".venv/bin/python"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(project / "src"), str(project)))}
    timeout = args.producer_timeout
    if not args.benchmarks:
        # The Math profile uses the producer's existing avg/pass configuration in an isolated worker.
        subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--math-worker", json.dumps(vars(args), default=str)],
            cwd=project,
            env=env,
            check=True,
            timeout=timeout,
        )
        result = json.loads((args.output / "result.json").read_text())
        _log_wandb(args.step, result["rows"])
        return args.output / "eval_metrics.json"

    endpoints = [url.strip().rstrip("/") for url in args.base_url.split(",") if url.strip()]
    if len(endpoints) != 8:
        raise ValueError(f"expected eight inference endpoints, got {len(endpoints)}")
    models_path, benchmarks_path = _write_manifests(
        args.output / ".manifests",
        args.served_model_name or Path(args.model).name,
        endpoints,
        max_num_seqs=args.max_num_seqs,
        context_length=args.context_length,
    )
    shutil.copyfile(args.benchmarks, benchmarks_path)
    command = [
        str(python),
        "-m",
        "src.rwkv_eval.main",
        "--models",
        str(models_path),
        "--benchmarks",
        str(benchmarks_path),
        "--prompt-template",
        args.prompt_template,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output),
        "--no-upload",
    ]
    if args.max_samples is not None:
        command.extend(("--max-samples", str(args.max_samples)))
    if args.test_mode:
        command.append("--test-mode")
    subprocess.run(command, cwd=project, env=env, check=True, timeout=timeout)
    scores = [json.loads(path.read_text()) for path in sorted((args.output / "scores").glob("*/*.json"))]
    expected = {name for names in tomllib.loads(benchmarks_path.read_text()).values() for name in names}
    if not scores or {score["benchmark_name"] for score in scores} != expected or len(scores) != len(expected):
        raise RuntimeError("lighteval-rwkv did not publish exactly the requested benchmark scores")
    if any(score["upload_status"] != "local_only" for score in scores):
        raise RuntimeError("lighteval-rwkv returned failed benchmark records")
    output = args.output / "eval_metrics.json"
    metrics = {f"eval/{score['benchmark_name']}": float(score["score"]) for score in scores}
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    if getattr(args, "probe_max_samples", None) is not None:
        if args.max_samples is not None:
            raise ValueError("a cached probe requires a full evaluation")
        probe = args.output / "probe"
        with tempfile.TemporaryDirectory(dir=args.output) as directory:
            backup = Path(directory)
            shutil.copytree(args.output / "scores", backup / "scores")
            shutil.copyfile(output, backup / output.name)
            probe_args = copy.copy(args)
            probe_args.max_samples = args.probe_max_samples
            probe_args.probe_max_samples = None
            try:
                # Keep the producer output path: it is part of its cache key.
                # The fixed subset is scored from this full run's cached responses.
                evaluate(probe_args)
                shutil.copytree(args.output / "scores", probe / "scores", dirs_exist_ok=True)
                shutil.copyfile(output, probe / output.name)
            finally:
                shutil.copytree(backup / "scores", args.output / "scores", dirs_exist_ok=True)
                shutil.copyfile(backup / output.name, output)
    return output


def _evaluate_math(args: argparse.Namespace) -> Path:
    """Run the default Math avg/pass profile inside the producer environment."""
    project = args.lighteval_project.expanduser().resolve()
    sys.path[:0] = [str(project / "src"), str(project)]
    from src.rwkv_eval.configs import BenchmarkSpec, ModelEndpoint
    from src.rwkv_eval.main import RWKVRegistry
    from src.rwkv_eval.main import evaluate as producer_evaluate

    endpoints = [url.strip().rstrip("/") for url in args.base_url.split(",") if url.strip()]
    if len(endpoints) != 8:
        raise ValueError(f"expected eight inference endpoints, got {len(endpoints)}")
    model_name = args.served_model_name or Path(args.model).name
    models_path, benchmarks_path = _write_manifests(
        args.output / ".manifests",
        model_name,
        endpoints,
        max_num_seqs=args.max_num_seqs,
        context_length=args.context_length,
    )
    models = tuple(ModelEndpoint(url, "EMPTY", model_name, args.max_num_seqs, args.context_length) for url in endpoints)
    benchmarks = tuple(BenchmarkSpec(field, name) for field, name, _ in BENCHMARKS)

    def run_producer(output_dir: Path):
        return asyncio.run(
            producer_evaluate(
                models,
                benchmarks,
                prompt_template=args.prompt_template,
                max_samples=args.max_samples,
                test_mode=args.test_mode,
                seed=args.seed,
                no_upload=True,
                output_dir=str(output_dir),
            )
        )

    scores, failures = run_producer(args.output / "avg")
    if failures:
        raise RuntimeError(f"lighteval-rwkv returned {len(failures)} failed benchmark records")

    original_update = RWKVRegistry._update_task_configs

    def pass_metrics(registry):
        configs = original_update(registry)
        from lighteval.metrics.metrics import Metrics

        expected = {name: k for _, name, k in BENCHMARKS}
        for task_configs in configs.values():
            for config in task_configs:
                if config.name not in expected:
                    continue
                metric = copy.deepcopy(Metrics.pass_at_k_math.value)
                metric.sample_level_fn.k = expected[config.name]
                metric.sample_level_fn.n = expected[config.name]
                config.metrics = [metric]
        return configs

    RWKVRegistry._update_task_configs = pass_metrics
    try:
        pass_scores, pass_failures = run_producer(args.output / "pass")
    finally:
        RWKVRegistry._update_task_configs = original_update
    if pass_failures:
        raise RuntimeError(f"lighteval-rwkv returned {len(pass_failures)} failed pass benchmark records")
    pass_by_name = {score.benchmark_name: float(score.score) for score in pass_scores}

    rows = []
    for score in scores:
        k = next(expected for _, name, expected in BENCHMARKS if name == score.benchmark_name)
        value = float(score.score)
        rows.append(
            {
                "benchmark": score.benchmark_name,
                "k": k,
                "avg": value,
                "pass": pass_by_name[score.benchmark_name],
                "avg_k": score.avg_k,
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {f"eval/{row['benchmark']}/avg@{row['k']}": row["avg"] for row in rows}
    metrics.update({f"eval/{row['benchmark']}/pass@{row['k']}": row["pass"] for row in rows})
    result = {
        "global_step": args.step,
        "rows": rows,
        "metrics": metrics,
        "models": str(models_path),
        "benchmarks": str(benchmarks_path),
    }
    output = args.output / "eval_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
    return output


def _save_progress(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(state, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _progress(path: Path, spec: dict) -> tuple[Path, dict]:
    saved = path.with_suffix(".progress.json")
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    state = json.loads(saved.read_text()) if saved.exists() else {"spec_sha256": digest}
    if state["spec_sha256"] != digest:
        raise ValueError("training/evaluation configuration changed; use a new lifecycle")
    return saved, state


def _budgeted(command: list[str], env: dict, seconds: float, path: Path, state: dict, phase: str) -> None:
    receipt = state.setdefault(phase, {"seconds": 0})
    remaining = seconds - receipt["seconds"]
    if remaining <= 0:
        raise TimeoutError(f"{phase} exhausted its budget")
    began = time.monotonic()
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while True:
            elapsed = time.monotonic() - began
            receipt["seconds"] = seconds - remaining + elapsed
            _save_progress(path, state)
            if elapsed >= remaining:
                raise subprocess.TimeoutExpired(command, remaining)
            try:
                code = process.wait(timeout=min(10, remaining - elapsed))
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        receipt["seconds"] = seconds - remaining + time.monotonic() - began
        _save_progress(path, state)


def train_and_evaluate(path: Path, command: list[str]) -> None:
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads(path.read_text())
        saved, state = _progress(path, spec)
        command_hash = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        if state.get("command_sha256", command_hash) != command_hash:
            raise ValueError("training command changed; use a new lifecycle")
        state["command_sha256"] = command_hash
        if not state.get("training_done"):
            _budgeted(command, dict(os.environ), int(os.environ["TRAIN_MAX_SECONDS"]), saved, state, "training")
            state["training_done"] = True
            _save_progress(saved, state)
        after_train(path)


def after_train(path: Path) -> None:
    """Publish an exact export, evaluate it, and confirm a passing probe at full size.

    The caller supplies a JSON argv for its acceptance callback. Exit 0 requests
    full confirmation, exit 1 rejects the probe; other failures stop the lifecycle.
    Evaluation has its own wall budget after the training process exits.
    """
    spec = json.loads(path.read_text())
    saved, state = _progress(path, spec)
    source, model = Path(spec["export"]).expanduser(), Path(spec["model"]).expanduser()
    marker = ".rwkv-publication.json"
    if model.exists():
        receipt = model / marker
        if not receipt.is_file() or json.loads(receipt.read_text()) != {"spec_sha256": state["spec_sha256"]}:
            raise FileExistsError(f"candidate already exists: {model}")
    else:
        if not (source / "config.json").is_file():
            raise ValueError(f"missing HF export: {source}")
        model.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{model.name}-", dir=model.parent) as directory:
            staging = Path(directory) / "model"
            shutil.copytree(source, staging)
            _save_progress(staging / marker, {"spec_sha256": state["spec_sha256"]})
            staging.rename(model)
    for size in ("probe", "full"):
        settings = spec[size]
        env = {
            **os.environ,
            **spec["env"],
            "MODEL": str(model),
            "OUTPUT": settings["output"],
            "MAX_SAMPLES": str(settings.get("max_samples") or ""),
        }
        receipt = state.setdefault(size, {"seconds": 0})
        if "passed" in receipt:
            if not receipt["passed"]:
                return
            continue
        if not receipt.get("evaluated"):
            _budgeted(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=120s",
                    f"{settings['seconds']}s",
                    "bash",
                    str(Path(__file__).with_name("run.sh")),
                ],
                env,
                settings["seconds"],
                saved,
                state,
                size,
            )
            receipt["evaluated"] = True
            _save_progress(saved, state)
        # The orchestrator records provenance and the gate result while GPUs are still handed over.
        result = subprocess.run([*spec["callback"], size], check=False, timeout=60)
        if result.returncode in (0, 1):
            receipt["passed"] = result.returncode == 0
            _save_progress(saved, state)
        if result.returncode == 1:
            return
        result.check_returncode()


def main() -> None:
    if len(sys.argv) > 3 and sys.argv[1] == "--train-and-evaluate":

        def interrupted(*_):
            raise InterruptedError("training lifecycle stopped")

        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            train_and_evaluate(Path(sys.argv[2]), sys.argv[3:])
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--after-train":
        after_train(Path(sys.argv[2]))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--math-worker":
        args = argparse.Namespace(**json.loads(sys.argv[2]))
        for name in ("model", "output", "lighteval_project"):
            setattr(args, name, Path(getattr(args, name)))
        _evaluate_math(args)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Eight comma-separated OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--lighteval-project",
        type=Path,
        default=Path.home() / "Projects/MachineLearning/lighteval-rwkv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--probe-max-samples", type=int, help="Score a subset from the completed full-run cache")
    parser.add_argument("--benchmarks", type=Path)
    parser.add_argument("--served-model-name")
    parser.add_argument("--prompt-template", default="bot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--producer-timeout", type=int, default=21600, help="Producer wall-time limit in seconds")
    parser.add_argument("--test_mode", "--test-mode", action="store_true")
    args = parser.parse_args()
    if args.step < 0 or args.context_length < 1 or args.max_num_seqs < 1 or args.producer_timeout < 1:
        parser.error("step must be non-negative and serving limits and producer timeout must be positive")
    if args.probe_max_samples is not None and (args.probe_max_samples < 1 or args.max_samples is not None):
        parser.error("--probe-max-samples requires a full evaluation and a positive subset size")
    evaluate(args)


if __name__ == "__main__":
    main()
