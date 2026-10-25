"""Run the pinned ``lighteval-rwkv`` producer and publish step metrics.

The producer owns task loading, RWKV prompt templates, sampling, answer
extraction, and scoring. This file only supplies the eight endpoint replicas
and converts the producer's score records into the training run's W&B schema.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

BENCHMARKS = (
    ("maths", "gsm8k", 1),
    ("maths", "math_500", 8),
    ("maths", "aime24", 128),
    ("maths", "aime25", 128),
    ("maths", "aimo_progress_prize_1", 384),
)


def _write_manifests(root: Path, model: str, endpoints: list[str]) -> tuple[Path, Path]:
    """Write private model and benchmark manifests for the stable producer."""
    root.mkdir(parents=True, exist_ok=True)
    models = root / "models.toml"
    models.write_text(
        "\n".join(
            "[[model]]\n"
            f'url = "{url}"\napi_key = "EMPTY"\nmodel_name = "{model}"\n'
            "max_num_seqs = 1024\nctx_len = 32768\n"
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
    """Evaluate one checkpoint through the existing eight-worker producer."""
    project = args.lighteval_project.expanduser().resolve()
    sys.path[:0] = [str(project / "src"), str(project)]
    from src.rwkv_eval.configs import BenchmarkSpec, ModelEndpoint
    from src.rwkv_eval.main import RWKVRegistry
    from src.rwkv_eval.main import evaluate as producer_evaluate

    endpoints = [url.strip().rstrip("/") for url in args.base_url.split(",") if url.strip()]
    if len(endpoints) != 8:
        raise ValueError(f"expected eight inference endpoints, got {len(endpoints)}")
    model_name = Path(args.model).name
    models_path, benchmarks_path = _write_manifests(args.output / ".manifests", model_name, endpoints)
    models = tuple(ModelEndpoint(url, "EMPTY", model_name, args.max_num_seqs, args.context_length) for url in endpoints)
    benchmarks = tuple(BenchmarkSpec(field, name) for field, name, _ in BENCHMARKS)

    def run_producer(output_dir: Path):
        return asyncio.run(
            producer_evaluate(
                models,
                benchmarks,
                prompt_template="bot",
                max_samples=args.max_samples,
                test_mode=args.test_mode,
                seed=args.seed,
                no_upload=True,
                output_dir=str(output_dir),
            )
        )

    scores, failures = run_producer(args.output / "avg")
    if failures:
        raise RuntimeError(f"lighteval-rwkv returned {len(failures)} failed avg benchmark records")

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
    _log_wandb(args.step, rows)
    return output


def main() -> None:
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
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_mode", "--test-mode", action="store_true")
    args = parser.parse_args()
    if args.step < 0 or args.context_length < 1 or args.max_num_seqs < 1:
        parser.error("step must be non-negative and serving limits must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
