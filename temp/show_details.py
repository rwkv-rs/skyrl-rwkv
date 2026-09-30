#!/usr/bin/env python3
"""Collect train or eval rollout details from a SkyRL experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_LIMIT = 20
DEFAULT_RUN_ROOT = "~/skyrl-rwkv-runs"
CATEGORIES = ("passed_details", "wrong_details", "failed_details")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def _reward_value(value: Any) -> float:
    if isinstance(value, list):
        return sum(float(item) for item in value)
    return float(value)


def _ground_truth(env_extras: Any) -> str:
    if not isinstance(env_extras, dict):
        return ""
    reward_spec = env_extras.get("reward_spec")
    if isinstance(reward_spec, dict) and "ground_truth" in reward_spec:
        return _as_text(reward_spec["ground_truth"])
    return _as_text(env_extras.get("ground_truth", ""))


def _resolve_run_dir(exp: str, run_root: Path) -> Path:
    supplied = Path(exp).expanduser()
    if supplied.is_absolute() or supplied.exists():
        return supplied
    configured_output = os.environ.get("OUTPUT_ROOT")
    if configured_output and Path(configured_output).expanduser().name == exp:
        return Path(configured_output).expanduser()
    return run_root.expanduser() / exp


def _detail_from_row(row: dict[str, Any], step: int) -> tuple[str, dict[str, Any]]:
    answer = _as_text(row.get("output_response"))
    failed = (
        row.get("stop_reason") == "length"
        or not answer.strip()
        or ("extracted_answer" in row and row["extracted_answer"] is None)
    )
    passed = not failed and _reward_value(row.get("score", 0.0)) > 0
    category = "passed_details" if passed else "failed_details" if failed else "wrong_details"
    prompt = row.get("input_prompt")
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": _as_text(prompt)}]
    return category, {
        "step": step,
        "messages": [*messages, {"role": "assistant", "content": answer}],
        "answer": answer,
        "ground_truth": _ground_truth(row.get("env_extras")),
        "is_passed": passed,
        "rendered_input_prompt": row.get("rendered_input_prompt"),
        "score": row.get("score"),
        "stop_reason": row.get("stop_reason"),
        "extracted_answer": row.get("extracted_answer"),
    }


def _collect_rows(
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, list[dict[str, Any]]]:
    """Collect at most 20 details per outcome across the requested steps."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    limit = min(limit, DEFAULT_LIMIT)
    details = {category: [] for category in CATEGORIES}
    counts = dict.fromkeys(CATEGORIES, 0)
    rng = random.Random(0)
    for step, row in rows:
        category, detail = _detail_from_row(row, step)
        counts[category] += 1
        if len(details[category]) < limit:
            details[category].append(detail)
        else:
            index = rng.randrange(counts[category])
            if index < limit:
                details[category][index] = detail
    for bucket in details.values():
        bucket.sort(key=lambda detail: detail["step"])
    return details


def collect_eval_details(
    run_dir: Path,
    steps: list[int],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for step in steps:
        step_dir = run_dir / "exports" / "dumped_evals" / f"global_step_{step}_evals"
        if not step_dir.is_dir():
            raise FileNotFoundError(f"evaluation step directory not found: {step_dir}")
        for file_path in sorted(step_dir.glob("*.jsonl")):
            if file_path.name == "aggregated_results.jsonl":
                continue
            with file_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"{file_path}:{line_number}: expected a JSON object")
                    rows.append((step, row))
    return _collect_rows(rows, limit=limit)


def collect_train_details(
    run_dir: Path,
    steps: list[int],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, list[dict[str, Any]]]:
    def rollout_rows() -> Iterable[tuple[int, dict[str, Any]]]:
        for start in sorted(set(steps)):
            available = [
                (step, run_dir / "exports" / "dumped_rollouts" / f"global_step_{step}.jsonl")
                for step in range(start, start + 25)
            ]
            available = [(step, path) for step, path in available if path.is_file()]
            if not available:
                raise FileNotFoundError(f"no training rollouts for steps [{start}, {start + 25}) in {run_dir}")
            for step, file_path in available:
                with file_path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise TypeError(f"{file_path}:{line_number}: expected a JSON object")
                        yield step, row

    return _collect_rows(rollout_rows(), limit=limit)


def _display_text(value: Any) -> str:
    """Render escaped control sequences as terminal-friendly text."""
    return _as_text(value).replace("\\r\\n", "\\n").replace("\\n", "\n").replace("\\t", "\t")


def _format_human(payload: dict[str, Any]) -> str:
    lines = [
        f"Experiment: {payload['experiment']}",
        f"Source: {payload['source']}",
        f"Steps: {', '.join(str(step) for step in payload['steps'])}",
        "Counts: "
        + ", ".join(f"{category.removesuffix('_details')}={count}" for category, count in payload["counts"].items()),
    ]
    for category in CATEGORIES:
        title = category.removesuffix("_details").upper()
        for index, detail in enumerate(payload[category], start=1):
            lines.extend(
                (
                    "",
                    f"{'=' * 20} {title} #{index} | step={detail['step']} {'=' * 20}",
                    f"Reward: {detail['score']} | stop={detail['stop_reason']} | extracted={detail['extracted_answer']}",
                    f"Ground truth: {_display_text(detail['ground_truth'])}",
                )
            )
            for message in detail["messages"]:
                role = _display_text(message.get("role", "user")).upper()
                lines.extend((f"[{role}]", _display_text(message.get("content", "")), ""))
            if detail["rendered_input_prompt"] is not None:
                lines.extend(("[RENDERED INPUT]", _display_text(detail["rendered_input_prompt"]), ""))
    return "\n".join(lines).rstrip() + "\n"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect train or eval rollout details from a SkyRL run.")
    parser.add_argument("--exp", required=True, help="Experiment name or output directory")
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        action="extend",
        required=True,
        help="Train window start(s): [step, step+25); eval uses exact steps",
    )
    parser.add_argument("--source", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ.get("SKYRL_RUN_ROOT", DEFAULT_RUN_ROOT)),
        help="Root containing experiment output directories",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum details per outcome (up to 20)")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format; text renders escaped newlines for terminal viewing",
    )
    parser.add_argument("--output", type=Path, help="Write formatted output to this file instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if any(step < 0 for step in args.step):
        raise SystemExit("--step must be non-negative")

    run_dir = _resolve_run_dir(args.exp, args.run_root)
    if args.source == "train":
        details = collect_train_details(run_dir, args.step, limit=args.limit)
    else:
        details = collect_eval_details(run_dir, args.step, limit=args.limit)
    payload = {
        "experiment": args.exp,
        "source": args.source,
        "steps": args.step,
        "counts": {category: len(details[category]) for category in CATEGORIES},
        **details,
    }
    output = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n" if args.format == "json" else _format_human(payload)
    )
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
