#!/usr/bin/env python3
"""Build exact-token value datasets from rollout.py --save-rollouts output.

Follows ReDiPPO's build_standard_critic_data.py outcome-to-value-target pipeline
(Apache-2.0), using stored tokens and explicit action masks instead of re-tokenizing
text. All samples of a question UUID are assigned to the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import orjson

from skyrl.train.dataset.value_dataset import ValueDataset


def question_split(uuid: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{uuid}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def score_benchmarks(input_dir: Path, prompts_path: Path, output_dir: Path, lighteval_project: Path):
    """Score fresh Antidoom collector tokens with the scoreboard's native math verifier."""
    sys.path.insert(0, str(lighteval_project / "src"))
    sys.path.insert(0, str(lighteval_project))
    from src.rwkv_eval.answer_extract.free_response import (
        answer_extractor_for_benchmark,
        extract_answer,
        verify_math_answer,
    )
    from skyrl.utils.tok import get_tokenizer

    config = orjson.loads((input_dir / "generation.json").read_bytes())
    if hashlib.sha256(prompts_path.read_bytes()).hexdigest() != config["input_sha256"]:
        raise ValueError("Scoring prompts must match the input used to generate these rollouts.")
    prompts = [orjson.loads(line) for line in prompts_path.read_bytes().splitlines()]
    tokenizer = get_tokenizer(config["model"])
    output_dir.mkdir(parents=True, exist_ok=False)
    counts, domains, seen = Counter(), defaultdict(Counter), set()
    for path in sorted(input_dir.glob("rollouts*.jsonl")):
        with path.open("rb") as handle:
            for line in handle:
                row = orjson.loads(line)
                if row["model"] != config["model"]:
                    raise ValueError("Scoring requires the checkpoint that generated the tokens.")
                sample = prompts[row["prompt_index"]]
                source = sample["source"]
                name = source["benchmark"]
                extractor = answer_extractor_for_benchmark(name)
                if extractor != "math":
                    raise ValueError("Benchmark value scoring currently supports native math tasks.")
                identity = (row["prompt_index"], row["sample_index"])
                if identity in seen:
                    raise ValueError("Duplicate benchmark trajectory.")
                seen.add(identity)
                rendered = tokenizer.decode(row["prompt_ids"], skip_special_tokens=False)
                answer = extract_answer(row["text"], extractor, rendered_prompt=rendered)
                reward = float(any(verify_math_answer(gold, answer or "") for gold in sample["gold_answers"]))
                truncated = row["finish_reason"] in {"length", "max_tokens"}
                record = {
                    "uuid": source["problem_key"],
                    "domain": name,
                    "sample_index": row["sample_index"],
                    "prompt_ids": row["prompt_ids"],
                    "response_ids": row["response_ids"],
                    "reward": reward,
                    "truncated": truncated,
                    "parsed_answer": answer,
                    "gold_answers": sample["gold_answers"],
                    "source": source,
                }
                shard = output_dir / "rollouts" / name / f'{row["prompt_index"]}.jsonl'
                shard.parent.mkdir(parents=True, exist_ok=True)
                with shard.open("ab") as target:
                    target.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                counts["rollouts"] += 1
                counts["positive_rewards"] += reward > 0
                counts["truncated"] += truncated
                domains[name]["rollouts"] += 1
                domains[name]["positive_rewards"] += reward > 0
                domains[name]["truncated"] += truncated
                if counts["rollouts"] % 128 == 0:
                    print(orjson.dumps({"counts": counts, "domains": domains}).decode(), flush=True)
    expected = len(prompts) * config["samples_per_prompt"]
    if len(seen) != expected:
        raise ValueError(f"Expected {expected} benchmark trajectories, found {len(seen)}.")
    summary = {
        "config": {**config, "save_rollouts": True, "scoring": "native LightEval RWKV math verifier"},
        "counts": counts,
        "domains": domains,
    }
    (output_dir / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())
    return summary


def balance_data(path: Path, seed: int):
    """Downsample each domain's majority outcome without duplicating trajectories."""
    offsets = defaultdict(lambda: {False: [], True: []})
    with path.open("rb") as source:
        while line := source.readline():
            row = orjson.loads(line)
            if row["reward"] not in (0, 1):
                raise ValueError("Reward balancing requires binary 0/1 outcomes.")
            offsets[row["domain"]][bool(row["reward"])].append(source.tell() - len(line))
    rng = random.Random(seed)
    selected = []
    for domain in sorted(offsets):
        classes = offsets[domain]
        count = min(len(classes[False]), len(classes[True]))
        for positive in (False, True):
            selected.extend(rng.sample(classes[positive], count))
    rng.shuffle(selected)
    counts = Counter(rollouts=0, positive_rewards=0, action_tokens=0, questions=0)
    questions, domains = set(), defaultdict(Counter)
    balanced = path.with_suffix(".balanced")
    with path.open("rb") as source, balanced.open("wb") as target:
        for offset in selected:
            source.seek(offset)
            line = source.readline()
            row = orjson.loads(line)
            target.write(line)
            counts["rollouts"] += 1
            counts["positive_rewards"] += row["reward"] > 0
            counts["action_tokens"] += sum(row["loss_mask"])
            questions.add(row["uuid"])
            domains[row["domain"]]["positive" if row["reward"] else "negative"] += 1
    counts["questions"] = len(questions)
    balanced.replace(path)
    return counts, dict(domains)


def build_data(
    input_dir: Path,
    output_dir: Path,
    validation_fraction=0.05,
    seed=42,
    max_length=8192,
    *,
    additional_inputs=(),
    balance_rewards=False,
):
    """Stream complete scored shards into atomically published train/validation files."""
    inputs = [input_dir, *additional_inputs]
    configs = [orjson.loads((path / "summary.json").read_bytes())["config"] for path in inputs]
    for config in configs:
        if not config.get("save_rollouts"):
            raise ValueError("Collect token trajectories with rollout.py --save-rollouts first.")
        if config["model"] != configs[0]["model"]:
            raise ValueError("All rollout inputs must use the same checkpoint.")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    shards = [shard for path in inputs for shard in sorted((path / "rollouts").glob("*/*.jsonl"))]
    if not shards:
        raise ValueError("No completed token trajectory shards found.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    questions = {split: set() for split in ("train", "validation")}
    with tempfile.TemporaryDirectory(prefix=".value-data-", dir=output_dir.parent) as directory:
        staging = Path(directory)
        with (staging / "train.jsonl").open("wb") as train, (staging / "validation.jsonl").open("wb") as validation:
            files = {"train": train, "validation": validation}
            for shard in shards:
                with shard.open("rb") as source:
                    for line in source:
                        row = orjson.loads(line)
                        counts["read_rollouts"] += 1
                        # A time-limit truncation is not a terminal MC target.
                        # Keep it only when explicit bootstrapped returns exist.
                        if row["truncated"] and row.get("returns") is None:
                            counts["skipped_truncated"] += 1
                            continue
                        prompt, response = row["prompt_ids"], row["response_ids"]
                        if not response:
                            counts["skipped_empty"] += 1
                            continue
                        if len(prompt) + len(response) > max_length:
                            counts["skipped_overlength"] += 1
                            continue
                        response_mask = row.get("response_mask", [1] * len(response))
                        if len(response_mask) != len(response):
                            raise ValueError("response_mask must have one entry per response token.")
                        record = {
                            "uuid": row["uuid"],
                            "domain": row["domain"],
                            "sample_index": row["sample_index"],
                            "input_ids": prompt + response,
                            "loss_mask": [0] * len(prompt) + response_mask,
                            "reward": row["reward"],
                            "truncated": row["truncated"],
                        }
                        if row.get("returns") is not None:
                            record["returns"] = row["returns"]
                        ValueDataset._normalize_row(record)
                        split = question_split(row["uuid"], seed, validation_fraction)
                        files[split].write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                        counts[f"{split}_rollouts"] += 1
                        counts[f"{split}_positive_rewards"] += row["reward"] > 0
                        counts[f"{split}_action_tokens"] += sum(response_mask)
                        questions[split].add(row["uuid"])
        for split in questions:
            counts[f"{split}_questions"] = len(questions[split])
        domains = {}
        if balance_rewards:
            for split in ("train", "validation"):
                balanced_counts, domains[split] = balance_data(staging / f"{split}.jsonl", seed)
                for key, value in balanced_counts.items():
                    counts[f"{split}_before_balance_{key}"] = counts[f"{split}_{key}"]
                    counts[f"{split}_{key}"] = value
        if not counts["train_rollouts"] or not counts["validation_rollouts"]:
            raise ValueError(
                "Question split produced an empty train/validation set; collect more questions or change the split."
            )
        manifest = {
            "source": str(input_dir.resolve()),
            "source_config": configs[0],
            "seed": seed,
            "validation_fraction": validation_fraction,
            "max_length": max_length,
            "counts": counts,
        }
        if additional_inputs:
            manifest["additional_sources"] = [
                {"source": str(path.resolve()), "source_config": config}
                for path, config in zip(additional_inputs, configs[1:], strict=True)
            ]
        if balance_rewards:
            manifest["balance_rewards"] = "1:1 within each split and domain, downsampling without replacement"
            manifest["domains"] = domains
        (staging / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
        staging.replace(output_dir)
    print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="rollout.py output directory")
    parser.add_argument(
        "--additional-input", type=Path, action="append", default=[], help="Additional scored rollout directory"
    )
    parser.add_argument(
        "--balance-rewards", action="store_true", help="Keep equal 0/1 outcomes within each split and domain"
    )
    parser.add_argument(
        "--benchmark-prompts", type=Path, help="Score Antidoom collector output against exported native gold answers"
    )
    parser.add_argument(
        "--lighteval-project", type=Path, help="Source checkout of the scoreboard's native LightEval implementation"
    )
    parser.add_argument("--output", type=Path, required=True, help="New dataset output directory")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.benchmark_prompts:
        if not args.lighteval_project:
            parser.error("--benchmark-prompts requires --lighteval-project.")
        score_benchmarks(args.input, args.benchmark_prompts, args.output, args.lighteval_project)
        return
    if not math.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1.")
    if args.max_length < 2:
        parser.error("--max-length must be >= 2.")
    build_data(
        args.input,
        args.output,
        args.validation_fraction,
        args.seed,
        args.max_length,
        additional_inputs=args.additional_input,
        balance_rewards=args.balance_rewards,
    )


if __name__ == "__main__":
    main()
