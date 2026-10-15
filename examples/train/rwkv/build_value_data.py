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
import tempfile
from collections import Counter
from pathlib import Path

import orjson

from skyrl.train.dataset.value_dataset import ValueDataset


def question_split(uuid: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{uuid}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def build_data(input_dir: Path, output_dir: Path, validation_fraction=0.05, seed=42, max_length=8192):
    """Stream complete scored shards into atomically published train/validation files."""
    config = orjson.loads((input_dir / "summary.json").read_bytes())["config"]
    if not config.get("save_rollouts"):
        raise ValueError("Collect token trajectories with rollout.py --save-rollouts first.")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    shards = sorted((input_dir / "rollouts").glob("*/*.jsonl"))
    if not shards:
        raise ValueError("No completed token trajectory shards found.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix=".value-data-", dir=output_dir.parent) as directory:
        staging = Path(directory)
        with (staging / "train.jsonl").open("wb") as train, (staging / "validation.jsonl").open("wb") as validation:
            files = {"train": train, "validation": validation}
            for shard in shards:
                kept_question = False
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
                        kept_question = True
                if kept_question:
                    counts[f"{split}_questions"] += 1
        if not counts["train_rollouts"] or not counts["validation_rollouts"]:
            raise ValueError(
                "Question split produced an empty train/validation set; collect more questions or change the split."
            )
        manifest = {
            "source": str(input_dir.resolve()),
            "source_config": config,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "max_length": max_length,
            "counts": counts,
        }
        (staging / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
        staging.replace(output_dir)
    print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="rollout.py output directory")
    parser.add_argument("--output", type=Path, required=True, help="New dataset output directory")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if not math.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1.")
    if args.max_length < 2:
        parser.error("--max-length must be >= 2.")
    build_data(args.input, args.output, args.validation_fraction, args.seed, args.max_length)


if __name__ == "__main__":
    main()
