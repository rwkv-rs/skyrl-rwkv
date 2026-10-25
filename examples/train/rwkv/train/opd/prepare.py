"""Stream local UltraData questions into native single-turn OPD or RL datasets."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from examples.train.rwkv.train.sao.build_value_data import question_split
from examples.train.rwkv.train.common.rollout import DOMAINS


def prepare(input_dir: Path, output: Path, task_rewards=False, validation_fraction=0.0, seed=42):
    fields = [
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("env_class", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
        ("uuid", pa.string()),
    ]
    if task_rewards:
        fields.append(("reward_spec", pa.struct([("source_path", pa.string()), ("source_offset", pa.int64())])))
    schema = pa.schema(fields)
    counts = Counter()
    batches = {"train": [], "validation": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writers = {"train": stack.enter_context(pq.ParquetWriter(output, schema, compression="zstd"))}
        if validation_fraction:
            writers["validation"] = stack.enter_context(
                pq.ParquetWriter(output.with_name("validation.parquet"), schema, compression="zstd")
            )
        for domain in DOMAINS:
            paths = sorted((input_dir / domain).glob("*.jsonl"))
            if not paths:
                raise FileNotFoundError(f"No UltraData shards found in {input_dir / domain}")
            for path in paths:
                with path.open("rb") as stream:
                    offset = 0
                    for line in stream:
                        row = orjson.loads(line)
                        record = {
                            "prompt": [{"role": "user", "content": row["query"]}],
                            "env_class": "ultradata" if task_rewards else "aime",
                            # OPD replaces task rewards. Only Math has an AIME-compatible answer.
                            "reward_model": {"ground_truth": row["ground_truth"] if domain == "Math" else ""},
                            "data_source": domain,
                            "uuid": row["uuid"],
                        }
                        if task_rewards:
                            record["reward_spec"] = {"source_path": str(path.resolve()), "source_offset": offset}
                        offset += len(line)
                        split = (
                            question_split(row["uuid"], seed, validation_fraction) if validation_fraction else "train"
                        )
                        batches[split].append(record)
                        counts[domain] += 1
                        if len(batches[split]) == 4096:
                            writers[split].write_table(pa.Table.from_pylist(batches[split], schema=schema))
                            batches[split].clear()
        for split, batch in batches.items():
            if batch:
                writers[split].write_table(pa.Table.from_pylist(batch, schema=schema))
    manifest = {"input": str(input_dir), "output": str(output), "counts": dict(counts)}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "data/UltraData-RL-2609/files")
    parser.add_argument("--output", type=Path, default=Path.home() / "data/ultradata-opd/train.parquet")
    parser.add_argument("--task-rewards", action="store_true", help="Use native UltraData rewards for online RL")
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1).")
    prepare(args.input, args.output, args.task_rewards, args.validation_fraction, args.seed)


if __name__ == "__main__":
    main()
