"""Stream local UltraData questions into the native single-turn OPD dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from examples.train.rwkv.rollout import DOMAINS


def prepare(input_dir: Path, output: Path):
    schema = pa.schema(
        [
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("env_class", pa.string()),
            ("reward_model", pa.struct([("ground_truth", pa.string())])),
            ("data_source", pa.string()),
            ("uuid", pa.string()),
        ]
    )
    counts = Counter()
    batch = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for domain in DOMAINS:
            paths = sorted((input_dir / domain).glob("*.jsonl"))
            if not paths:
                raise FileNotFoundError(f"No UltraData shards found in {input_dir / domain}")
            for path in paths:
                with path.open("rb") as stream:
                    for line in stream:
                        row = orjson.loads(line)
                        batch.append(
                            {
                                "prompt": [{"role": "user", "content": row["query"]}],
                                "env_class": "aime",
                                # OPD replaces task rewards. Only Math has an AIME-compatible answer.
                                "reward_model": {"ground_truth": row["ground_truth"] if domain == "Math" else ""},
                                "data_source": domain,
                                "uuid": row["uuid"],
                            }
                        )
                        counts[domain] += 1
                        if len(batch) == 4096:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                            batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    manifest = {"input": str(input_dir), "output": str(output), "counts": dict(counts)}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "data/UltraData-RL-2609/files")
    parser.add_argument("--output", type=Path, default=Path.home() / "data/ultradata-opd/train.parquet")
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
