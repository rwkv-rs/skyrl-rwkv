"""Prepare complete Nemotron CoT for native RWKV FSDP SFT.

Follows prepare_apigen_mt.py's messages conversion, sampling_sft/data.py's
question split, and curriculum_sampler.py's checkpointable sampler.

Usage from the repository root::

    uv run --isolated --no-project .venv/bin/python \\
        -m examples.train.rwkv.prepare_sft \\
        --input ~/Datasets/Nemotron-SFT-Math-v4 --output ~/data/rwkv-direct-sft
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import Sampler

from examples.train.sampling_sft.data import expert_rows, validation_question
from skyrl.utils.tok import get_tokenizer


def convert_example(row, tokenizer):
    """Supervise complete expert reasoning, final answer, separator and EOS."""
    if not row["reasoning_content"].strip() or not row["content"].strip():
        raise ValueError(f"Incomplete CoT expert trace: {row['uuid']}")
    messages = [
        {"role": "user", "content": row["prompt"]},
        {
            "role": "assistant",
            "content": f"<think>{row['reasoning_content']}\n</think>\n{row['content']}",
        },
    ]
    template_kwargs = dict(
        tokenize=False,
        rwkv_prompt_template="bot",
        rwkv_generation_prompt="open_think",
    )
    prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, **template_kwargs)
    full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **template_kwargs)
    # Replay rollout's prompt IDs and encode the supervised suffix separately.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(full_text[len(prompt) :], add_special_tokens=False) + [tokenizer.eos_token_id]
    return {
        "uuid": row["uuid"],
        "question_id": row["question_id"],
        "input_ids": prompt_ids + response_ids,
        "loss_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
    }


class LengthGroupedSampler(Sampler):
    """Shuffle batches of similar lengths, visiting every row once per epoch.

    Uses SFTDataset.sequence_lengths without materializing token arrays. The
    final partial batch stays last so native collation can mask its padding.
    """

    def __init__(self, data_source, batch_size, seed=0):
        rng = random.Random(seed)
        lengths = data_source.sequence_lengths
        indices = list(range(len(data_source)))
        rng.shuffle(indices)
        indices.sort(key=lambda index: lengths[index])
        complete = len(indices) // batch_size * batch_size
        batches = [indices[start : start + batch_size] for start in range(0, complete, batch_size)]
        rng.shuffle(batches)
        self._plan = [index for batch in batches for index in batch] + indices[complete:]
        self.position = 0

    def __iter__(self):
        while self.position < len(self._plan):
            index = self._plan[self.position]
            self.position += 1
            yield index
        self.position = 0

    def __len__(self):
        return len(self._plan)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def prepare_shard(args):
    """Stream one source shard to bounded Parquet row groups."""
    source, staging, model, seed, fraction, limit = args
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizer = get_tokenizer(model)
    schema = pa.schema(
        [
            ("uuid", pa.string()),
            ("question_id", pa.string()),
            ("input_ids", pa.list_(pa.int32())),
            ("loss_mask", pa.list_(pa.int8())),
        ]
    )
    counts = Counter()
    questions = {"train": set(), "validation": set()}
    lengths = {"train": [], "validation": []}
    buffers = {"train": [], "validation": []}
    writers = {}

    def flush(split):
        records = buffers[split]
        if not records:
            return
        # Evaluation is sequential; nearby lengths reduce padded recurrent work.
        records.sort(key=lambda record: len(record["input_ids"]))
        if split not in writers:
            writers[split] = pq.ParquetWriter(
                str(Path(staging) / split / Path(source).name), schema, compression="zstd"
            )
        writers[split].write_table(pa.Table.from_pylist(records, schema=schema), row_group_size=32)
        records.clear()

    try:
        for index, row in enumerate(expert_rows(source, subset="cot")):
            if limit is not None and index >= limit:
                break
            split = "validation" if validation_question(row["question_id"], seed, fraction) else "train"
            record = convert_example(row, tokenizer)
            length = len(record["input_ids"])
            lengths[split].append(length)
            questions[split].add(row["question_id"])
            counts[f"{split}_rows"] += 1
            counts[f"{split}_tokens"] += length
            counts[f"{split}_supervised_tokens"] += sum(record["loss_mask"])
            buffers[split].append(record)
            if len(buffers[split]) >= 256:
                flush(split)
            if (index + 1) % 1000 == 0:
                print(f"{Path(source).name}: {index + 1} complete CoT rows", flush=True)
        for split in buffers:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()
    return counts, questions, lengths


def prepare_data(args):
    """Publish tokenized splits and provenance only after every shard succeeds."""
    sources = sorted(args.input.expanduser().rglob("*.parquet"))
    if not sources:
        raise ValueError(f"No Nemotron Parquet shards found in {args.input}")
    if args.limit is not None:
        sources = sources[:1]
    output = args.output.expanduser()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    questions = {"train": set(), "validation": set()}
    lengths = {"train": [], "validation": []}
    with tempfile.TemporaryDirectory(prefix=".direct-sft-", dir=output.parent) as directory:
        staging = Path(directory)
        for split in questions:
            (staging / split).mkdir()
        tasks = [
            (str(source), directory, str(args.model.expanduser()), args.seed, args.validation_fraction, args.limit)
            for source in sources
        ]
        with ProcessPoolExecutor(
            max_workers=min(args.num_workers, len(tasks)), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for shard_counts, shard_questions, shard_lengths in pool.map(prepare_shard, tasks):
                counts.update(shard_counts)
                for split, ids in questions.items():
                    ids.update(shard_questions[split])
                    lengths[split].extend(shard_lengths[split])
        if not counts["train_rows"] or not counts["validation_rows"]:
            raise ValueError("Question split produced an empty train or validation dataset.")
        # A fixed held-out panel bounds the cost of evaluation every 50 steps.
        validation_rows = []
        for source in sorted((staging / "validation").glob("*.parquet")):
            table = pq.read_table(source, columns=["uuid", "question_id"])
            validation_rows.extend((row["uuid"], source, index) for index, row in enumerate(table.to_pylist()))
        validation_rows.sort(key=lambda item: hashlib.sha256(f"{args.seed}:{item[0]}".encode()).digest())
        selected = validation_rows[: args.eval_rows]
        by_source = {}
        for _, source, index in selected:
            by_source.setdefault(source, []).append(index)
        eval_tables = [pq.read_table(source).take(indices) for source, indices in by_source.items()]
        eval_table = pa.concat_tables(eval_tables)
        eval_lengths = pc.list_value_length(eval_table["input_ids"])
        eval_table = eval_table.take(pc.sort_indices(eval_lengths))
        (staging / "eval").mkdir()
        pq.write_table(eval_table, staging / "eval" / "validation.parquet", compression="zstd", row_group_size=32)
        tokenizer = get_tokenizer(str(args.model.expanduser()))
        manifest = {
            "source": str(args.input.expanduser().resolve()),
            "source_files": [{"path": str(source.resolve()), "bytes": source.stat().st_size} for source in sources],
            "model": str(args.model.expanduser().resolve()),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "subset": "cot",
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "max_length": None,
            "counts": dict(counts),
            "questions": {split: len(ids) for split, ids in questions.items()},
            "length_percentiles": {
                split: np.percentile(values, [0, 50, 90, 99, 100]).tolist() for split, values in lengths.items()
            },
            "eval_rows": len(selected),
            "eval_selection": "SHA256(seed:uuid), lowest hashes from validation partition",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.replace(output)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path.home() / "Datasets/Nemotron-SFT-Math-v4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path.home() / "Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument(
        "--limit", type=int, help="Prepare only this many CoT rows from the first shard for a smoke test"
    )
    args = parser.parse_args()
    if not math.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.num_workers < 1 or args.eval_rows < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--num-workers, --eval-rows and --limit must be positive")
    prepare_data(args)


if __name__ == "__main__":
    main()
