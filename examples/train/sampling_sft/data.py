"""Streaming data and resumable JSONL output for the sampling-SFT recipe.

Prototypes: examples/train/rwkv/rollout.py (UltraData streaming / UUID resume)
and examples/train/sft/prepare_apigen_mt.py (HF dataset ingestion).
"""

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def load_rows(dataset, *, config=None, split="train", revision=None):
    """Read local JSONL/Parquet shards or a streaming Hugging Face split."""
    path = Path(dataset).expanduser()
    if path.exists():
        files = sorted(p for p in path.rglob("*") if p.suffix in {".jsonl", ".parquet"}) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"No JSONL or Parquet files found: {path}")
        for file in files:
            if file.suffix == ".jsonl":
                with file.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
            elif file.suffix == ".parquet":
                from pyarrow.parquet import ParquetFile

                for batch in ParquetFile(file).iter_batches(batch_size=64):
                    yield from batch.to_pylist()
            else:
                raise ValueError(f"Unsupported input file: {file}; expected JSONL or Parquet")
    else:
        if path.suffix in {".jsonl", ".parquet"} or dataset.startswith(("/", "~", ".")):
            raise FileNotFoundError(path)
        from datasets import load_dataset

        yield from load_dataset(dataset, name=config, split=split, revision=revision, streaming=True)


def expert_rows(dataset, *, subset=None):
    """Read existing traces or Nemotron messages with their complete expert reasoning."""
    for row in load_rows(dataset):
        if subset is not None and row.get("subset") != subset:
            continue
        if "solution" in row:
            yield row
            continue
        messages = row["messages"]
        user, assistant = messages
        if user["role"] != "user" or assistant["role"] != "assistant":
            raise ValueError(f"Expected a user/assistant expert trace for {row['uuid']}")
        reasoning = assistant.get("reasoning_content") or ""
        content = assistant.get("content") or ""
        yield {
            "uuid": row["uuid"],
            "question_id": hashlib.sha256(row["problem"].encode()).hexdigest(),
            "prompt": user["content"],
            "solution": "\n".join(part for part in (reasoning, content) if part),
            "reasoning_content": reasoning,
            "content": content,
            "ground_truth": row["expected_answer"],
            "domain": "Math",
        }


def questions(args):
    """Apply the limit before resume, and shard the domain-filtered source."""
    selected = 0
    for row in load_rows(args.dataset, config=args.dataset_config, split=args.dataset_split, revision=args.revision):
        if args.domain and row.get("domain") not in args.domain:
            continue
        if args.limit and selected >= args.limit:
            break
        index = selected
        selected += 1
        if index % args.num_shards != args.shard_index:
            continue
        prompt = row.get(args.question_key)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row {index}: {args.question_key!r} must contain a non-empty question string")
        uuid = row.get(args.id_key)
        yield {
            "uuid": str(uuid) if uuid is not None else hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "domain": str(row.get("domain", "")),
            "ground_truth": row.get(args.answer_key),
        }


def validation_question(uuid, seed, fraction):
    """Keep every trajectory of a question in the same held-out partition."""
    digest = hashlib.sha256(f"{seed}:{uuid}".encode()).digest()
    return int.from_bytes(digest, "big") / 2**256 < fraction


@contextmanager
def open_output(path, config, resume, *, execution_keys=()):
    """Resume exact records, recording explicitly allowed execution-setting changes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_suffix(path.suffix + ".meta.json")
    completed = set()
    if resume:
        previous = json.loads(metadata.read_text())
        changed = {
            key
            for key in previous.keys() | config.keys()
            if key not in previous or key not in config or previous[key] != config[key]
        }
        if changed - set(execution_keys):
            raise ValueError("Resume configuration differs; use the original settings or a new output path")
        with path.open("r+b") as handle:
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    handle.truncate(handle.tell() - len(line))
                    break
                completed.add(json.loads(line)["uuid"])
        if changed:
            history = path.with_suffix(path.suffix + ".meta.history.jsonl")
            with history.open("a", encoding="utf-8") as handle:
                append_jsonl(
                    handle,
                    {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "completed_records": len(completed),
                        "previous": previous,
                        "current": config,
                    },
                )
            metadata.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    else:
        if path.exists() or metadata.exists():
            raise FileExistsError(f"Output exists: {path}; use --resume or a new output path")
        metadata.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    with path.open("a", encoding="utf-8") as handle:
        yield handle, completed


def append_jsonl(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
