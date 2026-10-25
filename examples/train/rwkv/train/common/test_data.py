"""Small, streaming dataset operations used by the RWKV smoke launchers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prefix_jsonl(source: Path, destination: Path, rows: int) -> int:
    """Copy at most ``rows`` non-empty JSONL records without reading the file fully."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open() as source_stream, destination.open("w") as destination_stream:
        for line in source_stream:
            if not line.strip():
                continue
            destination_stream.write(line)
            count += 1
            if count == rows:
                break
    return count


def prefix_parquet(source: Path, destination: Path, rows: int) -> int:
    """Copy a parquet prefix while retaining the source schema and compression."""
    import pyarrow as pa
    from pyarrow import parquet

    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = parquet.ParquetFile(source)
    remaining = rows
    batches = []
    for batch in reader.iter_batches(batch_size=min(rows, 4096)):
        if remaining <= 0:
            break
        batch = batch.slice(0, min(batch.num_rows, remaining))
        batches.append(batch)
        remaining -= batch.num_rows
    if not batches:
        raise ValueError(f"{source} does not contain any rows")
    parquet.write_table(
        pa.Table.from_batches(batches, schema=reader.schema_arrow),
        destination,
        compression="zstd",
    )
    return rows - remaining


def prefix_dataset_dir(source: Path, destination: Path, rows: int) -> None:
    """Create a small pretokenized SFT store while preserving split names."""
    train_files = sorted((source / "train").rglob("*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"no train parquet files under {source}")
    remaining = rows
    for path in train_files:
        if remaining <= 0:
            break
        target = destination / "train" / path.relative_to(source / "train")
        copied = prefix_parquet(path, target, remaining)
        remaining -= copied
    eval_files = sorted((source / "eval").rglob("*.parquet"))
    for path in eval_files:
        target = destination / "eval" / path.relative_to(source / "eval")
        prefix_parquet(path, target, rows)
    manifest = source / "manifest.json"
    if manifest.is_file():
        (destination / "manifest.json").write_bytes(manifest.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--format", choices=("parquet", "jsonl"), required=True)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.directory is None and args.source is None:
        parser.error("--source is required unless --directory is used")
    if args.directory is not None:
        prefix_dataset_dir(args.directory, args.destination, args.rows)
        copied = args.rows
    else:
        copied = (
            prefix_parquet(args.source, args.destination, args.rows)
            if args.format == "parquet"
            else prefix_jsonl(args.source, args.destination, args.rows)
        )
    print(
        json.dumps(
            {
                "source": str(args.directory or args.source),
                "destination": str(args.destination),
                "rows": copied,
            }
        )
    )


if __name__ == "__main__":
    main()
