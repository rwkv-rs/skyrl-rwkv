"""Filter normalized RL source rows whose RWKV rollout pass rate is exactly one.

The rollout stage writes one record per question to
``correct_counts_by_question.jsonl``.  This module joins those records back to
the original JSONL shards and publishes a copy of the dataset with only fully
solved questions removed.  A question with any wrong or unanswered rollout is
kept for RWKV training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

DOMAINS = ("Math", "Code", "Long_Context", "Knowledge")
COUNTS_FILE = "correct_counts_by_question.jsonl"
STATUSES = frozenset(("correct", "wrong", "unanswered"))


def load_rollout_counts(
    path: Path,
    *,
    expected_samples: int | None = None,
) -> dict[str, tuple[str, int, int]]:
    """Read per-question rollout counts.

    Returns a mapping from UUID to ``(domain, correct, total)``.  The counts
    are validated before any output files are created so an interrupted or
    malformed rollout cannot silently produce a partial training dataset.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Rollout counts file does not exist: {path}")
    results: dict[str, tuple[str, int, int]] = {}
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = orjson.loads(line)
            uuid = record.get("uuid")
            domain = record.get("domain")
            counts = record.get("counts")
            if not isinstance(uuid, str) or not uuid:
                raise ValueError(f"{path}:{line_number}: rollout record has no UUID")
            if uuid in results:
                raise ValueError(f"{path}:{line_number}: duplicate rollout result for {uuid}")
            if domain not in DOMAINS:
                raise ValueError(f"{path}:{line_number}: unknown domain {domain!r} for {uuid}")
            if not isinstance(counts, dict) or not counts:
                raise ValueError(f"{path}:{line_number}: missing counts for {uuid}")
            if set(counts) - STATUSES:
                raise ValueError(f"{path}:{line_number}: unknown rollout status for {uuid}: {counts!r}")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
                raise ValueError(f"{path}:{line_number}: invalid counts for {uuid}: {counts!r}")
            total = sum(counts.values())
            correct = counts.get("correct", 0)
            if total <= 0 or correct > total:
                raise ValueError(f"{path}:{line_number}: invalid rollout total for {uuid}: {counts!r}")
            if expected_samples is not None and total != expected_samples:
                raise ValueError(f"{path}:{line_number}: expected {expected_samples} rollouts for {uuid}, got {total}")
            results[uuid] = (domain, correct, total)
    if not results:
        raise ValueError(f"Rollout counts file is empty: {path}")
    return results


def filter_dataset(
    input_dir: Path,
    rollout_output: Path,
    output_dir: Path,
    *,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Copy ``input_dir`` while dropping only questions with pass rate 1.

    The destination is assembled in a sibling temporary directory and renamed
    into place after every source question has a matching rollout result.  An
    existing destination is rejected to avoid mixing results from different
    checkpoints or rollout settings.
    """

    input_dir = input_dir.resolve()
    rollout_output = rollout_output.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dataset directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output dataset directories must be different")
    if output_dir.exists():
        raise FileExistsError(f"Output dataset directory already exists: {output_dir}")

    counts_path = rollout_output / COUNTS_FILE
    results = load_rollout_counts(counts_path, expected_samples=expected_samples)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary output directory already exists: {temporary}")

    per_domain: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    seen: set[str] = set()
    try:
        for domain in DOMAINS:
            source_paths = sorted((input_dir / domain).glob("*.jsonl"))
            if not source_paths:
                raise FileNotFoundError(f"No JSONL shards found in {input_dir / domain}")
            for source_path in source_paths:
                destination_path = temporary / domain / source_path.name
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                with source_path.open("rb") as source, destination_path.open("wb") as destination:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        row = orjson.loads(line)
                        uuid = row.get("uuid")
                        row_domain = row.get("domain")
                        if not isinstance(uuid, str) or not uuid:
                            raise ValueError(f"{source_path}:{line_number}: dataset row has no UUID")
                        if uuid in seen:
                            raise ValueError(f"{source_path}:{line_number}: duplicate dataset UUID {uuid}")
                        result = results.get(uuid)
                        if result is None:
                            raise ValueError(f"No rollout result for dataset row {uuid}")
                        result_domain, correct, total = result
                        if row_domain != domain or result_domain != domain:
                            raise ValueError(
                                f"Domain mismatch for {uuid}: shard={domain!r}, row={row_domain!r}, "
                                f"rollout={result_domain!r}"
                            )
                        seen.add(uuid)
                        per_domain[domain]["input_rows"] += 1
                        if correct == total:
                            per_domain[domain]["removed_pass_rate_1"] += 1
                        else:
                            destination.write(line)
                            per_domain[domain]["kept_rows"] += 1

        missing = set(results) - seen
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"Rollout results have {len(missing)} dataset rows missing from input: {preview}{suffix}")

        total_input = sum(values["input_rows"] for values in per_domain.values())
        total_kept = sum(values["kept_rows"] for values in per_domain.values())
        total_removed = sum(values["removed_pass_rate_1"] for values in per_domain.values())
        manifest = {
            "input": str(input_dir),
            "rollout_output": str(rollout_output),
            "rollout_counts": str(counts_path),
            "rule": "remove only rows whose RWKV rollout pass rate is exactly 1.0",
            "expected_samples_per_prompt": expected_samples,
            "domains": {domain: dict(values) for domain, values in per_domain.items()},
            "total_input_rows": total_input,
            "total_kept_rows": total_kept,
            "total_removed_pass_rate_1": total_removed,
        }
        source_manifest = input_dir / "source_manifest.json"
        if source_manifest.is_file():
            (temporary / "source_manifest.json").write_bytes(source_manifest.read_bytes())
            manifest["source_manifest"] = str(source_manifest)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Unfiltered normalized source corpus directory")
    parser.add_argument(
        "--rollout-output",
        type=Path,
        required=True,
        help="Completed rollout output containing correct_counts_by_question.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for filtered shards")
    parser.add_argument(
        "--expected-samples",
        type=int,
        help="Require this many rollouts for every question (the launcher defaults to 64)",
    )
    args = parser.parse_args()
    if args.expected_samples is not None and args.expected_samples < 1:
        parser.error("--expected-samples must be positive")
    manifest = filter_dataset(
        args.input,
        args.rollout_output,
        args.output,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
