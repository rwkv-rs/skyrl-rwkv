"""Prepare a bounded Antidoom JSONL prefix for ``--test_mode`` runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from examples.train.rwkv.train.common.test_data import prefix_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    print(f"copied {prefix_jsonl(args.source, args.destination, args.rows)} rows")


if __name__ == "__main__":
    main()
