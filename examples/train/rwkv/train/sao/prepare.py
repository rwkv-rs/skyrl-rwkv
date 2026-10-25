"""Prepare the UltraData parquet input used by the SAO launcher."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from examples.train.rwkv.train.opd.prepare import main  # noqa: I001


if __name__ == "__main__":
    from examples.train.rwkv.train.opd.prepare import main

    main()
