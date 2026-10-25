"""Compatibility entry point for the shared FlashRWKV2 preflight."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from examples.train.rwkv.train.common.prepare_flashrwkv2 import check, prepare  # noqa: I001


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--check"]:
        check()
    else:
        prepare()
