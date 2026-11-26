# Shared RWKV training code

This directory contains the single canonical copies of rollout, value-data,
FlashRWKV2 preflight, and small data utilities used by the method launchers.

RWKV launchers share the upstream FlashRWKV2 cache at `~/.cache/flashrwkv2`.
Set `FLASHRWKV2_CACHE_DIR` before starting the driver to select another shared
directory. The driver forwards this setting and explicitly configured
Torch/XDG cache paths to Ray actors and rollout subprocesses.

`prepare_flashrwkv2.py` and the `--prepare-flashrwkv2` launcher option prepare
the native extension. Use the same Python executable for preparation and rollout
so both resolve the same native cache entry. vLLM warms RWKV model and sampling
kernels before CUDA Graph capture. TileLang manages its own JIT cache.
