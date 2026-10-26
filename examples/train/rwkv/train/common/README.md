# Shared RWKV training code

This directory contains the single canonical copies of rollout, value-data,
FlashRWKV2 preflight, and small data utilities used by the method launchers.

RWKV launchers share the upstream FlashRWKV2 cache at `~/.cache/flashrwkv2`.
Set `FLASHRWKV2_CACHE_DIR` before starting the driver to select another shared
directory. The driver forwards this setting, `VLLM_RWKV_TUNING_MODE`,
`VLLM_RWKV_TUNING_LEGACY_DIR`, and explicitly configured Torch/XDG cache paths
to Ray actors; rollout subprocesses preserve them as well.

The launchers default to `VLLM_RWKV_TUNING_MODE=require`. vLLM checks the
actual RWKV dispatches before CUDA Graph capture using FlashRWKV2's public
`prepare_tuning` API. Missing shape, dtype, or layout coverage stops startup
with the missing key and cache path. Each worker logs the upstream coverage
report, including the selected configurations and their fingerprints.

Prepare missing configurations explicitly with `VLLM_RWKV_TUNING_MODE=tune`
using the same model, engine settings, and representative prefill/decode
requests as the intended workload. Existing verified keys are reused. To
reuse an old namespace, also set `VLLM_RWKV_TUNING_LEGACY_DIR` to its exact
directory; upstream revalidates each matching candidate before publishing it
in the shared cache. Restart with `require` after preparation so the serving
process loads the verified selections before capturing its graphs.

`prepare_flashrwkv2.py` and the `--prepare-flashrwkv2` launcher option prepare
the native extension. Their `cached` result describes that extension;
model dispatch coverage remains pending until the vLLM worker runs its
preparation. Verified tuning profiles and native libraries have separate
compatibility keys in the shared root. TileLang manages its own JIT cache.
