"""
Environment variable configuration for SkyRL.

All environment variables used by SkyRL should be defined here for discoverability.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Ray / Placement Group
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_RAY_PG_TIMEOUT_IN_S = int(os.environ.get("SKYRL_RAY_PG_TIMEOUT_IN_S", 180))
"""
Timeout for allocating the placement group for different actors in SkyRL.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Worker / NCCL
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_WORKER_NCCL_TIMEOUT_IN_S = int(os.environ.get("SKYRL_WORKER_NCCL_TIMEOUT_IN_S", 600))
"""
Timeout for initializing the NCCL process group for the worker, defaults to 10 minutes.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inference Servers
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_VLLM_DP_PORT_OFFSET = int(os.environ.get("SKYRL_VLLM_DP_PORT_OFFSET", 500))
"""
Offset for the data parallel port of the vLLM server.
"""
SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S = int(
    os.environ.get("SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S", 600)
)
"""
Timeout for waiting until the inference server is healthy.
"""

SKYRL_INFERENCE_CONTROL_PLANE_TIMEOUT_S = float(os.environ.get("SKYRL_INFERENCE_CONTROL_PLANE_TIMEOUT_S", 600))
"""
Per-request timeout for control-plane calls from `RemoteInferenceClient` to the inference servers (pause, resume,
sleep, wake_up, weight updates, LoRA load/unload, ...). Defaults to `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`'s default.
``<= 0`` disables it. Data-plane requests (generate, chat completions, ...) are not bounded.
"""

SKYRL_HTTP_CONNECTION_LIMIT = int(os.environ.get("SKYRL_HTTP_CONNECTION_LIMIT", 50_000))
"""
Maximum number of concurrent HTTP connections for the inference client, router,
and server.

This controls:
- aiohttp TCPConnector limit in `RemoteInferenceClient`
- connection pool limits in the router
- uvicorn TCP backlog in the router and vLLM server
"""

SKYRL_GENERATE_CONCURRENCY_PER_ENGINE = int(os.environ.get("SKYRL_GENERATE_CONCURRENCY_PER_ENGINE", 512))
"""
Maximum number of concurrent generate tasks per inference engine.

The effective concurrency limit is ``SKYRL_GENERATE_CONCURRENCY_PER_ENGINE * num_engines``.
Large batch sizes (e.g. 5120) can overwhelm the router's single-threaded
event loop and vLLM's accept queue when all requests fire simultaneously.
We ensure that at most this many tasks per engine are in-flight at
once; the rest queue locally and proceed as slots free up.

Set to 0 to disable throttling (all tasks fire immediately).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Tinker
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_TINKER_CONTINUOUS_SAMPLING = str(os.environ.get("SKYRL_TINKER_CONTINUOUS_SAMPLING", "True")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether the Tinker engine completes each sample future as soon as its own
generation finishes (continuous sampling) instead of batching pending sample
requests and completing them together once the whole batch is done.

Only takes effect with backends that expose the per-request async sample path
(the skyrl-train backends); other backends always use the batched loop.

Default: True. Set ``SKYRL_TINKER_CONTINUOUS_SAMPLING=0`` to use the batched loop.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Runtime Environment Exports
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_LD_LIBRARY_PATH_EXPORT = str(os.environ.get("SKYRL_LD_LIBRARY_PATH_EXPORT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to export ``LD_LIBRARY_PATH`` environment variable from the driver to the workers with Ray's runtime env.

For example, if you are using RDMA, you may need to customize the ``LD_LIBRARY_PATH`` to include the RDMA libraries (Ex: EFA on AWS).
"""

SKYRL_PYTHONPATH_EXPORT = str(os.environ.get("SKYRL_PYTHONPATH_EXPORT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Whether to export ``PYTHONPATH`` environment variable from the driver to the workers with Ray's runtime env.

See https://github.com/ray-project/ray/issues/56697 for details on why this is needed.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_DISABLE_FA4 = str(os.environ.get("SKYRL_DISABLE_FA4", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
Force Transformer Engine to ignore FlashAttention 4, falling back to FA2 (or
cuDNN fused attention). FA4 is opt-in via the ``fa4`` extra -- SkyRL ships FA4's
kernels inside the combined ``flash-attn`` wheel, but TE only enables them when
the metadata-only ``flash-attn-4`` companion is installed. This variable turns
FA4 back off for an environment that already has it, without re-resolving.

Useful for A/B-ing FA2 against FA4 without rebuilding the environment, and as an
escape hatch if an FA4 kernel misbehaves on a shape SkyRL exercises.

Default: False (use FA4 where supported).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

SKYRL_DUMP_INFRA_LOG_TO_STDOUT = str(os.environ.get("SKYRL_DUMP_INFRA_LOG_TO_STDOUT", "False")).lower() in (
    "true",
    "1",
    "yes",
)
"""
When enabled, infrastructure logs (vLLM, Ray, workers) are shown on stdout
instead of being redirected to the log file. Useful for debugging startup issues.

Default: False (infrastructure logs go to file only, stdout shows training progress).
Set ``SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`` to show all logs on stdout.
"""

SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC = float(os.environ.get("SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", 300))
"""
Read timeout in seconds for API-side requests forwarded to the SkyRL-Train-managed
inference engine. This is applicable for SkyRL's Tinker server in non-colocated setups.

The timeout must cover time spent queued behind other requests as well as generation time.
Equivalent to the ``--forwarding-inference-timeout-sec`` flag of the Tinker API server
(``EngineConfig.forwarding_inference_timeout_sec``); the flag takes precedence when both are set.
"""

SKYRL_MEGATRON_RANDOM_INIT = str(os.environ.get("SKYRL_MEGATRON_RANDOM_INIT", "0")).lower() in ("1", "true")
"""Build the Megatron model from the HF config with random weights instead of loading the checkpoint.

For throughput/memory benchmarking (e.g. dummy SFT with random tokens), where only the architecture
matters: ``model.path`` then needs just ``config.json`` and tokenizer files, no safetensors."""

SKYRL_RWKV_LENGTH_BUCKET_SIZE = max(0, int(os.environ.get("SKYRL_RWKV_LENGTH_BUCKET_SIZE", "0")))
"""Optional recurrent length bucket in tokens; zero retains 16-token alignment only."""

SKYRL_BALANCED_DP_PACKING = os.environ.get("SKYRL_BALANCED_DP_PACKING", "0") == "1"
"""Balance padded RWKV work across policy ranks when training against rollout logprobs."""

SKYRL_RWKV_PREFETCH = os.environ.get("SKYRL_RWKV_PREFETCH", "0") == "1"
"""Prefetch one tensor-only CPU microbatch through pinned memory for RWKV DDP."""

SKYRL_RWKV_TRANSPOSED_GEMM_BACKEND = os.environ.get("SKYRL_RWKV_TRANSPOSED_GEMM_BACKEND", "tilelang")
"""Select tilelang or cublas for RWKV transposed BF16 projections."""

SKYRL_DEFER_LOSS_METRICS = os.environ.get("SKYRL_DEFER_LOSS_METRICS", "0") == "1"
"""Defer RWKV PPO scalar conversion until microbatch metrics are reduced."""

SKYRL_POLICY_ENTROPY_METRICS = os.environ.get("SKYRL_POLICY_ENTROPY_METRICS", "1") == "1"
"""Compute policy entropy for diagnostics; entropy loss always requires entropy."""

SKYRL_ROLLOUT_LOGPROB_DIFF_METRICS = os.environ.get("SKYRL_ROLLOUT_LOGPROB_DIFF_METRICS", "1") == "1"
"""Compute per-microbatch differences between training and rollout logprobs."""

SKYRL_COMPACT_ROLLOUT_LOGPROBS = os.environ.get("SKYRL_COMPACT_ROLLOUT_LOGPROBS", "0") == "1"
"""Transport sampled-token FP32 logprobs as a packed array when logprobs=0."""

SKYRL_ROLLOUT_ARCHIVE = os.environ.get("SKYRL_ROLLOUT_ARCHIVE")
"""Optional persistent directory for generated and completed rollout artifacts."""

SKYRL_ISOLATED_MATH_SCORING = os.environ.get("SKYRL_ISOLATED_MATH_SCORING", "0") == "1"
"""Enable Linux subprocess isolation for strict mathematical scoring."""

SKYRL_MATH_WORKERS = int(os.environ.get("SKYRL_MATH_WORKERS", "8"))
"""Maximum concurrent isolated mathematical scoring workers."""

SKYRL_MATH_TIMEOUT_SECONDS = max(0.1, float(os.environ.get("SKYRL_MATH_TIMEOUT_SECONDS", "3")))
"""Deadline for an isolated scoring attempt; timeouts receive reward zero."""

SKYRL_MATH_ARCHIVE = os.environ.get("SKYRL_MATH_ARCHIVE")
"""Persistent scoring audit directory, required when isolated scoring is enabled."""
