# SkyRL

SkyRL is a full-stack reinforcement learning library for training LLMs, designed for modularity and extensibility.

## Critical Rules

- **Always use `uv run --isolated`** to run commands. Never use bare `python`, `pip`, or `pip install`.
- **Log output to files**: `<cmd> > /tmp/results_1.log 2>&1` for persistence.
- Backend extras (`fsdp`, `megatron`, `jax`) conflict with each other -- never combine them.
- Always read the relevant documentation files in `.agents/docs` before troubleshooting or working on any changes. Follow the routing rules below.

## Test Commands

These mirror `.github/workflows/cpu_skyrl.yaml` and `cpu_jax.yaml`; keep them in sync.

```bash
# CPU tests -- skyrl_train. Needs a backend extra for `ray`: without one, collection fails
# in tests/backends/skyrl_train/conftest.py with "No module named 'ray'". CI splits on the
# `vllm` marker because the two halves need different extras.
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "not vllm"
uv run --isolated --extra fsdp --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "vllm"

# CPU tests -- tx / tinker / utils
uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tx tests/backends/test_jax_backend.py --ignore=tests/tx/gpu
uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tinker tests/utils --ignore=tests/tinker/skyrl_train
uv run --isolated --extra fsdp --extra tinker --extra dev pytest tests/tinker/skyrl_train/

# GPU tests (requires Ray cluster with GPUs)
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py

# The opt-in h100 GPU marker is auto-skipped unless requested by name:
uv run --isolated --extra dev --extra megatron pytest -m h100 tests/backends/skyrl_train/gpu/gpu_ci/megatron/

# Lint / format (needs pre-commit; `bash format.sh` fails if it is not on PATH)
uv run --isolated --extra dev pre-commit run --all-files
```

Tests that connect to Ray call bare `ray.init()`, which attaches to any cluster already
running on the box -- including a live training cluster, whose workers then die. Run them
with `RAY_ADDRESS=local` on a machine that has one up.

## Training Quick Start

```bash
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron trainer.policy.model.path=<model> environment.env_class=gsm8k ...
```

## Routing Rules

When working on these areas, read the corresponding doc first:

| Area | Read first |
|------|-----------|
| Package management, uv, formatting | `.agents/docs/development.md` |
| Overall guide for modifying or working with SkyRL | `.agents/docs/contributing.md` |
| Tests, fixtures, CI quirks | `.agents/docs/testing.md` |
| Project layout, Ray actors, config | `.agents/docs/architecture.md` |
| Training entrypoints, configs | `.agents/docs/training.md` |
| Inference engines, vLLM, PD disagg | `.agents/docs/inference.md` |
| GitHub Actions, Anyscale CI | `.agents/docs/ci.md` |
| Tinker API server | `.agents/docs/tinker.md` |
| Megatron backend | `.agents/docs/backends/megatron.md` |
| FSDP backend | `.agents/docs/backends/fsdp.md` |
| JAX/TPU backend | `.agents/docs/backends/jax.md` |
| Weight sync | `.agents/docs/weight_sync.md` |
| Bumping any pinned dependency (e.g. transformer-engine) -- required test matrix | `.agents/docs/dependency_bumps.md` |
| Bumping megatron-core / megatron-bridge, or Megatron patches / vendored code | `skyrl/backends/skyrl_train/patches/megatron/README.md` |


## Troubleshooting

For troubleshooting training runs with SkyRL:

1. Go through the troubleshooting section in the docs for known errors: `docs/content/docs/troubleshooting/troubleshooting.mdx`
2. Go through the contributing guide for overall guidelines: `.agents/docs/contributing.md`


## RWKV Project Integration Rules

Please Read ./CLAUDE.md first.

## Core Objectives
SkyRL is a mainstream reinforcement learning library in the LLM community. This repository needs to integrate the RWKV model through SkyRL's native Trainer, Generator, InferenceEngine, and Environment interfaces.
Code principle: For every file/type/function/variable, a similar implementation must be found to serve as a prototype. If that prototype carries a model name, replace it with `RWKV` or another case variant; otherwise keep the same name.
Process principle: Before starting any work on new functionality, first read the official documentation: https://docs.skyrl.ai/docs/

## Directory Conventions

```text
.
├── .agents/...
├── .claude/...
├── .codex/...
├── .gemini/...
├── .git/...
├── .github/...
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── AGENTS.md
├── CLAUDE.md
├── LICENSE
├── README.md
├── ci/...
├── docker/...
├── docs/...
├── examples/
│   ├── ...
│   └── train/
│       ├── ...
│       └── rwkv/                         # To be added later: RWKV one-click training configuration and launch scripts
├── format.sh
├── integrations/...
├── pyproject.toml
├── skyrl/...                             # Reuse native FSDP, Trainer, Generator, vLLM inference, and weight synchronization
├── skyrl-agent/...
├── skyrl-gym/...                         # SkyRL native Environment and reward; do not add an environment just because an item is missing from the target list
├── skyrl-train/...
├── skyrl-tx/...
├── tests/
│   ├── ...
│   └── backends/
│       ├── ...
│       └── skyrl_train/
│           ├── ...
│           └── rwkv/                     # To be added later: hermetic tests for the FSDP + transformers-rwkv + vLLM contract
└── uv.lock
```

model_name must clearly specify the exact weight version for Qwen (e.g., Qwen3.5-2B) / RWKV7 (see the `RWKV7 Weights` section for details).
Adding any new file requires user confirmation.

## Authoritative RWKV7 Implementations
(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- Authoritative low-level inference engine implementation repository (CUDA, for PRO6000, no scheduling, no varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- Authoritative pretraining implementation repository (CUDA, for H100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Mathematical principles of Diagonal Plus Low Rank (DPLR): parallel computation of explicit transition matrices
(6) **https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv** -- Authoritative RWKV Hugging Face Transformers adaptation repository (with Rust tokenizer, 10x faster than the Python implementation)
(7) **https://github.com/rwkv-rs/vllm-rwkv/tree/rwkv** -- Authoritative RWKV vLLM adaptation repository

## RWKV7 Weights
General weight naming convention: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
Example: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: architecture version, e.g., rwkv7 (default), rwkv7a (experimental, rwkv7 with DeepEmbed), rwkv7b (experimental, rwkv7 with DeepEmbedAttn)
data_version: data version, e.g., g1a, g1b... (The further back in the alphabet, the better)
param_size: parameter size; only 0.1b, 0.4b, 1.5b (often used in RL), 2.9b, 7.2b (often used in inference tests), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- Authoritative weight Release source (updated every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- Authoritative weight Test source (updated irregularly)
(3) https://huggingface.co/rwkv-rs/rwkv7-g1-st -- Authoritative weight Release source (for transformers)

## Correctness Checks
1. Whether the three sets of Prompt Templates provided in transformers-rwkv and the corresponding rwkv7-g1-st weight repository can be correctly applied
2. Use wkv_mode=fp32io16 by default
3. During rollout, use decoding parameters temp 1, top_p 0.95, top_k -1, with penalty disabled; during eval, use decoding parameters temp 0.96, top_p 0.76, top_k 32, presence_penalty 1.0, frequency_penalty 0.1, penalty_decay 0.988
4. The tokens, logprobs, and loss masks of the rollout policy and trainer policy should be strictly aligned
5. After each optimizer step, the weights of the trainer policy and inference policy should be correctly synchronized
6. The model reward should show a training trend similar to that of a Qwen3.5 model with a similar parameter count

## Result Saving
Save weights and run evaluation every 50 steps; upload detailed training and evaluation data to wandb to plot curves.

## Env
Use uv to manage the local and remote dedicated environment ./.venv. This project is strictly prohibited from using other environments, and other projects are strictly prohibited from using this project's environment, to avoid environment pollution issues.

## Machine for Testing and Benchmarking
```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/skyrl-rwkv
```
use git to sync your changes instead of rsync.
