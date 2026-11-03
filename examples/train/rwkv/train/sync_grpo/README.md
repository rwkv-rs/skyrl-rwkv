# Synchronous RWKV7 GRPO

Full-parameter RWKV7 training using SkyRL's synchronous Trainer, Gym generator
and colocated vLLM engines. Completed rollouts are persisted before selection,
so a large prompt pool does not retain every response as Python objects.

## Usage

Install the repository's RWKV environment using [the setup guide](../../README.md)
and prepare FlashRWKV2 tuning using [the common workflow](../common/README.md).
The isolated scorer requires Linux and `psutil`, supplied by Ray.

```bash
MODEL_DIR=/path/to/rwkv7-g1l-1.5b-20261031-ctx34304 \
DATA_DIR=/path/to/filtered-dapo-parquet \
NUM_GPUS=8 bash examples/train/rwkv/train/sync_grpo/run.sh
```

`DATA_DIR` must contain SkyRL-format `train.parquet` and `validation.parquet`.
The launcher does not silently download or filter data. The measured dataset
contained 4,763 filtered training prompts and 500 validation prompts.

Set `RESUME_PATH=/path/to/checkpoints/global_step_6` to restore training state.
DDP loads model/Adam state one rank at a time, releasing each CPU copy before
the next rank loads. Full-replica checkpoints support different world sizes,
subject to batch/topology validation. `trainer.resume_optimizer_states=false`
starts with a fresh optimizer. The separate
`trainer.policy.optimizer_config.resume_hyperparameters_from_config=true`
override preserves Adam moments but replaces hyperparameters; it requires a
constant schedule without warmup.

Logs, evaluations, scoring requests and rollout dumps go directly to
`ARTIFACT_ROOT` (default `~/skyrl-rwkv-artifacts/$RUN_NAME`). They are persistent
and separate from retained checkpoints in `OUTPUT_ROOT`. Preserve them during
bulk checkpoint cleanup. A lock prevents concurrent launches into one output
directory. No server paths, data or checkpoints are included in this recipe.

## Sampling

- Train 512 informative prompt groups, 16 responses each: 8,192 selected rows.
- Draw up to 2,048 distinct prompts per wave, with replacement across waves and
  steps. Each engine permits 1,024 running sequences and 2,048 client requests;
  queued work refills available inference slots.
- Stop admitting new groups after the target is reached, while admitted groups
  finish. Select surplus informative groups uniformly, rather than taking the
  fastest completions. If too few groups finish, draw another wave, up to 10.
- Persist each completed trajectory as gzip-compressed Torch data with compact
  token, mask and FP32 logprob arrays. Retain only references/rewards until
  selection, then reload selected payloads. Repeated source prompts receive
  unique training-group IDs to preserve GRPO and contiguous minibatch boundaries.
- Pack sampled FP32 logprobs through `/skyrl/v1/generate` with `logprobs=0`.
  `old_logprobs_source=rollout` fixes this denominator for all updates and skips
  the old-policy forward. Current-policy forward/backward still runs.
- Judge strict mathematical answers in reusable isolated processes with a
  3-second deadline. Timeout gives reward zero and replaces the worker; other
  resource failures remain errors. Exact modular counterexamples reject huge
  unequal integer expressions without expanding them. Matching residues never
  prove equality.

This bounds completed-result retention, not active-request or recurrent-state
memory. Inference utilization can still fall when the reservoir drains and only
a few long requests remain.

## Loss and training

GRPO advantages feed token-mean PPO with asymmetric clipping `[0.8, 1.28]`, one
update epoch, constant LR `2e-7` and gradient clipping at 1.0. There is no critic,
reference KL, entropy bonus or adapter training. `github_strict_loss=true`
selects the experiment's population-standard-deviation GRPO normalization and
direct exponential PPO ratio. These numerical conventions are opt-in. A
2,048-token overlong buffer subtracts a linear reward penalty near the
16,383-token response budget; raw task correctness is logged separately.

DDP keeps FP32 master parameters, gradients and Adam moments with BF16 compute
views. It accumulates gradients under `no_sync()` until the last microbatch.
This avoids FSDP2 DTensor optimizer overhead when the replica fits on each GPU.

Microbatches are sorted by valid prompt-plus-response length. The padded
recurrent work must satisfy `batch_size * padded_length <= 65535`. Four rows
therefore become three for long inputs; partial batches may have one/two rows.
Internal tail padding has zero loss and cannot affect earlier causal outputs.
Prompts still enter the recurrent forward despite having masked loss.

| Environment variable | Recipe value | Effect |
|---|---:|---|
| `SKYRL_RWKV_LENGTH_BUCKET_SIZE` | `512` | Reuse recurrent shapes; up to 511 bucket-padding tokens |
| `SKYRL_BALANCED_DP_PACKING` | `1` | Balance aggregate padded work across ranks within each minibatch |
| `SKYRL_RWKV_PREFETCH` | `1` | Pin CPU tensors; prefetch one DDP microbatch on a copy stream |
| `SKYRL_DEFER_LOSS_METRICS` | `1` | Defer detached PPO metric conversion to CPU |
| `SKYRL_POLICY_ENTROPY_METRICS` | `0` | Skip diagnostic entropy; entropy loss still forces computation |
| `SKYRL_ROLLOUT_LOGPROB_DIFF_METRICS` | `0` | Skip per-microbatch logprob-difference diagnostics |
| `SKYRL_RWKV_TRANSPOSED_GEMM_BACKEND` | `cublas` | FP32-accumulating cuBLAS for transposed BF16 projections |

DP packing currently estimates this four-row, 65,535-token recipe. It balances
total rank work, not individual slots or measured kernel time. Residual power
waves alone do not establish time spent waiting for ranks: no per-microbatch
cross-rank timing trace was collected in the historical experiment.

## Historical measurements

Eight RTX PRO 6000 Blackwell 96-GB GPUs, RWKV7 G1l 1.5B:

| Step | Mean response tokens | Policy train seconds | Full step seconds |
|---|---:|---:|---:|
| 1 | 10,992.6 | 2,418.96 | 5,002.36 |
| 2 | 10,487.6 | 2,183.61 | 4,417.59 |
| 3 | 10,042.1 | 2,117.54 | 4,328.05 |
| 4 | 9,012.3 | 1,620.52 | 3,466.21 |
| 5 | 8,391.9 | 1,493.43 | 3,414.87 |

Different batches and incremental optimizations confound attribution of these
speedups. A separate fixed-selection replay of 8,192 trajectories compared
TileLang plus entropy diagnostics against cuBLAS plus disabled diagnostics:
policy training decreased from 4,413.56 to 2,262.72 seconds, with all 16 updates
completing. That comparison does not isolate those two changes from each other.

The historical runtime used Torch `2.13.0+cu130` and RWKV vLLM revision
`2852a7b1d597f6e2131ec801de331b886471cd95`. Current dependency pins have moved;
fresh CUDA, weight-sync and end-to-end validation is required against them.
Step 6 saved successfully; step 7 failed during rollout with a GPU4 device/driver
failure. This change does not implement GPU recovery, engine auto-restart,
training CUDA graphs or slot-level DP balancing.
