# RWKV training recipes

The RWKV example is organized by responsibility. Each training method has one
`README.md`, `prepare.py`, and `run.sh`; shared rollout, data, evaluation, and
FlashRWKV2 preflight code lives in `train/common` or `eval`.

The five launchers are:

- `train/antidoom`
- `train/flashreinforce`
- `train/opd`
- `train/sao`
- `train/sft`

Use `--test_mode` for a bounded data prefix and five optimizer steps. Test mode
disables W&B and uses console logging. Normal runs force W&B on. The default
training settings target eight Pro6000 GPUs and use `wkv_mode=fp32io16`.

`eval/run.sh` starts eight one-GPU vLLM-RWKV workers behind one router and
reuses the stable `lighteval-rwkv` producer for GSM8K, MATH500, AIME24, AIME25,
and AIMO avg@k/pass@k evaluation.

Evaluation defaults to 128 concurrent sequences per worker, matching the OPD
launcher. RWKV startup profiles CUDA graphs before allocating the final state
cache budget: the 61-layer, 4096-hidden model needs about 61.95 MiB of FP32 state
and BF16 shifts per sequence. Its 1024-sequence startup state alone uses 61.95 GiB;
128 sequences use 7.74 GiB, leaving room for weights and tuning workspaces.
The context remains 32768 tokens. Chunked prefill schedules at most 8192 tokens
per batch without truncating the context. FULL CUDA graphs use power-of-two
capture sizes from 1 through 8192, so batches pad up to a prepared graph size.
This keeps verified profile coverage while bounding the number and size of
startup tuning workloads.

Before evaluation on a new GPU/model profile, run `eval/run.sh --prepare-flashrwkv2`
with the same `MODEL`, `OUTPUT` and serving environment settings. It starts the
same eight workers with `VLLM_RWKV_TUNING_MODE=tune`, waits for every worker and
the router to become healthy, then stops them without calling the producer.
Profiles are saved in `FLASHRWKV2_CACHE_DIR` (default: `~/.cache/flashrwkv2`).
Preparation allows up to 7200 seconds for startup; the calling job enforces its
overall budget. Evaluation requires verified profiles with tuning mode `require`
and allows 900 seconds for startup. Either mode fails immediately when a worker
or router exits and cleans up its remaining processes.

For a complete benchmark manifest, set `BENCHMARKS` to its TOML path and
`LIGHTEVAL_PROJECT` to the pinned producer checkout with its own `.venv`. skyrl-rwkv
calls the producer's public CLI in that environment and reads its score files;
the two projects retain their required grader dependency versions. This mode runs
the producer once with its own metrics, covering every field in the manifest. `SERVED_MODEL_NAME`
sets the model alias used by both vLLM and the producer; `SEED`, `MAX_SAMPLES`,
and `PROMPT_TEMPLATE` configure the evaluation. The existing fixed Math avg/pass
profile remains the default when `BENCHMARKS` is unset and runs in the same isolated
producer environment. skyrl-rwkv publishes the resulting metrics to W&B. The
`eval/prepare.py` adapter accepts `--producer-timeout` in seconds (default: 21600).

SFT, FlashREINFORCE and OPD support a bounded training-and-evaluation lifecycle
through `RWKV_TRAIN_EVAL=<json-path>` and `TRAIN_MAX_SECONDS=<seconds>`. After
training succeeds and releases its Ray resources, the launcher publishes the
specified HF export, runs a probe, and calls the supplied acceptance command.
Only a passing probe runs full confirmation. The JSON contains:

```json
{
  "export": "/run/hf_exports/global_step_5/policy",
  "model": "/weights/rwkv7-g1l-13.3b-heli-r01-sft",
  "env": {
    "BENCHMARKS": "/project/config/benchmarks.toml",
    "LIGHTEVAL_PROJECT": "/project/third_party/lighteval-rwkv",
    "SERVED_MODEL_NAME": "rwkv7-g1l-13.3b-heli-r01-sft",
    "SEED": "42"
  },
  "probe": {"output": "/run/evaluation/probe", "max_samples": 100, "seconds": 3600},
  "full": {"output": "/run/evaluation/full", "seconds": 21600},
  "callback": ["/project/.venv/bin/python", "-m", "helicopter", "checkpoint-result", "/run/spec.json", "train"]
}
```

The callback receives `probe` or `full` as its final argument. Exit 0 proceeds,
exit 1 ends evaluation normally after a failed gate, and other exit codes fail
the job. The destination must be new; publishing never overwrites an existing
checkpoint. Training and each evaluation have separate wall budgets. Native
training behavior remains available by leaving `RWKV_TRAIN_EVAL` unset.

SFT preprocessing accepts `--subset cot` (default), `--subset instruct` and
`--subset mixed`. Instruction rows use the native `fake_think` prefix and
supervise only the concise assistant answer, separator and EOS. CoT rows use
`open_think` and retain their supplied complete reasoning and answer.

## Eight-GPU performance profiles

The defaults below are the recommended hardware profile for one node with eight
96 GB Pro6000 GPUs. They keep the same launchers usable for a deliberate
memory/throughput sweep through environment overrides.

| Method | Default training profile | Rollout/evaluation profile |
| --- | --- | --- |
| AntiDoom / FTPO | 8-rank FSDP, global batch 128 from micro-batch 2 × eight accumulation steps, 32k-token cap | Offline preference training; validation capped at 1,024 rows |
| FlashREINFORCE | Fully async, 4 learner + 4 inference GPUs, batch 128/micro 4, 32k-token trainer cap | 16,384 inference tokens and 64 generation workers per engine; test batch 32/micro 2 |
| OPD | Fully async, 6 learner + 2 inference GPUs, batch 512/micro 4, 32k-token trainer cap | 16,384 inference tokens and 64 workers per engine; colocated mode uses micro 1 in test mode |
| SAO | Fully async, 2 policy + 4 critic + 2 inference GPUs, batch 128/micro 2, 32k-token trainer cap | 16,384 inference tokens and 64 workers per engine; critic makes two passes per rollout |
| SFT | 8-rank FSDP, production batch 128/micro 2, length-grouped sampler with four loader workers | Checkpoint evaluation uses all eight GPUs through `eval/run.sh` |

Async modes are the throughput-oriented defaults for FlashREINFORCE, OPD, and
SAO because rollout is the long-pole stage.  Set `ASYNC_MODE=disabled` when a
fully colocated synchronous comparison is needed; the launchers automatically
lower the shared-memory microbatch where the colocated path needs it.

To construct an RWKV-specific verifiable RL corpus from the unfiltered,
normalized source data used to build UltraData-RL-2609, run
`get_ultradata_rwkv/run.sh`. It evaluates each question with the selected RWKV
checkpoint and removes only questions with a 1.0 RWKV pass rate; all partially
solved and unsolved questions remain available for training.
