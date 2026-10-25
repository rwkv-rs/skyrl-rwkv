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
