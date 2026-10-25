# RWKV FlashREINFORCE

This launcher uses native SkyRL FSDP, `wkv_mode=fp32io16`, and the production
rollout/evaluation sampling policies. The default fully async configuration
dedicates four GPUs to the learner and four TP1 vLLM-RWKV engines to rollout,
so generation overlaps training. Set `ASYNC_MODE=disabled` for the eight-GPU
colocated synchronous baseline, or override `POLICY_GPUS` and
`INFERENCE_GPUS` for another split. Set `USE_NORA=true` to train adapters; the
default is full-parameter training. `--test_mode` prefixes both parquet splits
and runs exactly five optimizer steps.

The production profile is batch 128 with four learner GPUs and four TP1
inference engines. Each learner uses micro-batch 4 and a 32,768-token trainer
microbatch cap; each inference engine admits up to 16,384 batched tokens and the
launcher bounds request fan-out at 64 workers per engine. Test mode uses batch
32, micro-batch 2, and the same length-aware caps. Set `ASYNC_MODE=disabled`
to use all eight GPUs synchronously; the launcher automatically uses micro-batch
1 for that shared allocator path.

```bash
bash examples/train/rwkv/train/flashreinforce/run.sh --test_mode
bash examples/train/rwkv/train/flashreinforce/run.sh
```
